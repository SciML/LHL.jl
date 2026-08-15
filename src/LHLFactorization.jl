"""
    LHLFactorization

The **LHL factorization**: a reduction of a square matrix to upper Hessenberg form by
Gaussian similarity transformations with partial pivoting,

    J = Z H Z⁻¹,   Z = D·P·L

with `L` unit lower triangular (multipliers bounded by 1), `P` a permutation and `D` a
balancing diagonal.  This is Wilkinson's elimination method — EISPACK's `ELMHES` — packaged
for a purpose it is rarely exposed for: **solving a family of shifted systems**

    (σI + τJ) x = b

for many `(σ, τ)` against one reduction.  The shift never reaches `Z`, so

    σI + τJ = Z (σI + τH) Z⁻¹

and `σI + τH` is Hessenberg: a new shift costs `O(n²)`, against `O(n³)` for a fresh LU.
The motivating case is the iteration matrix `W = I - γJ` of a stiff ODE solver, where
adaptive step-size control changes `γ` every step while `J` is held fixed for tens of them.

## Example

```julia
using LHLFactorization, LinearAlgebra

J = randn(400, 400)
ws = lhl(J)                      # O(n³), once
for γ in (0.01, 0.013, 0.02)
    lhl_shift!(ws, 1, -γ)        # O(n²), per shift: builds I - γJ
    x = lhl_ldiv!(copy(b), ws)   # O(n²), per right-hand side
end
```

To drive this from a linear-solver interface rather than by hand, LinearSolve.jl's
`LHLFactorization` wraps it, and consumes `SciMLOperators.WOperator` — the split
`J - M/γ` an implicit ODE solver already builds — as its system matrix.

## Stability

`Z` is not orthogonal, so the backward error carries a factor `κ(Z)` an LU does not have.
Partial pivoting is not optional (without it the method loses every digit on ordinary
matrices), and one step of iterative refinement — see `lhl_refine!` — restores a backward
error comparable to LU's even on the near-nilpotent matrices where `κ(Z)` reaches `10¹⁰`.
"""
module LHLFactorization

using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare

export LHLWorkspace, lhl, lhl!, lhl_reduce!, lhl_shift!, lhl_ldiv!, lhl_refine!,
    applyZ!, applyZinv!

"""
    LHLWorkspace

Storage for one LHL factorization: the reduction of `J` and the LU of the current shifted
Hessenberg.  Build one with [`lhl`](@ref) or [`lhl!`](@ref).

`factors` holds `H` in `triu(factors, -1)` and the step-`k` multipliers in the annihilated
positions `factors[k+2:n, k]`, exactly the way an LU packs its own; `Lp` holds the same
multipliers packed column after column with no gaps (`applyZ!`/`applyZinv!` stream it as
one contiguous array, which the hardware prefetcher follows across column ends).  `Ht` and `Gt` hold
`H` and the LU of `I - γH` **transposed**: the Hessenberg elimination sweeps rows, and in
a column-major array the transposed layout turns every inner loop of the per-γ work
(fused rebuild and elimination, back substitution) into a contiguous one; `rdiag` holds the
reciprocals of the pivots of that LU.  `resid` doubles
as scratch for the reduction and for `lhl_shift!`; `Ht` and `Gt` are scratch during the
reduction, which fills `Ht` last.
"""
mutable struct LHLWorkspace{T, Tr}
    factors::Matrix{T}
    Lp::Vector{T}
    ipiv::Vector{Int}
    scale::Vector{Tr}
    Ht::Matrix{T}
    Gt::Matrix{T}
    rdiag::Vector{T}
    swap::Vector{Bool}
    resid::Vector{T}
    σ::T
    τ::T
    # Whether `factors` holds a valid reduction. A consumer tracking *whose* Jacobian it
    # is must do so itself; this only says one was computed.
    reduced::Bool
    n::Int
    info::Int
end

function LHLWorkspace{T}(n::Integer) where {T}
    Tr = real(T)
    return LHLWorkspace{T, Tr}(
        Matrix{T}(undef, n, n), Vector{T}(undef, _lhl_lpack_len(n)),
        Vector{Int}(undef, max(n - 2, 0)), ones(Tr, n),
        Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Vector{T}(undef, n),
        Vector{Bool}(undef, n), Vector{T}(undef, n), zero(T), zero(T), false, n, 0
    )
end

# Multipliers of step k occupy Lp[_lhl_loff(k, n) .+ (1:n-k-1)] (rows k+2:n).
_lhl_loff(k::Int, n::Int) = (k - 1) * (n - 2) - ((k - 1) * (k - 2)) >> 1
_lhl_lpack_len(n::Int) = n >= 3 ? _lhl_loff(n - 1, n) : 0

function _lhl_resize!(ws::LHLWorkspace{T}, n::Int) where {T}
    n == ws.n && size(ws.factors, 1) == n && return ws
    ws.factors = Matrix{T}(undef, n, n)
    ws.Ht = Matrix{T}(undef, n, n)
    ws.Gt = Matrix{T}(undef, n, n)
    resize!(ws.Lp, _lhl_lpack_len(n))
    resize!(ws.ipiv, max(n - 2, 0))
    resize!(ws.scale, n)
    resize!(ws.rdiag, n)
    resize!(ws.swap, n)
    resize!(ws.resid, n)
    ws.n = n
    ws.reduced = false
    return ws
end

# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

# Parlett–Reinsch scaling: equalize each row/column norm pair by a power of two, which is
# exact in binary floating point and so costs no accuracy.
function _lhl_balance!(A::AbstractMatrix{T}, d::AbstractVector) where {T}
    n = size(A, 1)
    Tr = real(T)
    fill!(d, one(eltype(d)))
    for _ in 1:20
        converged = true
        for i in 1:n
            c = zero(Tr)
            r = zero(Tr)
            @inbounds for j in 1:n
                j == i && continue
                c += abs(A[j, i])
                r += abs(A[i, j])
            end
            (iszero(c) || iszero(r)) && continue
            f = one(Tr)
            s = c + r
            while c < r / 2
                c *= 2
                r /= 2
                f *= 2
            end
            while c >= 2r
                c /= 2
                r *= 2
                f /= 2
            end
            if c + r < Tr(0.95) * s
                converged = false
                d[i] *= f
                @inbounds for j in 1:n
                    A[i, j] /= f
                    A[j, i] *= f
                end
            end
        end
        converged && break
    end
    return A, d
end

# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------

"""
    lhl_reduce!(ws, J, balance) -> ws

Reduce `J` to upper Hessenberg form by Gaussian similarity transformations with partial
pivoting, into `ws`.  `J` is not modified.
"""
function lhl_reduce!(ws::LHLWorkspace{T}, J::AbstractMatrix, balance::Bool) where {T}
    n = LinearAlgebra.checksquare(J)
    _lhl_resize!(ws, n)
    A = ws.factors
    copyto!(A, J)
    if balance
        _lhl_balance!(A, ws.scale)
    else
        fill!(ws.scale, one(eltype(ws.scale)))
    end
    if n >= _lhl_block_min(T)
        _lhl_reduce_blocked!(A, ws.ipiv, ws.Ht, ws.resid, ws.Gt, _lhl_panel_width(n))
    else
        _lhl_reduce_unblocked!(A, ws.ipiv)
    end
    Ht = ws.Ht
    @inbounds for j in 1:n, i in 1:min(j + 1, n)
        Ht[j, i] = A[i, j]
    end
    Lp = ws.Lp
    @inbounds for k in 1:(n - 2)
        o = _lhl_loff(k, n) - k - 1
        @simd for i in (k + 2):n
            Lp[o + i] = A[i, k]
        end
    end
    ws.reduced = true
    return ws
end

function _lhl_reduce_unblocked!(A::AbstractMatrix{T}, ipiv) where {T}
    n = size(A, 1)
    @inbounds for k in 1:(n - 2)
        p = k + 1
        amax = abs(A[k + 1, k])
        for i in (k + 2):n
            a = abs(A[i, k])
            if a > amax
                amax = a
                p = i
            end
        end
        ipiv[k] = p
        if p != k + 1
            for j in 1:n
                A[k + 1, j], A[p, j] = A[p, j], A[k + 1, j]
            end
            for i in 1:n
                A[i, k + 1], A[i, p] = A[i, p], A[i, k + 1]
            end
        end
        piv = A[k + 1, k]
        iszero(piv) && continue
        for i in (k + 2):n
            A[i, k] /= piv
        end
        pk = A[k + 1, k + 1]
        if !iszero(pk)
            @simd for i in (k + 2):n
                A[i, k + 1] -= A[i, k] * pk
            end
        end
        _lhl_trailing_update!(A, k, n)
    end
    return A
end

# Nₖ⁻¹A (rowᵢ ← rowᵢ - lᵢ·row_{k+1}) and then (Nₖ⁻¹A)Nₖ (col_{k+1} ← col_{k+1} +
# Σ lᵢ·colᵢ) over columns k+2:n; column k+1's own left update is already done.  The right
# update must see the columns the left one already transformed, but it needs each column
# only *after* that column is done — so the two sweep the trailing submatrix together, two
# columns at a time so that column k+1 is read and written once per pair.
@inline function _lhl_trailing_update!(A::AbstractMatrix, k::Int, n::Int)
    @inbounds begin
        j = k + 2
        while j + 1 <= n
            pj = A[k + 1, j]
            pj1 = A[k + 1, j + 1]
            @simd for i in (k + 2):n
                l = A[i, k]
                A[i, j] -= l * pj
                A[i, j + 1] -= l * pj1
            end
            vj = A[j, k]
            vj1 = A[j + 1, k]
            @simd for i in 1:n
                A[i, k + 1] += vj * A[i, j] + vj1 * A[i, j + 1]
            end
            j += 2
        end
        if j <= n
            pj = A[k + 1, j]
            @simd for i in (k + 2):n
                A[i, j] -= A[i, k] * pj
            end
            vj = A[j, k]
            @simd for i in 1:n
                A[i, k + 1] += vj * A[i, j]
            end
        end
    end
    return A
end

# The same update with the loops the other way round: a block of rows sweeps every column
# once, holding its multipliers and its column-(k+1) accumulators in registers, so each
# trailing element is loaded and stored once and column k+1 once per row block (the paired
# sweep re-reads column k+1 per column pair).  Rows 1:k+1 only accumulate; rows k+2:n get
# the left update first.  Blocks are 4 vectors of W rows; the block straddling row k+1 and
# the last < W rows (a vector overlapping already-finished rows) go through the masked
# single-vector kernel, whose lanes select between the updated and the untouched value.
function _lhl_trailing_update!(A::Matrix{T}, k::Int, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    n < max(W, 8) && return invoke(_lhl_trailing_update!, Tuple{AbstractMatrix, Int, Int}, A, k, n)
    GC.@preserve A begin
        pA = pointer(A)
        lds = stride(A, 2) * sizeof(T)
        i = _lhl_rb_rows!(Val(false), V, pA, lds, 1, k + 1, k, n)
        if i <= k + 1
            i0 = min(i, n - W + 1)
            _lhl_rb_masked!(V, pA, lds, i0, i, k, n)
            i = i0 + W
        end
        if i <= n
            i = _lhl_rb_rows!(Val(true), V, pA, lds, i, n, k, n)
            i <= n && _lhl_rb_masked!(V, pA, lds, n - W + 1, i, k, n)
        end
    end
    return A
end

@inline _lhl_vadd(a::V, b::V) where {W, T, V <: NTuple{W, VecElement{T}}} =
    ntuple(w -> VecElement(a[w].value + b[w].value), Val(W))
@inline _lhl_vselect(m::NTuple{W, Bool}, a::V, b::V) where {W, T, V <: NTuple{W, VecElement{T}}} =
    ntuple(w -> VecElement(ifelse(m[w], a[w].value, b[w].value)), Val(W))
@inline _lhl_lanemask(::Val{W}, i0::Int, lo::Int) where {W} = ntuple(w -> i0 + w - 1 >= lo, Val(W))

# Whole blocks of rows ia:ib, all with (L = true) or all without the left update; returns
# the first row not covered.
@inline function _lhl_rb_rows!(
        ::Val{L}, ::Type{V}, pA::Ptr{T}, lds::Int, ia::Int, ib::Int, k::Int, n::Int
    ) where {L, W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    vb = W * sz
    prk0 = pA + k * sz + (k + 1) * lds          # A[k+1, k+2]
    pck0 = pA + (k + 1) * sz + (k - 1) * lds    # A[k+2, k]
    i = ia
    while i + 4W - 1 <= ib
        prow = pA + (i - 1) * sz
        pl = prow + (k - 1) * lds
        l1 = _lhl_vload(V, pl); l2 = _lhl_vload(V, pl + vb)
        l3 = _lhl_vload(V, pl + 2vb); l4 = _lhl_vload(V, pl + 3vb)
        s1 = s2 = s3 = s4 = _lhl_bcast(V, zero(T))
        pcol = prow + (k + 1) * lds
        prk = prk0
        pck = pck0
        for _ in (k + 2):n
            vj = _lhl_bcast(V, unsafe_load(pck))
            a1 = _lhl_vload(V, pcol); a2 = _lhl_vload(V, pcol + vb)
            a3 = _lhl_vload(V, pcol + 2vb); a4 = _lhl_vload(V, pcol + 3vb)
            if L
                pj = _lhl_bcast(V, -unsafe_load(prk))
                a1 = _lhl_fma(l1, pj, a1); a2 = _lhl_fma(l2, pj, a2)
                a3 = _lhl_fma(l3, pj, a3); a4 = _lhl_fma(l4, pj, a4)
                _lhl_vstore!(pcol, a1); _lhl_vstore!(pcol + vb, a2)
                _lhl_vstore!(pcol + 2vb, a3); _lhl_vstore!(pcol + 3vb, a4)
            end
            s1 = _lhl_fma(vj, a1, s1); s2 = _lhl_fma(vj, a2, s2)
            s3 = _lhl_fma(vj, a3, s3); s4 = _lhl_fma(vj, a4, s4)
            pcol += lds
            prk += lds
            pck += sz
        end
        pc = prow + k * lds
        _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), s1))
        _lhl_vstore!(pc + vb, _lhl_vadd(_lhl_vload(V, pc + vb), s2))
        _lhl_vstore!(pc + 2vb, _lhl_vadd(_lhl_vload(V, pc + 2vb), s3))
        _lhl_vstore!(pc + 3vb, _lhl_vadd(_lhl_vload(V, pc + 3vb), s4))
        i += 4W
    end
    while i + W - 1 <= ib
        prow = pA + (i - 1) * sz
        l1 = _lhl_vload(V, prow + (k - 1) * lds)
        s1 = _lhl_bcast(V, zero(T))
        pcol = prow + (k + 1) * lds
        prk = prk0
        pck = pck0
        for _ in (k + 2):n
            vj = _lhl_bcast(V, unsafe_load(pck))
            a1 = _lhl_vload(V, pcol)
            if L
                a1 = _lhl_fma(l1, _lhl_bcast(V, -unsafe_load(prk)), a1)
                _lhl_vstore!(pcol, a1)
            end
            s1 = _lhl_fma(vj, a1, s1)
            pcol += lds
            prk += lds
            pck += sz
        end
        pc = prow + k * lds
        _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), s1))
        i += W
    end
    return i
end

# One vector of rows i0:i0+W-1 in which only rows ≥ lo take part, and of those only rows
# ≥ k+2 get the left update; the other lanes are stored back unchanged.
@inline function _lhl_rb_masked!(
        ::Type{V}, pA::Ptr{T}, lds::Int, i0::Int, lo::Int, k::Int, n::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    ma = _lhl_lanemask(Val(W), i0, lo)
    ml = _lhl_lanemask(Val(W), i0, max(lo, k + 2))
    z = _lhl_bcast(V, zero(T))
    prow = pA + (i0 - 1) * sz
    l1 = _lhl_vselect(ml, _lhl_vload(V, prow + (k - 1) * lds), z)
    s1 = z
    pcol = prow + (k + 1) * lds
    prk = pA + k * sz + (k + 1) * lds
    pck = pA + (k + 1) * sz + (k - 1) * lds
    for _ in (k + 2):n
        vj = _lhl_bcast(V, unsafe_load(pck))
        a1 = _lhl_vload(V, pcol)
        a1 = _lhl_vselect(ml, _lhl_fma(l1, _lhl_bcast(V, -unsafe_load(prk)), a1), a1)
        _lhl_vstore!(pcol, a1)
        s1 = _lhl_fma(vj, a1, s1)
        pcol += lds
        prk += lds
        pck += sz
    end
    pc = prow + k * lds
    _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), _lhl_vselect(ma, s1, z)))
    return nothing
end

# Blocked (delayed-update) reduction.  Steps k0..kb form one panel.  Left transforms
# N⁻¹ = I - l e_{k+1}ᵀ and row interchanges are applied to the panel's own columns at once
# but to the trailing block T = A[:, kb+2:n] only at the panel end, as an interchange sweep,
# a small triangular solve on rows k0+1:kb+1 and a rank-nb GEMM on the rest — exactly the
# structure of a blocked LU with the pivot row shifted down by one.  Right transforms
# N = I + l e_{k+1}ᵀ multiply in increasing order to I + Σ l_k e_{k+1}ᵀ, so their effect on
# T is nil and on column k+1 it is A[:, k+2:n]·l_k, which the pivot search of step k+1
# needs immediately: that is a GEMV against T once per step (T is unchanged all panel long,
# so it is a GEMV against the *pre-panel* T, corrected by the panel's own interchanges and
# left transforms afterwards — linear maps commute with the sum).  Rows 1:k0 of the panel
# columns are untouched by the panel's left transforms and interchanges, so their part of
# the right updates is deferred and done as one more GEMM.
#
# A column interchange can pull a trailing column into the panel and push a panel column
# out; the pushed column must land in T in T's (pre-panel) state, so `B0` keeps a pre-panel
# copy of the panel columns.  Interchanges are deferred on the multipliers of earlier panels
# and applied in one column sweep at the end.
# Below `_lhl_block_min` the panel bookkeeping costs more than the GEMM saves; a narrow
# panel wins because the trailing GEMV that dominates is independent of `nb` while the
# per-step panel work grows with it.  With the row-block trailing update the unblocked
# reduction stays ahead up to n ≈ 384 (Float64) / 768 (Float32) on average, but it dips
# well below the blocked one wherever a small multiple of the column stride is within a
# vector of a multiple of 4 KiB (its stores and loads a few columns apart then alias in the
# low address bits): the deep dips start at n ≈ 249 (Float64) and n ≈ 500 (Float32), and
# the crossover sits just below them.
_lhl_block_min(::Type{Float64}) = 248
_lhl_block_min(::Type{Float32}) = 496
_lhl_block_min(::Type{T}) where {T} = sizeof(real(T)) >= 8 ? 160 : 320
_lhl_panel_width(n::Int) = n < 512 ? 12 : 16

function _lhl_reduce_blocked!(
        A::AbstractMatrix{T}, ipiv, B0::AbstractMatrix{T},
        w::AbstractVector{T}, pack::AbstractArray{T}, nb::Int
    ) where {T}
    n = size(A, 1)
    @inbounds for k0 in 1:nb:(n - 2)
        kb = min(k0 + nb - 1, n - 2)
        for j in (k0 + 1):(kb + 1)
            @simd for i in 1:n
                B0[i, j - k0] = A[i, j]
            end
        end
        for k in k0:kb
            p = k + 1
            amax = abs(A[k + 1, k])
            for i in (k + 2):n
                a = abs(A[i, k])
                if a > amax
                    amax = a
                    p = i
                end
            end
            ipiv[k] = p
            if p != k + 1
                for j in k0:(kb + 1)
                    A[k + 1, j], A[p, j] = A[p, j], A[k + 1, j]
                end
                if p <= kb + 1
                    for i in 1:n
                        A[i, k + 1], A[i, p] = A[i, p], A[i, k + 1]
                        B0[i, k + 1 - k0], B0[i, p - k0] = B0[i, p - k0], B0[i, k + 1 - k0]
                    end
                else
                    for i in 1:n
                        t = A[i, p]
                        A[i, p] = B0[i, k + 1 - k0]
                        B0[i, k + 1 - k0] = t
                        A[i, k + 1] = t
                    end
                    for kk in k0:k
                        q = ipiv[kk]
                        q != kk + 1 && ((A[kk + 1, k + 1], A[q, k + 1]) = (A[q, k + 1], A[kk + 1, k + 1]))
                    end
                    for kk in k0:(k - 1)
                        x = A[kk + 1, k + 1]
                        iszero(x) && continue
                        @simd for i in (kk + 2):n
                            A[i, k + 1] -= A[i, kk] * x
                        end
                    end
                end
            end
            piv = A[k + 1, k]
            iszero(piv) && continue
            for i in (k + 2):n
                A[i, k] /= piv
            end
            for j in (k + 1):(kb + 1)
                pj = A[k + 1, j]
                iszero(pj) && continue
                @simd for i in (k + 2):n
                    A[i, j] -= A[i, k] * pj
                end
            end
            for j in (k + 2):(kb + 1)
                vj = A[j, k]
                iszero(vj) && continue
                @simd for i in (k0 + 1):n
                    A[i, k + 1] += vj * A[i, j]
                end
            end
            if kb + 2 <= n
                for i in (k0 + 1):n
                    w[i] = zero(T)
                end
                _lhl_gemv_cols!(w, A, k, k0 + 1, n, kb + 2, n)
                for kk in k0:k
                    q = ipiv[kk]
                    q != kk + 1 && ((w[kk + 1], w[q]) = (w[q], w[kk + 1]))
                end
                for kk in k0:k
                    x = w[kk + 1]
                    iszero(x) && continue
                    @simd for i in (kk + 2):n
                        w[i] -= A[i, kk] * x
                    end
                end
                @simd for i in (k0 + 1):n
                    A[i, k + 1] += w[i]
                end
            end
        end
        if kb + 2 <= n
            for c in (kb + 2):n, k in k0:kb
                p = ipiv[k]
                p != k + 1 && ((A[k + 1, c], A[p, c]) = (A[p, c], A[k + 1, c]))
            end
            _lhl_trsm_block!(A, k0, kb, pack)
            _lhl_gemm!(A, kb + 2, n, k0, kb, k0 + 1, kb + 2, kb + 2, n, -one(T), pack)
        end
        for k in k0:kb, j in (k + 2):(kb + 1)
            vj = A[j, k]
            iszero(vj) && continue
            @simd for i in 1:k0
                A[i, k + 1] += vj * A[i, j]
            end
        end
        for kk in (kb + 2):nb:n
            ke = min(kk + nb - 1, n)
            _lhl_gemm!(A, 1, k0, kk, ke, kk, k0, k0 + 1, kb + 1, one(T), pack)
        end
    end
    @inbounds for k0 in 1:nb:(n - 2)
        kb = min(k0 + nb - 1, n - 2)
        for c in k0:kb, k in (kb + 1):(n - 2)
            p = ipiv[k]
            p != k + 1 && ((A[k + 1, c], A[p, c]) = (A[p, c], A[k + 1, c]))
        end
    end
    return A
end

# w[r0:r1] += Σ_{j=c0}^{c1} A[j, k] * A[r0:r1, j]
function _lhl_gemv_cols!(w, A::AbstractMatrix{T}, k::Int, r0::Int, r1::Int, c0::Int, c1::Int) where {T}
    j = c0
    @inbounds while j + 3 <= c1
        l1 = A[j, k]; l2 = A[j + 1, k]; l3 = A[j + 2, k]; l4 = A[j + 3, k]
        @simd for i in r0:r1
            w[i] += (l1 * A[i, j] + l2 * A[i, j + 1]) + (l3 * A[i, j + 2] + l4 * A[i, j + 3])
        end
        j += 4
    end
    @inbounds while j <= c1
        lj = A[j, k]
        if !iszero(lj)
            @simd for i in r0:r1
                w[i] += lj * A[i, j]
            end
        end
        j += 1
    end
    return w
end

# A[i0:i1, c0:c1] += sgn * A[i0:i1, j0:j1] * A[r0:r0+K-1, cB:cB+(c1-c0)],  K = j1 - j0 + 1.
# The three blocks must not overlap.
function _lhl_gemm!(
        A::AbstractMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack
    ) where {T}
    (i1 < i0 || j1 < j0 || c1 < c0) && return nothing
    @inbounds for c in c0:c1, (jj, j) in enumerate(j0:j1)
        b = sgn * A[r0 + jj - 1, cB + c - c0]
        iszero(b) && continue
        @simd for i in i0:i1
            A[i, c] += A[i, j] * b
        end
    end
    return nothing
end

function _lhl_gemm!(
        A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}
    ) where {T <: Union{Float32, Float64}}
    (i1 < i0 || j1 < j0 || c1 < c0) && return nothing
    if stride(A, 1) == 1
        _lhl_gemm_micro!(A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack)
    else
        invoke(
            _lhl_gemm!, Tuple{AbstractMatrix{T}, Int, Int, Int, Int, Int, Int, Int, Int, T, Any},
            A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack
        )
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Register-blocked GEMM microkernel (Base only): NTuple{W,VecElement} + llvm.fmuladd, after
# LinearSolve.jl's blocked_lufact.jl.  Holds a 3W×4 tile of C in registers over the whole
# K loop; P (the multiplier panel) is packed with the sign folded in so the tile streams it
# with unit stride and no leading-dimension aliasing.
# ---------------------------------------------------------------------------
const _LHL_VEC_BYTES = Sys.ARCH === :x86_64 ? 32 : 16

for (T, sfx) in ((Float64, "f64"), (Float32, "f32")), W in (2, 4, 8, 16)
    V = NTuple{W, VecElement{T}}
    @eval @inline _lhl_fma(a::$V, b::$V, c::$V) = ccall(
        $("llvm.fmuladd.v$(W)$(sfx)"), llvmcall, $V, ($V, $V, $V), a, b, c
    )
end
@inline _lhl_vectype(::Type{T}) where {T} = NTuple{_LHL_VEC_BYTES ÷ sizeof(T), VecElement{T}}
@inline _lhl_vload(::Type{V}, p::Ptr) where {V} = unsafe_load(Ptr{V}(p))
@inline _lhl_vstore!(p::Ptr, v::V) where {V} = unsafe_store!(Ptr{V}(p), v)
@inline _lhl_bcast(::Type{NTuple{W, VecElement{T}}}, x::T) where {W, T} =
    ntuple(_ -> VecElement(x), Val(W))

# C[ib:ie, c0:c1] += P * B one vector (or scalar) at a time; packed row i - ibase + 1 of P
# holds C row i, and column c of B sits at pB + (c - 1) * ldb.
@inline function _lhl_micro_edge!(
        ::Type{V}, pC::Ptr{T}, ldc::Int, pP::Ptr{T}, ldp::Int, pB::Ptr{T}, ldb::Int,
        K::Int, ibase::Int, ib::Int, ie::Int, c0::Int, c1::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    ib > ie && return nothing
    sz = sizeof(T)
    ldcs = ldc * sz
    ldps = ldp * sz
    ldbs = ldb * sz
    for c in c0:c1
        pcol = pC + (c - 1) * ldcs
        pb0 = pB + (c - 1) * ldbs
        i = ib
        while i + W - 1 <= ie
            q = pcol + (i - 1) * sz
            acc = _lhl_vload(V, q)
            pk = pP + (i - ibase) * sz
            pb = pb0
            for _ in 1:K
                acc = _lhl_fma(_lhl_vload(V, pk), _lhl_bcast(V, unsafe_load(pb)), acc)
                pk += ldps
                pb += sz
            end
            _lhl_vstore!(q, acc)
            i += W
        end
        while i <= ie
            q = pcol + (i - 1) * sz
            s = unsafe_load(q)
            pk = pP + (i - ibase) * sz
            pb = pb0
            for _ in 1:K
                s = muladd(unsafe_load(pk), unsafe_load(pb), s)
                pk += ldps
                pb += sz
            end
            unsafe_store!(q, s)
            i += 1
        end
    end
    return nothing
end

# Whole 3W×4 tiles of C[ib:ie, c0:c1] += P * B; returns the first column no tile covered.
@inline function _lhl_micro_tile!(
        ::Type{V}, pC::Ptr{T}, ldc::Int, pP::Ptr{T}, ldp::Int, pB::Ptr{T}, ldb::Int,
        K::Int, ibase::Int, ib::Int, ie::Int, c0::Int, c1::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    ldcs = ldc * sz
    ldps = ldp * sz
    ldbs = ldb * sz
    vb = W * sz
    c = c0
    while c + 3 <= c1
        pcol = pC + (c - 1) * ldcs
        pb0 = pB + (c - 1) * ldbs
        i = ib
        while i + 3W - 1 <= ie
            q1 = pcol + (i - 1) * sz
            q2 = q1 + ldcs
            q3 = q2 + ldcs
            q4 = q3 + ldcs
            t11 = _lhl_vload(V, q1); t21 = _lhl_vload(V, q1 + vb); t31 = _lhl_vload(V, q1 + 2vb)
            t12 = _lhl_vload(V, q2); t22 = _lhl_vload(V, q2 + vb); t32 = _lhl_vload(V, q2 + 2vb)
            t13 = _lhl_vload(V, q3); t23 = _lhl_vload(V, q3 + vb); t33 = _lhl_vload(V, q3 + 2vb)
            t14 = _lhl_vload(V, q4); t24 = _lhl_vload(V, q4 + vb); t34 = _lhl_vload(V, q4 + 2vb)
            pk = pP + (i - ibase) * sz
            pb = pb0
            for _ in 1:K
                p1 = _lhl_vload(V, pk)
                p2 = _lhl_vload(V, pk + vb)
                p3 = _lhl_vload(V, pk + 2vb)
                b = _lhl_bcast(V, unsafe_load(pb))
                t11 = _lhl_fma(p1, b, t11); t21 = _lhl_fma(p2, b, t21); t31 = _lhl_fma(p3, b, t31)
                b = _lhl_bcast(V, unsafe_load(pb + ldbs))
                t12 = _lhl_fma(p1, b, t12); t22 = _lhl_fma(p2, b, t22); t32 = _lhl_fma(p3, b, t32)
                b = _lhl_bcast(V, unsafe_load(pb + 2ldbs))
                t13 = _lhl_fma(p1, b, t13); t23 = _lhl_fma(p2, b, t23); t33 = _lhl_fma(p3, b, t33)
                b = _lhl_bcast(V, unsafe_load(pb + 3ldbs))
                t14 = _lhl_fma(p1, b, t14); t24 = _lhl_fma(p2, b, t24); t34 = _lhl_fma(p3, b, t34)
                pk += ldps
                pb += sz
            end
            _lhl_vstore!(q1, t11); _lhl_vstore!(q1 + vb, t21); _lhl_vstore!(q1 + 2vb, t31)
            _lhl_vstore!(q2, t12); _lhl_vstore!(q2 + vb, t22); _lhl_vstore!(q2 + 2vb, t32)
            _lhl_vstore!(q3, t13); _lhl_vstore!(q3 + vb, t23); _lhl_vstore!(q3 + 2vb, t33)
            _lhl_vstore!(q4, t14); _lhl_vstore!(q4 + vb, t24); _lhl_vstore!(q4 + 2vb, t34)
            i += 3W
        end
        c += 4
    end
    return c
end

# One W×4 tile of C[ib:ib+W-1, c0:c1] += P * B (the last < 3W rows of a block, four
# independent accumulators instead of the edge kernel's single chain); returns the first
# column no tile covered.
@inline function _lhl_micro_tile1!(
        ::Type{V}, pC::Ptr{T}, ldc::Int, pP::Ptr{T}, ldp::Int, pB::Ptr{T}, ldb::Int,
        K::Int, ibase::Int, ib::Int, c0::Int, c1::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    ldcs = ldc * sz
    ldps = ldp * sz
    ldbs = ldb * sz
    c = c0
    while c + 3 <= c1
        q1 = pC + (c - 1) * ldcs + (ib - 1) * sz
        q2 = q1 + ldcs
        q3 = q2 + ldcs
        q4 = q3 + ldcs
        t1 = _lhl_vload(V, q1); t2 = _lhl_vload(V, q2); t3 = _lhl_vload(V, q3); t4 = _lhl_vload(V, q4)
        pk = pP + (ib - ibase) * sz
        pb = pB + (c - 1) * ldbs
        for _ in 1:K
            p1 = _lhl_vload(V, pk)
            t1 = _lhl_fma(p1, _lhl_bcast(V, unsafe_load(pb)), t1)
            t2 = _lhl_fma(p1, _lhl_bcast(V, unsafe_load(pb + ldbs)), t2)
            t3 = _lhl_fma(p1, _lhl_bcast(V, unsafe_load(pb + 2ldbs)), t3)
            t4 = _lhl_fma(p1, _lhl_bcast(V, unsafe_load(pb + 3ldbs)), t4)
            pk += ldps
            pb += sz
        end
        _lhl_vstore!(q1, t1); _lhl_vstore!(q2, t2); _lhl_vstore!(q3, t3); _lhl_vstore!(q4, t4)
        c += 4
    end
    return c
end

function _lhl_gemm_micro!(
        A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}
    ) where {T <: Union{Float32, Float64}}
    K = j1 - j0 + 1
    rows = i1 - i0 + 1
    ldp = rows % 256 == 0 ? rows + 4 : rows
    length(pack) >= ldp * K || throw(ArgumentError("LHL gemm scratch too small"))
    @inbounds for (jj, j) in enumerate(j0:j1)
        off = (jj - 1) * ldp - i0 + 1
        @simd for i in i0:i1
            pack[off + i] = sgn * A[i, j]
        end
    end
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    mr = 3W
    rowblock = 96
    ld = stride(A, 2)
    sz = sizeof(T)
    GC.@preserve A pack begin
        pA = pointer(A)
        pP = pointer(pack)
        pB = pA + ((cB - c0) * ld + (r0 - 1)) * sz
        ib = i0
        while ib <= i1
            ie = min(ib + rowblock - 1, i1)
            rfull = ib + ((ie - ib + 1) ÷ mr) * mr - 1
            ct = _lhl_micro_tile!(V, pA, ld, pP, ldp, pB, ld, K, i0, ib, rfull, c0, c1)
            _lhl_micro_edge!(V, pA, ld, pP, ldp, pB, ld, K, i0, rfull + 1, ie, c0, c1)
            _lhl_micro_edge!(V, pA, ld, pP, ldp, pB, ld, K, i0, ib, rfull, ct, c1)
            ib = ie + 1
        end
    end
    return nothing
end


# Rows k0+1:kb+1 of the trailing block T = A[:, kb+2:n] ← M⁻¹ · rows, M the unit lower
# triangle of the panel's multipliers (M[i, k+1] = A[i, k] for i ≥ k+2, i.e. the panel's left
# transforms restricted to its own rows).  Row k0+1 is unchanged.
function _lhl_trsm_block!(A::AbstractMatrix{T}, k0::Int, kb::Int, pack) where {T}
    n = size(A, 1)
    @inbounds for c in (kb + 2):n, k in k0:kb
        x = A[k + 1, c]
        iszero(x) && continue
        for i in (k + 2):(kb + 1)
            A[i, c] -= A[i, k] * x
        end
    end
    return A
end

# Same as a GEMM: rows k0+1:kb+1 += (M⁻¹ - I) · rows k0+1:kb+1, with the small inverse formed
# explicitly (nb³/6) and the source rows copied out first (they are the destination).  Both
# live in `pack`, in the microkernel's packed-P / B layouts.
function _lhl_trsm_block!(
        A::StridedMatrix{T}, k0::Int, kb::Int, pack::Array{T}
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 1)
    nb = kb - k0 + 1
    ncol = n - kb - 1
    if stride(A, 1) != 1 || nb < 2 || length(pack) < nb * nb + nb * ncol
        return invoke(_lhl_trsm_block!, Tuple{AbstractMatrix{T}, Int, Int, Any}, A, k0, kb, pack)
    end
    # P = M⁻¹ - I, column-major with leading dimension nb; M[r, s] = A[k0+r, k0+s-1] for r > s.
    @inbounds for q in 1:nb
        for r in 1:q
            pack[r + (q - 1) * nb] = zero(T)
        end
        for r in (q + 1):nb
            acc = -A[k0 + r, k0 + q - 1]
            for s in (q + 1):(r - 1)
                acc -= A[k0 + r, k0 + s - 1] * pack[s + (q - 1) * nb]
            end
            pack[r + (q - 1) * nb] = acc
        end
    end
    off = nb * nb
    @inbounds for c in (kb + 2):n
        o = off + (c - kb - 2) * nb - k0
        @simd for i in (k0 + 1):(kb + 1)
            pack[o + i] = A[i, c]
        end
    end
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    ld = stride(A, 2)
    GC.@preserve A pack begin
        pA = pointer(A)
        pP = pointer(pack)
        pB = pointer(pack) + (off - (kb + 1) * nb) * sz
        ib = k0 + 1
        ie = kb + 1
        rfull = ib + (nb ÷ (3W)) * 3W - 1
        ct = _lhl_micro_tile!(V, pA, ld, pP, nb, pB, nb, nb, ib, ib, rfull, kb + 2, n)
        _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, ib, rfull, ct, n)
        r = rfull + 1
        while r + W - 1 <= ie
            ct = _lhl_micro_tile1!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, kb + 2, n)
            _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, r + W - 1, ct, n)
            r += W
        end
        _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, ie, kb + 2, n)
    end
    return A
end

"""
    applyZ!(x, ws)

`x ← Z x`.  `n²/2` multiply–adds.
"""
function applyZ!(x::AbstractVector, ws::LHLWorkspace)
    n = ws.n
    _lhl_zsweep!(x, ws.Lp, n)
    @inbounds for k in (n - 2):-1:1
        p = ws.ipiv[k]
        p != k + 1 && ((x[k + 1], x[p]) = (x[p], x[k + 1]))
    end
    d = ws.scale
    @inbounds @simd for i in 1:n
        x[i] *= d[i]
    end
    return x
end

"""
    applyZinv!(x, ws)

`x ← Z⁻¹x`.  `n²/2` multiply–adds.
"""
function applyZinv!(x::AbstractVector, ws::LHLWorkspace)
    n = ws.n
    d = ws.scale
    @inbounds @simd for i in 1:n
        x[i] /= d[i]
    end
    @inbounds for k in 1:(n - 2)
        p = ws.ipiv[k]
        p != k + 1 && ((x[k + 1], x[p]) = (x[p], x[k + 1]))
    end
    _lhl_zinvsweep!(x, ws.Lp, n)
    return x
end

# x ← L x, four columns per sweep.  Column k uses x[k+1], which columns k-1, k-2, k-3
# (applied after it) do not touch, so all four scalars are read up front and only rows
# k-1:k+1 need the sequential order.
function _lhl_zsweep!(x::AbstractVector, Lp::AbstractVector{T}, n::Int) where {T}
    k = n - 2
    @inbounds while k - 3 >= 1
        o1 = _lhl_loff(k, n) - k - 1
        o2 = _lhl_loff(k - 1, n) - k
        o3 = _lhl_loff(k - 2, n) - k + 1
        o4 = _lhl_loff(k - 3, n) - k + 2
        x1 = x[k + 1]
        x2 = x[k]
        x3 = x[k - 1]
        x4 = x[k - 2]
        x[k - 1] += Lp[o4 + k - 1] * x4
        x[k] += Lp[o3 + k] * x3 + Lp[o4 + k] * x4
        x[k + 1] += (Lp[o2 + k + 1] * x2 + Lp[o3 + k + 1] * x3) + Lp[o4 + k + 1] * x4
        @simd for i in (k + 2):n
            x[i] += (Lp[o1 + i] * x1 + Lp[o2 + i] * x2) + (Lp[o3 + i] * x3 + Lp[o4 + i] * x4)
        end
        k -= 4
    end
    @inbounds while k >= 1
        xk = x[k + 1]
        o1 = _lhl_loff(k, n) - k - 1
        @simd for i in (k + 2):n
            x[i] += Lp[o1 + i] * xk
        end
        k -= 1
    end
    return x
end

# x ← L⁻¹ x, four columns per sweep; x[k+2:k+4] are finished sequentially first because
# each column's scalar depends on the previous column's update.
function _lhl_zinvsweep!(x::AbstractVector, Lp::AbstractVector{T}, n::Int) where {T}
    k = 1
    @inbounds while k + 3 <= n - 2
        o1 = _lhl_loff(k, n) - k - 1
        o2 = _lhl_loff(k + 1, n) - k - 2
        o3 = _lhl_loff(k + 2, n) - k - 3
        o4 = _lhl_loff(k + 3, n) - k - 4
        x1 = x[k + 1]
        x2 = x[k + 2] - Lp[o1 + k + 2] * x1
        x[k + 2] = x2
        x3 = x[k + 3] - (Lp[o1 + k + 3] * x1 + Lp[o2 + k + 3] * x2)
        x[k + 3] = x3
        x4 = x[k + 4] - ((Lp[o1 + k + 4] * x1 + Lp[o2 + k + 4] * x2) + Lp[o3 + k + 4] * x3)
        x[k + 4] = x4
        @simd for i in (k + 5):n
            x[i] -= (Lp[o1 + i] * x1 + Lp[o2 + i] * x2) + (Lp[o3 + i] * x3 + Lp[o4 + i] * x4)
        end
        k += 4
    end
    @inbounds while k <= n - 2
        xk = x[k + 1]
        o1 = _lhl_loff(k, n) - k - 1
        @simd for i in (k + 2):n
            x[i] -= Lp[o1 + i] * xk
        end
        k += 1
    end
    return x
end

# Explicit W-wide kernels for the two sweeps: same rank-4 structure, but a single vector
# loop with a ≤W-1 scalar tail and no alias check per column block, which is what the
# short trip counts at n ≤ 256 need.
function _lhl_zsweep!(x::Vector{T}, Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    k = n - 2
    GC.@preserve x Lp begin
        px = pointer(x)
        pL = pointer(Lp)
        @inbounds while k - 3 >= 1
            c1 = pL + _lhl_loff(k, n) * sz
            c2 = pL + _lhl_loff(k - 1, n) * sz
            c3 = pL + _lhl_loff(k - 2, n) * sz
            c4 = pL + _lhl_loff(k - 3, n) * sz
            x1 = x[k + 1]
            x2 = x[k]
            x3 = x[k - 1]
            x4 = x[k - 2]
            x[k - 1] = muladd(unsafe_load(c4), x4, x[k - 1])
            x[k] = muladd(unsafe_load(c3), x3, muladd(unsafe_load(c4, 2), x4, x[k]))
            x[k + 1] = muladd(
                unsafe_load(c2), x2, muladd(unsafe_load(c3, 2), x3, muladd(unsafe_load(c4, 3), x4, x[k + 1]))
            )
            b1 = _lhl_bcast(V, x1)
            b2 = _lhl_bcast(V, x2)
            b3 = _lhl_bcast(V, x3)
            b4 = _lhl_bcast(V, x4)
            i = k + 2
            p1 = c1
            p2 = c2 + sz
            p3 = c3 + 2sz
            p4 = c4 + 3sz
            q = px + (i - 1) * sz
            while i + W - 1 <= n
                v = _lhl_vload(V, q)
                v = _lhl_fma(_lhl_vload(V, p1), b1, v)
                v = _lhl_fma(_lhl_vload(V, p2), b2, v)
                v = _lhl_fma(_lhl_vload(V, p3), b3, v)
                v = _lhl_fma(_lhl_vload(V, p4), b4, v)
                _lhl_vstore!(q, v)
                p1 += W * sz; p2 += W * sz; p3 += W * sz; p4 += W * sz; q += W * sz
                i += W
            end
            while i <= n
                v = unsafe_load(q)
                v = muladd(unsafe_load(p1), x1, v)
                v = muladd(unsafe_load(p2), x2, v)
                v = muladd(unsafe_load(p3), x3, v)
                v = muladd(unsafe_load(p4), x4, v)
                unsafe_store!(q, v)
                p1 += sz; p2 += sz; p3 += sz; p4 += sz; q += sz
                i += 1
            end
            k -= 4
        end
        @inbounds while k >= 1
            xk = x[k + 1]
            o1 = _lhl_loff(k, n) - k - 1
            @simd for i in (k + 2):n
                x[i] = muladd(Lp[o1 + i], xk, x[i])
            end
            k -= 1
        end
    end
    return x
end

function _lhl_zinvsweep!(x::Vector{T}, Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    k = 1
    GC.@preserve x Lp begin
        px = pointer(x)
        pL = pointer(Lp)
        @inbounds while k + 3 <= n - 2
            c1 = pL + _lhl_loff(k, n) * sz
            c2 = pL + _lhl_loff(k + 1, n) * sz
            c3 = pL + _lhl_loff(k + 2, n) * sz
            c4 = pL + _lhl_loff(k + 3, n) * sz
            x1 = x[k + 1]
            x2 = muladd(-unsafe_load(c1), x1, x[k + 2])
            x[k + 2] = x2
            x3 = muladd(-unsafe_load(c2), x2, muladd(-unsafe_load(c1, 2), x1, x[k + 3]))
            x[k + 3] = x3
            x4 = muladd(
                -unsafe_load(c3), x3, muladd(-unsafe_load(c2, 2), x2, muladd(-unsafe_load(c1, 3), x1, x[k + 4]))
            )
            x[k + 4] = x4
            b1 = _lhl_bcast(V, -x1)
            b2 = _lhl_bcast(V, -x2)
            b3 = _lhl_bcast(V, -x3)
            b4 = _lhl_bcast(V, -x4)
            i = k + 5
            p1 = c1 + 3sz
            p2 = c2 + 2sz
            p3 = c3 + sz
            p4 = c4
            q = px + (i - 1) * sz
            while i + W - 1 <= n
                v = _lhl_vload(V, q)
                v = _lhl_fma(_lhl_vload(V, p1), b1, v)
                v = _lhl_fma(_lhl_vload(V, p2), b2, v)
                v = _lhl_fma(_lhl_vload(V, p3), b3, v)
                v = _lhl_fma(_lhl_vload(V, p4), b4, v)
                _lhl_vstore!(q, v)
                p1 += W * sz; p2 += W * sz; p3 += W * sz; p4 += W * sz; q += W * sz
                i += W
            end
            while i <= n
                v = unsafe_load(q)
                v = muladd(-unsafe_load(p1), x1, v)
                v = muladd(-unsafe_load(p2), x2, v)
                v = muladd(-unsafe_load(p3), x3, v)
                v = muladd(-unsafe_load(p4), x4, v)
                unsafe_store!(q, v)
                p1 += sz; p2 += sz; p3 += sz; p4 += sz; q += sz
                i += 1
            end
            k += 4
        end
        @inbounds while k <= n - 2
            xk = x[k + 1]
            o1 = _lhl_loff(k, n) - k - 1
            @simd for i in (k + 2):n
                x[i] = muladd(-Lp[o1 + i], xk, x[i])
            end
            k += 1
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# The γ-dependent half
# ---------------------------------------------------------------------------

"""
    lhl_shift!(ws, σ, τ) -> ws

Form `G = σI + τH` and LU-factorize it (partial pivoting) into `ws.Gt`, transposed.
`≈n²` multiply–adds.  `(σ, τ) = (1, -γ)` gives `I - γJ`; `(0, 1)` gives `J` itself.

An upper Hessenberg matrix has one subdiagonal, so each elimination step chooses between
two candidates and the element growth factor is bounded by `n` — far tighter than the
`2ⁿ⁻¹` of general partial pivoting.
"""
function lhl_shift!(ws::LHLWorkspace{T}, σ, τ) where {T}
    n = ws.n
    σ = convert(T, σ)
    τ = convert(T, τ)
    n == 0 && return ws
    if T <: Union{Float32, Float64} && n >= _LHL_SHIFT_FUSED_MIN
        # Columns a multiple of 4 KiB apart map to one L1 set: four of `Ht` plus four of `Gt`
        # fill its eight ways and the fused pass thrashes, so those strides get two rows.
        if (n * sizeof(T)) % 4096 == 0
            info = _lhl_shift_fused!(Val(2), ws, σ, τ)
        else
            info = _lhl_shift_fused!(Val(4), ws, σ, τ)
        end
    else
        info = _lhl_shift_rows!(ws, σ, τ)
    end
    ws.info = info
    return ws
end

const _LHL_SHIFT_FUSED_MIN = 512

# Row k of G is formed from Ht only when it enters the elimination, and the row that has
# not yet been chosen as a pivot row lives in `r`: at step k the candidates are the
# pending row (`r`, currently row k) and the fresh row k+1, whichever wins is written to
# Gt[:, k] as row k of U, and the loser minus its multiple becomes the new pending row.
# No row of G is ever copied twice and no interchange is ever performed on storage.
@inline function _lhl_shift_rows!(ws::LHLWorkspace{T}, σ::T, τ::T) where {T}
    Ht = ws.Ht
    Gt = ws.Gt
    swap = ws.swap
    r = ws.resid
    n = ws.n
    @inbounds begin
        @simd for j in 1:n
            r[j] = τ * Ht[j, 1]
        end
        r[1] += σ
        info = 0
        for k in 1:(n - 1)
            a = r[k]
            b = τ * Ht[k, k + 1]
            if abs(b) > abs(a)
                swap[k] = true
                Gt[k, k] = b
                l = a / b
                Gt[k, k + 1] = l
                @simd for j in (k + 1):n
                    g = τ * Ht[j, k + 1]
                    Gt[j, k] = g
                    r[j] -= l * g
                end
                Gt[k + 1, k] += σ
                r[k + 1] -= l * σ
            else
                swap[k] = false
                Gt[k, k] = a
                if iszero(a)
                    info == 0 && (info = k)
                    l = zero(T)
                else
                    l = b / a
                end
                Gt[k, k + 1] = l
                @simd for j in (k + 1):n
                    rj = r[j]
                    Gt[j, k] = rj
                    r[j] = τ * Ht[j, k + 1] - l * rj
                end
                r[k + 1] += σ
            end
        end
        Gt[n, n] = r[n]
        swap[n] = false
        iszero(Gt[n, n]) && info == 0 && (info = n)
        rd = ws.rdiag
        @simd for j in 1:n
            rd[j] = inv(Gt[j, j])
        end
    end
    return info
end

# The same elimination, R steps per pass over the pending row: element j reads Ht[j, k+1:k+R]
# and r[j] once and writes Gt[j, k:k+R-1] and r[j] once, R+1 loads and R+1 stores instead
# of 2R and 2R.  Elements k+1..k+R form a triangle (element k+m takes steps k..k+m-1 and
# then decides step k+m), the rest take all R steps in `_lhl_shift_pass!`.  Every element
# sees exactly the operations of `_lhl_shift_rows!` in the same order.
@inline function _lhl_shift_decide(a::T, b::T, info::Int, k::Int) where {T}
    s = abs(b) > abs(a)
    if s
        l = a / b
    elseif iszero(a)
        info == 0 && (info = k)
        l = zero(T)
    else
        l = b / a
    end
    return s, l, info
end

# (a closure over `s`/`l`, which the loop below reassigns, would box them)
@inline _lhl_fill(::Val{R}, x) where {R} = ntuple(_ -> x, Val(R))

@inline function _lhl_shift_step(s::Bool, l::T, τ::T, h::T, rj::T) where {T}
    g = τ * h
    return ifelse(s, g, rj), ifelse(s, rj - l * g, g - l * rj)
end

# Steps k..k+R-1 with decisions S on elements j0:n; S is a compile-time tuple so that each
# pivot pattern gets its own branch-free loop.  All loads precede all stores in the body:
# a store to Gt[j, k] followed by a load of Ht[j, k+2] a 4 KiB multiple away stalls otherwise.
@generated function _lhl_shift_pass!(
        ::Val{S}, L::NTuple{R, T}, Ht, Gt, r, n::Int, τ::T, k::Int, j0::Int
    ) where {S, R, T}
    loads = Expr(:block)
    steps = Expr(:block)
    stores = Expr(:block)
    for i in 1:R
        h = Symbol(:h_, i)
        u = Symbol(:u_, i)
        push!(loads.args, :($h = Ht[j, k + $i]))
        push!(steps.args, :(($u, rj) = _lhl_shift_step($(S[i]), L[$i], τ, $h, rj)))
        push!(stores.args, :(Gt[j, k + $(i - 1)] = $u))
    end
    return quote
        @inbounds @simd ivdep for j in j0:n
            rj = r[j]
            $loads
            $steps
            $stores
            r[j] = rj
        end
        return nothing
    end
end

@generated function _lhl_shift_pass!(S::NTuple{R, Bool}, L, Ht, Gt, r, n, τ, k, j0) where {R}
    ex = :(_lhl_shift_pass!(Val($(ntuple(_ -> false, R))), L, Ht, Gt, r, n, τ, k, j0))
    for idx in 1:(2^R - 1)
        pat = ntuple(i -> (idx >> (i - 1)) & 1 == 1, R)
        ex = :(idx == $idx ? _lhl_shift_pass!(Val($pat), L, Ht, Gt, r, n, τ, k, j0) : $ex)
    end
    sel = Expr(:block, :(idx = 0))
    for i in 1:R
        push!(sel.args, :(idx |= Int(S[$i]) << $(i - 1)))
    end
    return quote
        $sel
        $ex
    end
end

@inline function _lhl_shift_block!(
        ::Val{R}, Ht, Gt, swap, r, n::Int, σ::T, τ::T, k::Int, info::Int
    ) where {R, T}
    @inbounds begin
        b = τ * Ht[k, k + 1]
        s, l, info = _lhl_shift_decide(r[k], b, info, k)
        swap[k] = s
        Gt[k, k] = ifelse(s, b, r[k])
        Gt[k, k + 1] = l
        S = _lhl_fill(Val(R), s)
        L = _lhl_fill(Val(R), l)
        for m in 1:R
            j = k + m
            rj = r[j]
            for i in 1:m
                s = S[i]
                l = L[i]
                g = τ * Ht[j, k + i]
                if s
                    u = g
                    rj -= l * g
                    if i == m
                        u += σ
                        rj -= l * σ
                    end
                else
                    u = rj
                    rj = g - l * rj
                    i == m && (rj += σ)
                end
                Gt[j, k + i - 1] = u
            end
            r[j] = rj
            if m < R
                b = τ * Ht[j, j + 1]
                s, l, info = _lhl_shift_decide(rj, b, info, j)
                swap[j] = s
                Gt[j, j] = ifelse(s, b, rj)
                Gt[j, j + 1] = l
                S = Base.setindex(S, s, m + 1)
                L = Base.setindex(L, l, m + 1)
            end
        end
        _lhl_shift_pass!(S, L, Ht, Gt, r, n, τ, k, k + R + 1)
    end
    return info
end

function _lhl_shift_fused!(::Val{R}, ws::LHLWorkspace{T}, σ::T, τ::T) where {R, T}
    Ht = ws.Ht
    Gt = ws.Gt
    swap = ws.swap
    r = ws.resid
    n = ws.n
    @inbounds begin
        @simd for j in 1:n
            r[j] = τ * Ht[j, 1]
        end
        r[1] += σ
        info = 0
        k = 1
        while k + R - 1 <= n - 1
            info = _lhl_shift_block!(Val(R), Ht, Gt, swap, r, n, σ, τ, k, info)
            k += R
        end
        while k <= n - 1
            info = _lhl_shift_block!(Val(1), Ht, Gt, swap, r, n, σ, τ, k, info)
            k += 1
        end
        Gt[n, n] = r[n]
        swap[n] = false
        iszero(Gt[n, n]) && info == 0 && (info = n)
        rd = ws.rdiag
        @simd for j in 1:n
            rd[j] = inv(Gt[j, j])
        end
    end
    return info
end

function _hessenberg_solve!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    Gt = ws.Gt
    swap = ws.swap
    rd = ws.rdiag
    n = ws.n
    # The interchange is a select, not a branch: the pivot pattern is data and mispredicts.
    @inbounds for k in 1:(n - 1)
        s = swap[k]
        a = x[k]
        b = x[k + 1]
        xk = ifelse(s, b, a)
        x[k] = xk
        x[k + 1] = ifelse(s, a, b) - Gt[k, k + 1] * xk
    end
    # Back substitution four rows at a time.  The dot products of block j-4:j-7 over
    # x[j+1:n] do not depend on x[j-3:j], so they are issued right after the 4×4 triangle of
    # block j: the vector loop overlaps the serial chain instead of waiting for it.
    j = n
    s1 = zero(T)
    s2 = zero(T)
    s3 = zero(T)
    s4 = zero(T)
    @inbounds while j - 3 >= 1
        xj = (x[j] - s1) * rd[j]
        x[j] = xj
        s2 += Gt[j, j - 1] * xj
        s3 += Gt[j, j - 2] * xj
        s4 += Gt[j, j - 3] * xj
        xj1 = (x[j - 1] - s2) * rd[j - 1]
        x[j - 1] = xj1
        s3 += Gt[j - 1, j - 2] * xj1
        s4 += Gt[j - 1, j - 3] * xj1
        xj2 = (x[j - 2] - s3) * rd[j - 2]
        x[j - 2] = xj2
        s4 += Gt[j - 2, j - 3] * xj2
        xj3 = (x[j - 3] - s4) * rd[j - 3]
        x[j - 3] = xj3
        jn = j - 4
        if jn - 3 >= 1
            t1 = zero(T)
            t2 = zero(T)
            t3 = zero(T)
            t4 = zero(T)
            @simd for i in (j + 1):n
                xi = x[i]
                t1 += Gt[i, jn] * xi
                t2 += Gt[i, jn - 1] * xi
                t3 += Gt[i, jn - 2] * xi
                t4 += Gt[i, jn - 3] * xi
            end
            s1 = t1 + ((Gt[j, jn] * xj + Gt[j - 1, jn] * xj1) + (Gt[j - 2, jn] * xj2 + Gt[j - 3, jn] * xj3))
            s2 = t2 + ((Gt[j, jn - 1] * xj + Gt[j - 1, jn - 1] * xj1) + (Gt[j - 2, jn - 1] * xj2 + Gt[j - 3, jn - 1] * xj3))
            s3 = t3 + ((Gt[j, jn - 2] * xj + Gt[j - 1, jn - 2] * xj1) + (Gt[j - 2, jn - 2] * xj2 + Gt[j - 3, jn - 2] * xj3))
            s4 = t4 + ((Gt[j, jn - 3] * xj + Gt[j - 1, jn - 3] * xj1) + (Gt[j - 2, jn - 3] * xj2 + Gt[j - 3, jn - 3] * xj3))
        end
        j = jn
    end
    @inbounds while j >= 1
        s = zero(T)
        @simd for i in (j + 1):n
            s += Gt[i, j] * x[i]
        end
        x[j] = (x[j] - s) * rd[j]
        j -= 1
    end
    return x
end

"""
    lhl_ldiv!(x, ws)

`x ← W⁻¹x` for the `W` currently loaded by [`lhl_shift!`](@ref): `Z⁻¹`, Hessenberg solve,
`Z`.  `3n²/2` multiply–adds.
"""
function lhl_ldiv!(x::AbstractVector, ws::LHLWorkspace)
    applyZinv!(x, ws)
    _hessenberg_solve!(x, ws)
    applyZ!(x, ws)
    return x
end

"""
    lhl_refine!(x, A, b, ws, steps)

Apply `steps` rounds of fixed-precision iterative refinement to a solve of `A x = b`, where
`ws` holds the factorization of `A`.  `Z` is not orthogonal, so the raw solve's backward
error carries a factor `κ(Z)`; refinement buys it back for the price of a matvec and a
second `O(n²)` solve (Skeel 1980).  One step is enough to match LU's backward error even on
matrices where `κ(Z)` reaches `10¹⁰`.
"""
function lhl_refine!(x::AbstractVector, A, b::AbstractVector, ws::LHLWorkspace, steps::Int)
    steps <= 0 && return x
    r = ws.resid
    for _ in 1:steps
        mul!(r, A, x)
        r .= b .- r
        lhl_ldiv!(r, ws)
        x .+= r
    end
    return x
end


"""
    lhl(J; balance = true) -> LHLWorkspace
    lhl!(ws, J; balance = true) -> LHLWorkspace

Reduce `J` to upper Hessenberg form by Gaussian similarity with partial pivoting.  `J` is
not modified.  `lhl!` reuses an existing workspace, resizing it if needed.

Follow with [`lhl_shift!`](@ref) to load a shift and [`lhl_ldiv!`](@ref) to solve.
"""
function lhl(J::AbstractMatrix; balance::Bool = true)
    ws = LHLWorkspace{eltype(J)}(checksquare(J))
    lhl_reduce!(ws, J, balance)
    return ws
end

function lhl!(ws::LHLWorkspace, J::AbstractMatrix; balance::Bool = true)
    lhl_reduce!(ws, J, balance)
    return ws
end

end # module
