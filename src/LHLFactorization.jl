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
multipliers repacked for the solves (see `_lhl_lpack!`: groups of four columns interleaved
in `W`-row tiles, zero padded, so that a rank-4 sweep reads one contiguous stream with no
remainder loop).  `perm`/`iperm` are the pivot permutation `P` and its inverse, `iscale`
the reciprocals of the balancing `scale` (powers of two, so exact), and `xbuf` a padded
copy of the right-hand side the solves work in.  `Ht` and `Gt` hold
`H` and the LU of `I - γH` **transposed**: the Hessenberg elimination sweeps rows, and in
a column-major array the transposed layout turns every inner loop of the per-γ work
(fused rebuild and elimination, back substitution) into a contiguous one; `Gt` has `W`
extra zero rows below row `n` so the back substitution's dot products run to a full
vector.  `rdiag` holds the reciprocals of the pivots of that LU.  `resid` doubles
as scratch for the reduction and for `lhl_shift!`; `Ht` and `Gt` are scratch during the
reduction, which fills `Ht` last.  `factors` is an `n×n` view into `fstore`, whose leading
dimension is padded so that no small multiple of the column stride falls within a vector of
a multiple of 4 KiB (the reduction's row-block sweep would otherwise stall on loads that
alias its own stores a few columns back).

The solves write `xbuf` (and `lhl_refine!` `resid`), so one workspace must not serve
concurrent solves; give each thread its own.
"""
mutable struct LHLWorkspace{T, Tr}
    fstore::Matrix{T}
    factors::SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}
    Lp::Vector{T}
    ipiv::Vector{Int}
    perm::Vector{Int}
    iperm::Vector{Int}
    scale::Vector{Tr}
    iscale::Vector{Tr}
    Ht::Matrix{T}
    Gt::Matrix{T}
    rdiag::Vector{T}
    swap::Vector{Bool}
    resid::Vector{T}
    xbuf::Vector{T}
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
    W = _lhl_tilew(T)
    F = Matrix{T}(undef, _lhl_ld(n, T), n)
    return LHLWorkspace{T, Tr}(
        F, view(F, 1:n, 1:n), Vector{T}(undef, _lhl_lpack_len(n, W)),
        Vector{Int}(undef, max(n - 2, 0)), collect(1:n), collect(1:n), ones(Tr, n), ones(Tr, n),
        Matrix{T}(undef, n, n), zeros(T, n + W, n), Vector{T}(undef, n),
        Vector{Bool}(undef, n), Vector{T}(undef, n), zeros(T, n + W), zero(T), zero(T), false, n, 0
    )
end

# Rows per tile of the packed multipliers and of the buffer padding: one vector register
# for the explicit kernels, a fixed 8 for the generic sweeps.
_lhl_tilew(::Type{T}) where {T} = T <: Union{Float32, Float64} ? _LHL_VEC_BYTES ÷ sizeof(T) : 8

# Layout of `Lp` (multipliers of steps 1:n-2, rows k+2:n of step k).  Steps come in groups of
# four, k = 4g-3, g = 1:(n-2)÷4: eight head slots holding the 3×3 triangle rows k+2:k+4 in
# the order l[k+2,k]; l[k+3,k], l[k+3,k+1]; l[k+4,k], l[k+4,k+1], l[k+4,k+2]; then the body,
# rows k+5:n of the four columns zero padded to mp = cld(n-k-4, W)·W rows.  For the
# explicit kernels below 2 MiB (`_lhl_tiled`) the body is tiled — W rows of column 0, W
# rows of column 1, ... — so a rank-4 sweep reads one stream; otherwise it is four plain
# columns, which the hardware prefetcher streams from L3/DRAM faster than one stream four
# times as fast, and which the generic sweeps vectorize as long loops.  The remaining
# steps 4G+1:n-2 follow one at a time as plain zero-padded columns.
const _LHL_HEAD = 8
_lhl_tiled(n::Int, ::Type{T}) where {T} =
    T <: Union{Float32, Float64} && n * n * sizeof(T) <= 4 * 2^20
_lhl_group_size(n::Int, k::Int, W::Int) = _LHL_HEAD + 4 * cld(n - k - 4, W) * W
_lhl_single_size(n::Int, k::Int, W::Int) = cld(n - k - 1, W) * W
function _lhl_lpack_len(n::Int, W::Int)
    len = 0
    G = max(n - 2, 0) >> 2
    for g in 1:G
        len += _lhl_group_size(n, 4g - 3, W)
    end
    for k in (4G + 1):(n - 2)
        len += _lhl_single_size(n, k, W)
    end
    return len
end

# Leading dimension of `fstore`: columns 32-byte aligned, and no multiple m ≤ 8 of the
# column stride within 128 bytes of a multiple of 4 KiB.  Below n = 64 the aliasing costs
# nothing measurable.
function _lhl_ld(n::Int, ::Type{T}) where {T}
    n <= 64 && return n
    sz = isbitstype(T) ? sizeof(T) : sizeof(Ptr{Cvoid})
    W = max(32 ÷ sz, 1)
    ld = W * cld(n, W) - W
    ok = false
    while !ok
        ld += W
        ok = true
        for m in 1:8
            r = (m * ld * sz) % 4096
            ok &= min(r, 4096 - r) >= 128
        end
    end
    return ld
end

function _lhl_resize!(ws::LHLWorkspace{T}, n::Int) where {T}
    n == ws.n && size(ws.factors, 1) == n && return ws
    W = _lhl_tilew(T)
    ws.fstore = Matrix{T}(undef, _lhl_ld(n, T), n)
    ws.factors = view(ws.fstore, 1:n, 1:n)
    ws.Ht = Matrix{T}(undef, n, n)
    ws.Gt = zeros(T, n + W, n)
    resize!(ws.Lp, _lhl_lpack_len(n, W))
    resize!(ws.ipiv, max(n - 2, 0))
    resize!(ws.perm, n)
    resize!(ws.iperm, n)
    resize!(ws.scale, n)
    resize!(ws.iscale, n)
    resize!(ws.rdiag, n)
    resize!(ws.swap, n)
    resize!(ws.resid, n)
    resize!(ws.xbuf, n + W)
    ws.n = n
    ws.reduced = false
    return ws
end

# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

# Parlett–Reinsch scaling: equalize each row/column norm pair by a power of two, which is
# exact in binary floating point and so costs no accuracy.
function _lhl_balance!(A::AbstractMatrix{T}, d::AbstractVector, isc::AbstractVector) where {T}
    n = size(A, 2)
    Tr = real(T)
    fill!(d, one(eltype(d)))
    fill!(isc, one(eltype(isc)))
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
                isc[i] /= f
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
# The reduction.  The kernels reduce the leading n×n block of `A`, n = size(A, 2): `A` is
# `fstore`, whose leading dimension may exceed n.
# ---------------------------------------------------------------------------

"""
    lhl_reduce!(ws, J, balance) -> ws

Reduce `J` to upper Hessenberg form by Gaussian similarity transformations with partial
pivoting, into `ws`.  `J` is not modified.
"""
function lhl_reduce!(ws::LHLWorkspace{T}, J::AbstractMatrix, balance::Bool) where {T}
    n = LinearAlgebra.checksquare(J)
    _lhl_resize!(ws, n)
    A = ws.fstore
    if size(A, 1) == n
        copyto!(A, J)
    else
        @inbounds for j in 1:n
            @simd for i in 1:n
                A[i, j] = J[i, j]
            end
        end
    end
    if balance
        _lhl_balance!(A, ws.scale, ws.iscale)
    else
        fill!(ws.scale, one(eltype(ws.scale)))
        fill!(ws.iscale, one(eltype(ws.iscale)))
    end
    if n >= _lhl_block_min(T)
        _lhl_reduce_blocked!(A, ws.ipiv, ws.Ht, ws.resid, ws.Gt, _lhl_panel_width(n))
        # Gt was scratch; the solves need its pad rows zero again.
        Gt = ws.Gt
        @inbounds for j in 1:n, i in (n + 1):size(Gt, 1)
            Gt[i, j] = zero(T)
        end
    else
        _lhl_reduce_unblocked!(A, ws.ipiv)
    end
    Ht = ws.Ht
    @inbounds for j in 1:n, i in 1:min(j + 1, n)
        Ht[j, i] = A[i, j]
    end
    _lhl_lpack!(ws.Lp, A, n, Val(_lhl_tilew(T)))
    perm = ws.perm
    iperm = ws.iperm
    @inbounds for i in 1:n
        perm[i] = i
    end
    @inbounds for k in 1:(n - 2)
        p = ws.ipiv[k]
        perm[k + 1], perm[p] = perm[p], perm[k + 1]
    end
    @inbounds for i in 1:n
        iperm[perm[i]] = i
    end
    ws.reduced = true
    return ws
end

function _lhl_lpack!(Lp::Vector{T}, A::AbstractMatrix{T}, n::Int, ::Val{W}) where {T, W}
    G = max(n - 2, 0) >> 2
    tiled = _lhl_tiled(n, T)
    o = 0
    @inbounds for g in 1:G
        k = 4g - 3
        Lp[o + 1] = A[k + 2, k]
        Lp[o + 2] = A[k + 3, k]
        Lp[o + 3] = A[k + 3, k + 1]
        Lp[o + 4] = A[k + 4, k]
        Lp[o + 5] = A[k + 4, k + 1]
        Lp[o + 6] = A[k + 4, k + 2]
        o += _LHL_HEAD
        m = n - k - 4
        mp = cld(m, W) * W
        if tiled
            # tile t of column c starts at o + t*4W + c*W
            for c in 0:3
                ob = o + c * W
                i = 0
                while i + W <= m
                    for r in 1:W
                        Lp[ob + r] = A[k + 4 + i + r, k + c]
                    end
                    ob += 4W
                    i += W
                end
                if i < m
                    for r in 1:W
                        Lp[ob + r] = i + r <= m ? A[k + 4 + i + r, k + c] : zero(T)
                    end
                end
            end
        else
            for c in 0:3
                ob = o + c * mp
                @simd for i in 1:m
                    Lp[ob + i] = A[k + 4 + i, k + c]
                end
                for i in (m + 1):mp
                    Lp[ob + i] = zero(T)
                end
            end
        end
        o += 4mp
    end
    @inbounds for k in (4G + 1):(n - 2)
        m = n - k - 1
        @simd for i in 1:m
            Lp[o + i] = A[k + 1 + i, k]
        end
        for i in (m + 1):(cld(m, W) * W)
            Lp[o + i] = zero(T)
        end
        o += cld(m, W) * W
    end
    return Lp
end

function _lhl_reduce_unblocked!(A::AbstractMatrix{T}, ipiv) where {T}
    n = size(A, 2)
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
function _lhl_trailing_update!(A::StridedMatrix{T}, k::Int, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    (n < max(W, 8) || stride(A, 1) != 1) &&
        return invoke(_lhl_trailing_update!, Tuple{AbstractMatrix, Int, Int}, A, k, n)
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
# per-step panel work grows with it.  With the row-block trailing update on the padded
# `fstore` the unblocked reduction stays ahead up to n ≈ 500 (Float64) / 1000 (Float32);
# with the generic sweeps and GEMM (ComplexF64) the two paths trade places between n ≈ 700
# and 1300 depending on the Julia version.
_lhl_block_min(::Type{Float64}) = 500
_lhl_block_min(::Type{Float32}) = 1024
_lhl_block_min(::Type{T}) where {T} = 768
_lhl_panel_width(n::Int) = 16

function _lhl_reduce_blocked!(
        A::AbstractMatrix{T}, ipiv, B0::AbstractMatrix{T},
        w::AbstractVector{T}, pack::AbstractArray{T}, nb::Int
    ) where {T}
    n = size(A, 2)
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
        _lhl_top_gemm!(A, k0, kb, nb, pack)
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

# The deferred right updates of rows 1:k0 of the panel columns:
# A[1:k0, k0+1:kb+1] += A[1:k0, kb+2:n] * A[kb+2:n, k0:kb], one nb-wide K chunk at a time.
function _lhl_top_gemm!(A::AbstractMatrix{T}, k0::Int, kb::Int, nb::Int, pack) where {T}
    n = size(A, 2)
    for kk in (kb + 2):nb:n
        ke = min(kk + nb - 1, n)
        _lhl_gemm!(A, 1, k0, kk, ke, kk, k0, k0 + 1, kb + 1, one(T), pack)
    end
    return nothing
end

# Same in K chunks of 64: with only nb output columns, packing the k0×K left operand would
# cost as much as the product, so the tile reads it in place in row blocks small enough
# that a block's chunk stays in L1/L2.
function _lhl_top_gemm!(
        A::StridedMatrix{T}, k0::Int, kb::Int, nb::Int, pack::Array{T}
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 2)
    (kb + 2 > n || k0 < 1) && return nothing
    if stride(A, 1) != 1
        return invoke(_lhl_top_gemm!, Tuple{AbstractMatrix{T}, Int, Int, Int, Any}, A, k0, kb, nb, pack)
    end
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    mr = 3W
    rowblock = 4mr
    ld = stride(A, 2)
    sz = sizeof(T)
    GC.@preserve A begin
        pA = pointer(A)
        j = kb + 2
        while j <= n
            K = min(64, n - j + 1)
            pP = pA + (j - 1) * ld * sz
            pB = pA + (j - 1 - ld) * sz
            ib = 1
            while ib <= k0
                ie = min(ib + rowblock - 1, k0)
                rfull = ib + ((ie - ib + 1) ÷ mr) * mr - 1
                ct = _lhl_micro_tile!(V, pA, ld, pP, ld, pB, ld, K, 1, ib, rfull, k0 + 1, kb + 1)
                _lhl_micro_edge!(V, pA, ld, pP, ld, pB, ld, K, 1, rfull + 1, ie, k0 + 1, kb + 1)
                _lhl_micro_edge!(V, pA, ld, pP, ld, pB, ld, K, 1, ib, rfull, ct, kb + 1)
                ib = ie + 1
            end
            j += K
        end
    end
    return nothing
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
    rowblock = 384
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
    n = size(A, 2)
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
    n = size(A, 2)
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

`x ← Z x`.  `n²/2` multiply–adds.  Uses `ws.xbuf` as scratch, so not thread-safe on a
shared workspace.
"""
function applyZ!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    if eltype(x) === T
        y = ws.xbuf
        @inbounds for i in 1:n
            y[i] = x[i]
        end
        _lhl_zero_pad!(y, n)
        _lhl_zsweep_buf!(y, ws.Lp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            x[i] = y[ip[i]] * d[i]
        end
    else
        _lhl_zsweep!(x, ws.Lp, n)
        @inbounds for k in (n - 2):-1:1
            p = ws.ipiv[k]
            p != k + 1 && ((x[k + 1], x[p]) = (x[p], x[k + 1]))
        end
        d = ws.scale
        @inbounds @simd for i in 1:n
            x[i] *= d[i]
        end
    end
    return x
end

"""
    applyZinv!(x, ws)

`x ← Z⁻¹x`.  `n²/2` multiply–adds.  Uses `ws.xbuf` as scratch, so not thread-safe on a
shared workspace.
"""
function applyZinv!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    if eltype(x) === T
        y = ws.xbuf
        _lhl_gather!(y, x, ws)
        _lhl_zinvsweep_buf!(y, ws.Lp, n)
        @inbounds for i in 1:n
            x[i] = y[i]
        end
    else
        d = ws.scale
        @inbounds @simd for i in 1:n
            x[i] /= d[i]
        end
        @inbounds for k in 1:(n - 2)
            p = ws.ipiv[k]
            p != k + 1 && ((x[k + 1], x[p]) = (x[p], x[k + 1]))
        end
        _lhl_zinvsweep!(x, ws.Lp, n)
    end
    return x
end

# y ← P⁻¹D⁻¹x into the padded buffer (the pad stays zero through the sweeps: the padded
# multipliers are zero).
function _lhl_gather!(y::Vector, x::AbstractVector, ws::LHLWorkspace)
    n = ws.n
    perm = ws.perm
    isc = ws.iscale
    @inbounds for i in 1:n
        p = perm[i]
        y[i] = x[p] * isc[p]
    end
    _lhl_zero_pad!(y, n)
    return y
end
function _lhl_zero_pad!(y::Vector{T}, n::Int) where {T}
    @inbounds for i in (n + 1):length(y)
        y[i] = zero(T)
    end
    return y
end

# In-place sweeps for any vector type on the packed `Lp` (see `_lhl_lpack!`).  x ← L⁻¹x: the
# head of a group of four steps is a serial 3-step recurrence, then rows k+5:n take all
# four columns at once.
function _lhl_zinvsweep!(x::AbstractVector, Lp::AbstractVector{T}, n::Int) where {T}
    W = _lhl_tilew(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    o = 0
    @inbounds for g in 1:G
        k = 4g - 3
        x1 = x[k + 1]
        x2 = x[k + 2] - Lp[o + 1] * x1
        x[k + 2] = x2
        x3 = x[k + 3] - (Lp[o + 2] * x1 + Lp[o + 3] * x2)
        x[k + 3] = x3
        x4 = x[k + 4] - ((Lp[o + 4] * x1 + Lp[o + 5] * x2) + Lp[o + 6] * x3)
        x[k + 4] = x4
        o += _LHL_HEAD
        m = n - k - 4
        mp = cld(m, W) * W
        if tiled
            oc = o
            i = k + 5
            while i <= n
                rows = min(W, n - i + 1)
                @simd for r in 1:rows
                    x[i + r - 1] -= (Lp[oc + r] * x1 + Lp[oc + W + r] * x2) + (Lp[oc + 2W + r] * x3 + Lp[oc + 3W + r] * x4)
                end
                oc += 4W
                i += W
            end
        else
            @simd for i in 1:m
                x[k + 4 + i] -= (Lp[o + i] * x1 + Lp[o + mp + i] * x2) + (Lp[o + 2mp + i] * x3 + Lp[o + 3mp + i] * x4)
            end
        end
        o += 4mp
    end
    @inbounds for k in (4G + 1):(n - 2)
        xk = x[k + 1]
        m = n - k - 1
        @simd for i in 1:m
            x[k + 1 + i] -= Lp[o + i] * xk
        end
        o += cld(m, W) * W
    end
    return x
end

# x ← L x: steps in reverse order; a group reads its four scalars first, since step k+c
# only touches rows k+c+2:n.
function _lhl_zsweep!(x::AbstractVector, Lp::AbstractVector{T}, n::Int) where {T}
    W = _lhl_tilew(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    o = length(Lp)
    @inbounds for k in (n - 2):-1:(4G + 1)
        xk = x[k + 1]
        m = n - k - 1
        o -= cld(m, W) * W
        @simd for i in 1:m
            x[k + 1 + i] += Lp[o + i] * xk
        end
    end
    @inbounds for g in G:-1:1
        k = 4g - 3
        o -= _lhl_group_size(n, k, W)
        x1 = x[k + 1]
        x2 = x[k + 2]
        x3 = x[k + 3]
        x4 = x[k + 4]
        x[k + 2] = x2 + Lp[o + 1] * x1
        x[k + 3] = x3 + (Lp[o + 2] * x1 + Lp[o + 3] * x2)
        x[k + 4] = x4 + ((Lp[o + 4] * x1 + Lp[o + 5] * x2) + Lp[o + 6] * x3)
        ob = o + _LHL_HEAD
        m = n - k - 4
        mp = cld(m, W) * W
        if tiled
            oc = ob
            i = k + 5
            while i <= n
                rows = min(W, n - i + 1)
                @simd for r in 1:rows
                    x[i + r - 1] += (Lp[oc + r] * x1 + Lp[oc + W + r] * x2) + (Lp[oc + 2W + r] * x3 + Lp[oc + 3W + r] * x4)
                end
                oc += 4W
                i += W
            end
        else
            @simd for i in 1:m
                x[k + 4 + i] += (Lp[ob + i] * x1 + Lp[ob + mp + i] * x2) + (Lp[ob + 2mp + i] * x3 + Lp[ob + 3mp + i] * x4)
            end
        end
    end
    return x
end

# The same sweeps on the padded buffer `ws.xbuf` (length ≥ n + W, zero past n): every tile
# is a full vector, so there is no remainder loop and no branch on the row count.
_lhl_zinvsweep_buf!(y::AbstractVector, Lp::AbstractVector, n::Int) = _lhl_zinvsweep!(y, Lp, n)
_lhl_zsweep_buf!(y::AbstractVector, Lp::AbstractVector, n::Int) = _lhl_zsweep!(y, Lp, n)

function _lhl_zinvsweep_buf!(y::Vector{T}, Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    GC.@preserve y Lp begin
        py = pointer(y)
        h = pointer(Lp)
        @inbounds for g in 1:G
            k = 4g - 3
            x1 = y[k + 1]
            x2 = muladd(-unsafe_load(h), x1, y[k + 2])
            y[k + 2] = x2
            x3 = muladd(-unsafe_load(h, 3), x2, muladd(-unsafe_load(h, 2), x1, y[k + 3]))
            y[k + 3] = x3
            x4 = muladd(-unsafe_load(h, 6), x3, muladd(-unsafe_load(h, 5), x2, muladd(-unsafe_load(h, 4), x1, y[k + 4])))
            y[k + 4] = x4
            b1 = _lhl_bcast(V, -x1)
            b2 = _lhl_bcast(V, -x2)
            b3 = _lhl_bcast(V, -x3)
            b4 = _lhl_bcast(V, -x4)
            p = h + _LHL_HEAD * sz
            q = py + (k + 4) * sz
            mpb = cld(n - k - 4, W) * W * sz
            h = p + 4mpb
            # byte distance between the four columns' rows, and the advance per vector
            cs = tiled ? W * sz : mpb
            pa = tiled ? 4W * sz : W * sz
            pend = tiled ? h : p + mpb
            while p < pend
                v = _lhl_vload(V, q)
                v = _lhl_fma(_lhl_vload(V, p), b1, v)
                v = _lhl_fma(_lhl_vload(V, p + cs), b2, v)
                v = _lhl_fma(_lhl_vload(V, p + 2cs), b3, v)
                v = _lhl_fma(_lhl_vload(V, p + 3cs), b4, v)
                _lhl_vstore!(q, v)
                p += pa
                q += W * sz
            end
        end
        @inbounds for k in (4G + 1):(n - 2)
            b = _lhl_bcast(V, -y[k + 1])
            q = py + (k + 1) * sz
            pend = h + cld(n - k - 1, W) * W * sz
            while h < pend
                _lhl_vstore!(q, _lhl_fma(_lhl_vload(V, h), b, _lhl_vload(V, q)))
                h += W * sz
                q += W * sz
            end
        end
    end
    return y
end

function _lhl_zsweep_buf!(y::Vector{T}, Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    GC.@preserve y Lp begin
        py = pointer(y)
        h = pointer(Lp) + length(Lp) * sz
        @inbounds for k in (n - 2):-1:(4G + 1)
            b = _lhl_bcast(V, y[k + 1])
            m = cld(n - k - 1, W) * W
            h -= m * sz
            p = h
            q = py + (k + 1) * sz
            pend = p + m * sz
            while p < pend
                _lhl_vstore!(q, _lhl_fma(_lhl_vload(V, p), b, _lhl_vload(V, q)))
                p += W * sz
                q += W * sz
            end
        end
        @inbounds for g in G:-1:1
            k = 4g - 3
            h -= _lhl_group_size(n, k, W) * sz
            x1 = y[k + 1]
            x2 = y[k + 2]
            x3 = y[k + 3]
            x4 = y[k + 4]
            y[k + 2] = muladd(unsafe_load(h), x1, x2)
            y[k + 3] = muladd(unsafe_load(h, 3), x2, muladd(unsafe_load(h, 2), x1, x3))
            y[k + 4] = muladd(unsafe_load(h, 6), x3, muladd(unsafe_load(h, 5), x2, muladd(unsafe_load(h, 4), x1, x4)))
            b1 = _lhl_bcast(V, x1)
            b2 = _lhl_bcast(V, x2)
            b3 = _lhl_bcast(V, x3)
            b4 = _lhl_bcast(V, x4)
            p = h + _LHL_HEAD * sz
            q = py + (k + 4) * sz
            mpb = cld(n - k - 4, W) * W * sz
            cs = tiled ? W * sz : mpb
            pa = tiled ? 4W * sz : W * sz
            pend = tiled ? p + 4mpb : p + mpb
            while p < pend
                v = _lhl_vload(V, q)
                v = _lhl_fma(_lhl_vload(V, p), b1, v)
                v = _lhl_fma(_lhl_vload(V, p + cs), b2, v)
                v = _lhl_fma(_lhl_vload(V, p + 2cs), b3, v)
                v = _lhl_fma(_lhl_vload(V, p + 3cs), b4, v)
                _lhl_vstore!(q, v)
                p += pa
                q += W * sz
            end
        end
    end
    return y
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
        # Four rows per pass; `Gt`'s padded leading dimension keeps its columns out of the
        # L1 set the four `Ht` columns share when the stride is a multiple of 4 KiB.
        info = _lhl_shift_fused!(Val(4), ws, σ, τ)
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
    rd = ws.rdiag
    n = ws.n
    _lhl_hess_forward!(x, Gt, ws.swap, n)
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

# The interchange is a select, not a branch: the pivot pattern is data and mispredicts.
function _lhl_hess_forward!(x::AbstractVector, Gt::AbstractMatrix, swap::Vector{Bool}, n::Int)
    @inbounds for k in 1:(n - 1)
        s = swap[k]
        a = x[k]
        b = x[k + 1]
        xk = ifelse(s, b, a)
        x[k] = xk
        x[k + 1] = ifelse(s, a, b) - Gt[k, k + 1] * xk
    end
    return x
end

# On the padded buffer: same pipelining, the four dot products as vector accumulators
# running to a full vector past n (Gt's pad rows and y's pad are zero).
_hessenberg_solve_buf!(y::AbstractVector, ws::LHLWorkspace) = _hessenberg_solve!(y, ws)

function _hessenberg_solve_buf!(y::Vector{T}, ws::LHLWorkspace{T}) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    Gt = ws.Gt
    rd = ws.rdiag
    n = ws.n
    ldg = size(Gt, 1)
    _lhl_hess_forward!(y, Gt, ws.swap, n)
    GC.@preserve y Gt begin
        py = pointer(y)
        pG = pointer(Gt)
        j = n
        s1 = zero(T)
        s2 = zero(T)
        s3 = zero(T)
        s4 = zero(T)
        @inbounds while j - 3 >= 1
            xj = (y[j] - s1) * rd[j]
            y[j] = xj
            s2 = muladd(Gt[j, j - 1], xj, s2)
            s3 = muladd(Gt[j, j - 2], xj, s3)
            s4 = muladd(Gt[j, j - 3], xj, s4)
            xj1 = (y[j - 1] - s2) * rd[j - 1]
            y[j - 1] = xj1
            s3 = muladd(Gt[j - 1, j - 2], xj1, s3)
            s4 = muladd(Gt[j - 1, j - 3], xj1, s4)
            xj2 = (y[j - 2] - s3) * rd[j - 2]
            y[j - 2] = xj2
            s4 = muladd(Gt[j - 2, j - 3], xj2, s4)
            xj3 = (y[j - 3] - s4) * rd[j - 3]
            y[j - 3] = xj3
            jn = j - 4
            if jn - 3 >= 1
                c1 = pG + ((jn - 1) * ldg + j) * sz
                c2 = c1 - ldg * sz
                c3 = c2 - ldg * sz
                c4 = c3 - ldg * sz
                q = py + j * sz
                mb = cld(n - j, W) * W * sz
                a1 = _lhl_bcast(V, zero(T))
                a2 = a1
                a3 = a1
                a4 = a1
                o = 0
                while o < mb
                    v = _lhl_vload(V, q + o)
                    a1 = _lhl_fma(_lhl_vload(V, c1 + o), v, a1)
                    a2 = _lhl_fma(_lhl_vload(V, c2 + o), v, a2)
                    a3 = _lhl_fma(_lhl_vload(V, c3 + o), v, a3)
                    a4 = _lhl_fma(_lhl_vload(V, c4 + o), v, a4)
                    o += W * sz
                end
                s1 = _lhl_vsum(a1) + ((Gt[j, jn] * xj + Gt[j - 1, jn] * xj1) + (Gt[j - 2, jn] * xj2 + Gt[j - 3, jn] * xj3))
                s2 = _lhl_vsum(a2) + ((Gt[j, jn - 1] * xj + Gt[j - 1, jn - 1] * xj1) + (Gt[j - 2, jn - 1] * xj2 + Gt[j - 3, jn - 1] * xj3))
                s3 = _lhl_vsum(a3) + ((Gt[j, jn - 2] * xj + Gt[j - 1, jn - 2] * xj1) + (Gt[j - 2, jn - 2] * xj2 + Gt[j - 3, jn - 2] * xj3))
                s4 = _lhl_vsum(a4) + ((Gt[j, jn - 3] * xj + Gt[j - 1, jn - 3] * xj1) + (Gt[j - 2, jn - 3] * xj2 + Gt[j - 3, jn - 3] * xj3))
            end
            j = jn
        end
        @inbounds while j >= 1
            s = zero(T)
            @simd for i in (j + 1):n
                s += Gt[i, j] * y[i]
            end
            y[j] = (y[j] - s) * rd[j]
            j -= 1
        end
    end
    return y
end

@inline _lhl_vsum(v::NTuple{W, VecElement{T}}) where {W, T} = sum(ntuple(w -> v[w].value, Val(W)))

"""
    lhl_ldiv!(x, ws)

`x ← W⁻¹x` for the `W` currently loaded by [`lhl_shift!`](@ref): `Z⁻¹`, Hessenberg solve,
`Z`.  `3n²/2` multiply–adds.  Uses `ws.xbuf` as scratch, so concurrent solves must each
have their own workspace.
"""
function lhl_ldiv!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    if eltype(x) === T
        y = ws.xbuf
        _lhl_gather!(y, x, ws)
        _lhl_zinvsweep_buf!(y, ws.Lp, n)
        _hessenberg_solve_buf!(y, ws)
        _lhl_zsweep_buf!(y, ws.Lp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            x[i] = y[ip[i]] * d[i]
        end
    else
        applyZinv!(x, ws)
        _hessenberg_solve!(x, ws)
        applyZ!(x, ws)
    end
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
