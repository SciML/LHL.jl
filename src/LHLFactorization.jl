"""
    LHLFactorization

The **LHL factorization**: a reduction of a square matrix to upper Hessenberg form by
Gaussian similarity transformations with partial pivoting,

    J = Z H Z⁻¹,   Z = D·P·L

with `L` unit lower triangular (multipliers bounded by 1 in modulus — by `√2` for a
complex `J`, whose pivot magnitudes are `|re| + |im|`, as LAPACK's), `P` a permutation
and `D` a balancing diagonal.  This is Wilkinson's elimination method — EISPACK's `ELMHES` — packaged
for a purpose it is rarely exposed for: **solving a family of shifted systems**

    (σI + τJ) x = b

for many `(σ, τ)` against one reduction.  The shift never reaches `Z`, so

    σI + τJ = Z (σI + τH) Z⁻¹

and `σI + τH` is Hessenberg: a new shift costs `O(n²)`, against `O(n³)` for a fresh LU.
The motivating case is the iteration matrix `W = I - γJ` of a stiff ODE solver, where
adaptive step-size control changes `γ` every step while `J` is held fixed for tens of them.
A real `J` may take complex shifts — resolvents `(sI - J)⁻¹`, the complex stage systems of
Radau methods — while its reduction stays real: `lhl(J; shift = ComplexF64)`, or an
[`LHLShift`](@ref) held next to the workspace.

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

import LinearAlgebra
using LinearAlgebra: checksquare, mul!

export LHLWorkspace, LHLShift, lhl, lhl!, lhl_reduce!, lhl_shift!, lhl_ldiv!, lhl_refine!,
    applyZ!, applyZinv!

# The complex element types with explicit-vector kernels (planar shift state, interleaved
# real-view reduction kernels, planar solve sweeps).
const _LHL_CFloat = Union{ComplexF32, ComplexF64}

"""
    LHLShift{TG}(n)
    LHLShift{TG}(ws::LHLWorkspace)

The shift-dependent half of an LHL solve: the LU of one `σI + τH` and the buffers its
solves need, with element type `TG`.  Every [`LHLWorkspace`](@ref) owns one (`ws.shift`),
which the two-argument `lhl_shift!(ws, σ, τ)` / `lhl_ldiv!(x, ws)` use; build more to hold
several shifts against one reduction at once — a real one and a complex one on the same
real `J`, as a Radau-type implicit Runge–Kutta step needs — and pass them explicitly:
`lhl_shift!(sh, ws, σ, τ)`, `lhl_ldiv!(x, sh, ws)`.  `TG` is the reduction's element type
`T` or `Complex{T}`.

`Gt` holds the LU of `σI + τH` **transposed**: the Hessenberg elimination sweeps rows, and
in a column-major array the transposed layout turns every inner loop of the per-shift work
(fused rebuild and elimination, back substitution) into a contiguous one; it has extra zero
rows below row `n` so the back substitution's dot products run to a full vector.  For a
complex `TG` the storage is real and **planar**: each column of `Gt` holds the real parts in
rows `1:n` and the imaginary parts in rows `po+1:po+n` (`po` the plane offset), and `work`
and `xbuf` are laid out the same way, so that every inner loop is a real one — the Z sweeps
run the real multipliers over each plane, and the complex products of the elimination and
the back substitution are four real multiply–adds with no lane shuffling.  `rdiag` holds
the reciprocals of the pivots, `swap` the interchanges, `info` the index of the first zero
pivot (0 if none), `σ`/`τ` the shift loaded.  `work` is the pending row of `lhl_shift!`,
`resid` the residual of `lhl_refine!`, `xbuf` the padded copy of the right-hand side the
solves work in — so one `LHLShift` must not serve concurrent solves.
"""
mutable struct LHLShift{TG, Tr}
    Gt::Matrix{Tr}
    rdiag::Vector{TG}
    swap::Vector{Bool}
    work::Vector{Tr}
    resid::Vector{TG}
    xbuf::Vector{Tr}
    σ::TG
    τ::TG
    n::Int
    po::Int
    info::Int
end

# Rows per plane: `n` plus the vector pad; for two planes, kept a safe distance from a
# multiple of 4 KiB so the real and imaginary planes of one column do not alias in L1.
_lhl_planes(::Type{TG}) where {TG} = TG <: Complex ? 2 : 1
function _lhl_planeoff(n::Int, ::Type{TG}) where {TG}
    Tr = real(TG)
    m = n + _lhl_tilew(Tr)
    return TG <: Complex ? _lhl_ld(m, Tr) : m
end

function LHLShift{TG}(n::Integer) where {TG}
    Tr = real(TG)
    P = _lhl_planes(TG)
    po = _lhl_planeoff(n, TG)
    return LHLShift{TG, Tr}(
        zeros(Tr, P * po, n), Vector{TG}(undef, n), Vector{Bool}(undef, n),
        Vector{Tr}(undef, P * po), Vector{TG}(undef, n), zeros(Tr, P * po),
        zero(TG), zero(TG), n, po, 0
    )
end

"""
    LHLWorkspace{T}(n; shift = T)

Storage for one LHL factorization: the reduction of `J` (element type `T`) and, in
`ws.shift::LHLShift{shift}`, the LU of the current shifted Hessenberg.  Build one with
[`lhl`](@ref) or [`lhl!`](@ref).  `shift = Complex{T}` on a real `T` gives a real reduction
whose shifts and solves are complex — see [`LHLShift`](@ref); `T` and `Complex{real(T)}`
are the two choices.  (`LHLWorkspace{T, Tr, TG}`: `Tr = real(T)`, `TG` the shift type.)

`factors` holds `H` in `triu(factors, -1)` and the step-`k` multipliers in the annihilated
positions `factors[k+2:n, k]`, exactly the way an LU packs its own; `Lp` holds the same
multipliers repacked for the solves (see `_lhl_lpack!`: groups of four columns interleaved
in `W`-row tiles, zero padded, so that a rank-4 sweep reads one contiguous stream with no
remainder loop); for a `ComplexF32`/`ComplexF64` workspace `Lpp` holds them once more as
two real planes, which the solves' planar sweeps read instead.  `perm`/`iperm` are the pivot permutation `P` and its inverse, `iscale`
the reciprocals of the balancing `scale` (powers of two, so exact), and `xbuf` a padded
copy of the vector `applyZ!`/`applyZinv!` work in.  `Ht` holds `H` **transposed** (see
[`LHLShift`](@ref) for why); `Ht`, `work` and `pack` are scratch during the reduction,
which fills `Ht` last.  `factors` is an `n×n` view into `fstore`, whose leading dimension
is padded so that no small multiple of the column stride falls within a vector of a
multiple of 4 KiB (the reduction's row-block sweep would otherwise stall on loads that
alias its own stores a few columns back).

The fields of `ws.shift` (`Gt`, `rdiag`, `swap`, `resid`, `σ`, `τ`, `info`) are also
reachable as properties of `ws`.  The solves write `xbuf` (and `lhl_refine!` `resid`), so
one workspace must not serve concurrent solves; give each thread its own.
"""
mutable struct LHLWorkspace{T, Tr, TG}
    fstore::Matrix{T}
    factors::SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, UnitRange{Int}}, false}
    Lp::Vector{T}
    # For a ComplexF32/ComplexF64 `T`: the same multipliers again as two real planes (re,
    # then im, each in the real type's packed layout), for the planar solve sweeps.
    # Empty otherwise.
    Lpp::Vector{Tr}
    ipiv::Vector{Int}
    perm::Vector{Int}
    iperm::Vector{Int}
    scale::Vector{Tr}
    iscale::Vector{Tr}
    Ht::Matrix{T}
    work::Vector{T}
    pack::Vector{T}
    xbuf::Vector{T}
    shift::LHLShift{TG, Tr}
    # Whether `factors` holds a valid reduction. A consumer tracking *whose* Jacobian it
    # is must do so itself; this only says one was computed.
    reduced::Bool
    n::Int
end

function LHLWorkspace{T}(n::Integer; shift::Type = T) where {T}
    TG = shift
    Tr = real(T)
    (TG === T || TG === Complex{Tr}) || throw(
        ArgumentError("shift type must be $T or $(Complex{Tr}), got $TG")
    )
    W = _lhl_tilew(T)
    F = Matrix{T}(undef, _lhl_ld(n, T), n)
    return LHLWorkspace{T, Tr, TG}(
        F, view(F, 1:n, 1:n), Vector{T}(undef, _lhl_lpack_len(n, W)),
        Vector{Tr}(undef, _lhl_lpp_len(n, T)),
        Vector{Int}(undef, max(n - 2, 0)), collect(1:n), collect(1:n), ones(Tr, n), ones(Tr, n),
        Matrix{T}(undef, n, n), Vector{T}(undef, n), Vector{T}(undef, _lhl_pack_len(n, T)),
        zeros(T, n + W), LHLShift{TG}(n), false, n
    )
end

LHLShift{TG}(ws::LHLWorkspace) where {TG} = LHLShift{TG}(ws.n)

const _LHL_SHIFT_FIELDS = (:Gt, :rdiag, :swap, :resid, :σ, :τ, :info)
@inline function Base.getproperty(ws::LHLWorkspace, s::Symbol)
    s in _LHL_SHIFT_FIELDS && return getfield(getfield(ws, :shift), s)
    return getfield(ws, s)
end
@inline function Base.setproperty!(ws::LHLWorkspace, s::Symbol, v)
    s in _LHL_SHIFT_FIELDS && return setfield!(getfield(ws, :shift), s, v)
    return setfield!(ws, s, v)
end
Base.propertynames(::LHLWorkspace) = (fieldnames(LHLWorkspace)..., _LHL_SHIFT_FIELDS...)

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

_lhl_lpp_len(n::Int, ::Type{T}) where {T} =
    T <: _LHL_CFloat ? 2 * _lhl_lpack_len(n, _lhl_tilew(real(T))) : 0

# Scratch for the blocked reduction's packed GEMM/TRSM operands (`_lhl_gemm_micro!`,
# `_lhl_trsm_block!`) and, on the Float32/Float64 path, the GEMV partials of the column
# groups plus per-chunk and per-group tables (`_lhl_panel_steps!`).
function _lhl_pack_len(n::Int, ::Type{T}) where {T}
    nb = _lhl_panel_width(n)
    len = nb * (n + 4) + nb * nb
    if T <: Union{Float32, Float64}
        P = cld(n, _LHL_GEMV_GROUP)
        len = max(len, P * _lhl_ld(n, T) + 4nb * Threads.nthreads() + 2nb * P)
    elseif T <: _LHL_CFloat
        # room for the expanded GEMM panel (real view) and the panel GEMV's group partials
        len = max(2 * len, cld(n, _LHL_GEMV_GROUP) * (n + 8))
    end
    return len
end

function _lhl_resize!(ws::LHLWorkspace{T}, n::Int) where {T}
    n == ws.n && size(ws.factors, 1) == n && return ws
    W = _lhl_tilew(T)
    ws.fstore = Matrix{T}(undef, _lhl_ld(n, T), n)
    ws.factors = view(ws.fstore, 1:n, 1:n)
    ws.Ht = Matrix{T}(undef, n, n)
    resize!(ws.Lp, _lhl_lpack_len(n, W))
    resize!(ws.Lpp, _lhl_lpp_len(n, T))
    resize!(ws.ipiv, max(n - 2, 0))
    resize!(ws.perm, n)
    resize!(ws.iperm, n)
    resize!(ws.scale, n)
    resize!(ws.iscale, n)
    resize!(ws.work, n)
    resize!(ws.pack, _lhl_pack_len(n, T))
    resize!(ws.xbuf, n + W)
    _lhl_resize!(ws.shift, n)
    ws.n = n
    ws.reduced = false
    return ws
end

function _lhl_resize!(sh::LHLShift{TG, Tr}, n::Int) where {TG, Tr}
    n == sh.n && return sh
    P = _lhl_planes(TG)
    po = _lhl_planeoff(n, TG)
    sh.Gt = zeros(Tr, P * po, n)
    resize!(sh.rdiag, n)
    resize!(sh.swap, n)
    resize!(sh.work, P * po)
    resize!(sh.resid, n)
    resize!(sh.xbuf, P * po)
    sh.n = n
    sh.po = po
    sh.info = 0
    return sh
end

# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

# Parlett–Reinsch scaling: equalize each row/column norm pair by a power of two, which is
# exact in binary floating point and so costs no accuracy.  Complex magnitudes are
# |re| + |im| (`_lhl_pivmag`), as EISPACK's CBAL and LAPACK's CABS1 — no hypot per element.
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
                c += _lhl_pivmag(A[j, i])
                r += _lhl_pivmag(A[i, j])
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
    lhl_reduce!(ws, J, balance, thread = Val(true)) -> ws

Reduce `J` to upper Hessenberg form by Gaussian similarity transformations with partial
pivoting, into `ws`.  `J` is not modified.  `thread` (`Val(true)`/`Val(false)` or a
`Bool`) allows the reduction to run on Polyester threads when Polyester is loaded; see
[`lhl`](@ref).
"""
function lhl_reduce!(ws::LHLWorkspace{T}, J::AbstractMatrix, balance::Bool, thread = Val(true)) where {T}
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
    _lhl_reduce_core!(_LHL_BACKEND[], ws, thread)
    _lhl_lpack!(ws.Lp, A, n, Val(_lhl_tilew(T)))
    T <: _LHL_CFloat && _lhl_lpack2!(ws.Lpp, A, n, Val(_lhl_tilew(real(T))))
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

# The reduction proper and the transpose into `Ht`, on the chunk backend `bk` (`LHLSerial`
# or the extension's; see the threading section).
function _lhl_reduce_core!(bk, ws::LHLWorkspace{T}, thread) where {T}
    n = ws.n
    A = ws.fstore
    nt = _lhl_nthreads(bk, thread, n, T)
    if n >= _lhl_block_min(T)
        _lhl_reduce_blocked!(bk, A, ws.ipiv, ws.Ht, ws.work, ws.pack, _lhl_panel_width(n), nt)
    else
        _lhl_reduce_unblocked!(A, ws.ipiv)
    end
    _lhl_ht_fill!(bk, ws.Ht, A, n, nt)
    return nothing
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

# `_lhl_lpack!` once more for a complex `A`, splitting each multiplier into the re plane
# (`Lpp[1:end÷2]`) and the im plane behind it, both in the real type's packed layout
# (`W = _lhl_tilew(Tr)`, tiling by `_lhl_tiled(n, Tr)`) so the planar sweeps read them
# with the real kernels' geometry.
function _lhl_lpack2!(Lpp::Vector{Tr}, A::AbstractMatrix{T}, n::Int, ::Val{W}) where {Tr, T, W}
    Lh = length(Lpp) >> 1
    G = max(n - 2, 0) >> 2
    tiled = _lhl_tiled(n, Tr)
    o = 0
    @inline st!(o_, i_, v) = @inbounds begin
        Lpp[o_ + i_] = real(v)
        Lpp[Lh + o_ + i_] = imag(v)
        nothing
    end
    @inbounds for g in 1:G
        k = 4g - 3
        st!(o, 1, A[k + 2, k])
        st!(o, 2, A[k + 3, k])
        st!(o, 3, A[k + 3, k + 1])
        st!(o, 4, A[k + 4, k])
        st!(o, 5, A[k + 4, k + 1])
        st!(o, 6, A[k + 4, k + 2])
        o += _LHL_HEAD
        m = n - k - 4
        mp = cld(m, W) * W
        if tiled
            for c in 0:3
                ob = o + c * W
                i = 0
                while i + W <= m
                    for r in 1:W
                        st!(ob, r, A[k + 4 + i + r, k + c])
                    end
                    ob += 4W
                    i += W
                end
                if i < m
                    for r in 1:W
                        st!(ob, r, i + r <= m ? A[k + 4 + i + r, k + c] : zero(T))
                    end
                end
            end
        else
            for c in 0:3
                ob = o + c * mp
                for i in 1:m
                    st!(ob, i, A[k + 4 + i, k + c])
                end
                for i in (m + 1):mp
                    st!(ob, i, zero(T))
                end
            end
        end
        o += 4mp
    end
    @inbounds for k in (4G + 1):(n - 2)
        m = n - k - 1
        for i in 1:m
            st!(o, i, A[k + 1 + i, k])
        end
        for i in (m + 1):(cld(m, W) * W)
            st!(o, i, zero(T))
        end
        o += cld(m, W) * W
    end
    return Lpp
end

function _lhl_reduce_unblocked!(A::AbstractMatrix{T}, ipiv) where {T}
    n = size(A, 2)
    @inbounds for k in 1:(n - 2)
        p = k + 1
        amax = _lhl_pivmag(A[k + 1, k])
        for i in (k + 2):n
            a = _lhl_pivmag(A[i, k])
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

# `NTuple{W, VecElement{T}}` admits the empty tuple, which leaves `T` free (Aqua's
# unbound-type-parameter check); spelling out the first lane pins both parameters.
const _LHLVec{T, W} = Tuple{VecElement{T}, Vararg{VecElement{T}, W}}

@inline _lhl_vneg(a::_LHLVec{T, W}) where {T, W} =
    ntuple(w -> VecElement(-a[w].value), Val(W + 1))
@inline _lhl_vadd(a::_LHLVec{T, W}, b::_LHLVec{T, W}) where {T, W} =
    ntuple(w -> VecElement(a[w].value + b[w].value), Val(W + 1))
@inline _lhl_vselect(m::Tuple{Bool, Vararg{Bool, W}}, a::_LHLVec{T, W}, b::_LHLVec{T, W}) where {T, W} =
    ntuple(w -> VecElement(ifelse(m[w], a[w].value, b[w].value)), Val(W + 1))
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

# ---------------------------------------------------------------------------
# Complex kernels on the interleaved storage.  A complex column is read through a
# real-typed pointer: element i sits at real lanes 2i-1 (re) and 2i (im), so a vector of
# W real lanes holds W÷2 consecutive complex elements.  A product a·b then splits into
# b.re times the plain lanes plus b.im times the swap-negated lanes (-im, re) — and the
# swap-negation only ever happens *outside* the inner loops: on multiplier vectors held
# in registers across a whole row block, and on accumulators after a column sweep, since
# Σⱼ swapneg(aⱼ)·cⱼ = swapneg(Σⱼ aⱼ·cⱼ).  The inner loops are pure loads, FMAs and
# stores, exactly like the real kernels with twice the vectors — measured at parity with
# planar (two real planes) storage, which would have cost `factors` its documented layout.
# ---------------------------------------------------------------------------

@inline _lhl_vswapneg(v::_LHLVec{T, W}) where {T, W} =
    ntuple(w -> VecElement(isodd(w) ? -v[w + 1].value : v[w - 1].value), Val(W + 1))
@inline _lhl_clanemask(::Val{W}, i0::Int, lo::Int) where {W} =
    ntuple(w -> i0 + ((w - 1) >> 1) >= lo, Val(W))

function _lhl_trailing_update!(A::StridedMatrix{T}, k::Int, n::Int) where {T <: _LHL_CFloat}
    Tr = real(T)
    V = _lhl_vectype(Tr)
    W = _LHL_VEC_BYTES ÷ sizeof(Tr)
    Wc = W >> 1
    (n < max(W, 8) || stride(A, 1) != 1) &&
        return invoke(_lhl_trailing_update!, Tuple{AbstractMatrix, Int, Int}, A, k, n)
    GC.@preserve A begin
        pA = Ptr{Tr}(pointer(A))
        lds = stride(A, 2) * sizeof(T)
        i = _lhl_crb_rows!(Val(false), V, pA, lds, 1, k + 1, k, n)
        if i <= k + 1
            i0 = min(i, n - Wc + 1)
            _lhl_crb_masked!(V, pA, lds, i0, i, k, n)
            i = i0 + Wc
        end
        if i <= n
            i = _lhl_crb_rows!(Val(true), V, pA, lds, i, n, k, n)
            i <= n && _lhl_crb_masked!(V, pA, lds, n - Wc + 1, i, k, n)
        end
    end
    return A
end

# Whole blocks of complex rows ia:ib, with (L = true) or without the left update; two
# vectors (W complex rows), then one; returns the first row not covered.
@inline function _lhl_crb_rows!(
        ::Val{L}, ::Type{V}, pA::Ptr{T}, lds::Int, ia::Int, ib::Int, k::Int, n::Int
    ) where {L, W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    vb = W * sz
    Wc = W >> 1
    prk0 = pA + 2k * sz + (k + 1) * lds        # A[k+1, k+2]
    pck0 = pA + (2k + 2) * sz + (k - 1) * lds  # A[k+2, k]
    z = _lhl_bcast(V, zero(T))
    i = ia
    while i + W - 1 <= ib
        prow = pA + 2 * (i - 1) * sz
        pl = prow + (k - 1) * lds
        l1 = _lhl_vload(V, pl); l2 = _lhl_vload(V, pl + vb)
        m1 = _lhl_vswapneg(l1); m2 = _lhl_vswapneg(l2)
        sr1 = sr2 = si1 = si2 = z
        pcol = prow + (k + 1) * lds
        prk = prk0
        pck = pck0
        for _ in (k + 2):n
            bvr = _lhl_bcast(V, unsafe_load(pck))
            bvi = _lhl_bcast(V, unsafe_load(pck, 2))
            a1 = _lhl_vload(V, pcol); a2 = _lhl_vload(V, pcol + vb)
            if L
                bpr = _lhl_bcast(V, -unsafe_load(prk))
                bpi = _lhl_bcast(V, -unsafe_load(prk, 2))
                a1 = _lhl_fma(m1, bpi, _lhl_fma(l1, bpr, a1))
                a2 = _lhl_fma(m2, bpi, _lhl_fma(l2, bpr, a2))
                _lhl_vstore!(pcol, a1); _lhl_vstore!(pcol + vb, a2)
            end
            sr1 = _lhl_fma(a1, bvr, sr1); si1 = _lhl_fma(a1, bvi, si1)
            sr2 = _lhl_fma(a2, bvr, sr2); si2 = _lhl_fma(a2, bvi, si2)
            pcol += lds
            prk += lds
            pck += 2sz
        end
        pc = prow + k * lds
        _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), _lhl_vadd(sr1, _lhl_vswapneg(si1))))
        _lhl_vstore!(pc + vb, _lhl_vadd(_lhl_vload(V, pc + vb), _lhl_vadd(sr2, _lhl_vswapneg(si2))))
        i += W
    end
    while i + Wc - 1 <= ib
        prow = pA + 2 * (i - 1) * sz
        l1 = _lhl_vload(V, prow + (k - 1) * lds)
        m1 = _lhl_vswapneg(l1)
        sr1 = z
        si1 = z
        pcol = prow + (k + 1) * lds
        prk = prk0
        pck = pck0
        for _ in (k + 2):n
            bvr = _lhl_bcast(V, unsafe_load(pck))
            bvi = _lhl_bcast(V, unsafe_load(pck, 2))
            a1 = _lhl_vload(V, pcol)
            if L
                bpr = _lhl_bcast(V, -unsafe_load(prk))
                bpi = _lhl_bcast(V, -unsafe_load(prk, 2))
                a1 = _lhl_fma(m1, bpi, _lhl_fma(l1, bpr, a1))
                _lhl_vstore!(pcol, a1)
            end
            sr1 = _lhl_fma(a1, bvr, sr1)
            si1 = _lhl_fma(a1, bvi, si1)
            pcol += lds
            prk += lds
            pck += 2sz
        end
        pc = prow + k * lds
        _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), _lhl_vadd(sr1, _lhl_vswapneg(si1))))
        i += Wc
    end
    return i
end

# One vector of complex rows i0:i0+W÷2-1: only rows ≥ lo take part, of those only rows
# ≥ k+2 get the left update; excluded lanes are stored back unchanged.
@inline function _lhl_crb_masked!(
        ::Type{V}, pA::Ptr{T}, lds::Int, i0::Int, lo::Int, k::Int, n::Int
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    ma = _lhl_clanemask(Val(W), i0, lo)
    ml = _lhl_clanemask(Val(W), i0, max(lo, k + 2))
    z = _lhl_bcast(V, zero(T))
    prow = pA + 2 * (i0 - 1) * sz
    l1 = _lhl_vselect(ml, _lhl_vload(V, prow + (k - 1) * lds), z)
    m1 = _lhl_vswapneg(l1)
    sr1 = z
    si1 = z
    pcol = prow + (k + 1) * lds
    prk = pA + 2k * sz + (k + 1) * lds
    pck = pA + (2k + 2) * sz + (k - 1) * lds
    for _ in (k + 2):n
        bvr = _lhl_bcast(V, unsafe_load(pck))
        bvi = _lhl_bcast(V, unsafe_load(pck, 2))
        bpr = _lhl_bcast(V, -unsafe_load(prk))
        bpi = _lhl_bcast(V, -unsafe_load(prk, 2))
        a1 = _lhl_vload(V, pcol)
        a1 = _lhl_vselect(ml, _lhl_fma(m1, bpi, _lhl_fma(l1, bpr, a1)), a1)
        _lhl_vstore!(pcol, a1)
        sr1 = _lhl_fma(a1, bvr, sr1)
        si1 = _lhl_fma(a1, bvi, si1)
        pcol += lds
        prk += lds
        pck += 2sz
    end
    pc = prow + k * lds
    _lhl_vstore!(pc, _lhl_vadd(_lhl_vload(V, pc), _lhl_vselect(ma, _lhl_vadd(sr1, _lhl_vswapneg(si1)), z)))
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
# with the explicit complex kernels the measured crossovers sit at n ≈ 500 (ComplexF64)
# and 1024–1400 (ComplexF32), matching the real types of the same element size.
_lhl_block_min(::Type{Float64}) = 500
_lhl_block_min(::Type{Float32}) = 1024
_lhl_block_min(::Type{ComplexF64}) = 512
_lhl_block_min(::Type{ComplexF32}) = 1024
_lhl_block_min(::Type{T}) where {T} = 768
_lhl_panel_width(n::Int) = 16

# ---------------------------------------------------------------------------
# Threading.  `thread` is `Val(true)`/`Val(false)` or a `Bool`.  Only the blocked
# Float32/Float64 path threads, and the choice of path does not depend on the thread
# count.  The row and column partitions below depend on the sizes only, never on the
# thread count, every element is written by exactly one chunk with the arithmetic of the
# serial kernel, and reductions across chunks follow a fixed order, so the result is
# bit-identical for any `Threads.nthreads()`.  The unblocked path is not threaded: its
# row-block sweep splits by rows, and the per-step row interchange then moves two rows'
# worth of cache lines between the cores every step, which measured slower than serial.
# The chunks go through `_lhl_foreach_chunk!(f, bk, nchunks)`, a serial loop for the
# `LHLSerial` backend; the LHLFactorizationPolyesterExt extension (loaded by `using
# Polyester`) defines its own backend type with a `@batch` method and puts an instance
# into `_LHL_BACKEND`.  Without it `thread = Val(true)` runs the serial path.  The backend
# is passed down as an argument from one dynamic call in `lhl_reduce!`, and its type lives
# in the extension: this module's image then never infers anything about it (adding the
# method would otherwise invalidate the image), and the specializations on it are cached
# in the extension's image (specializations on a type of this module would not be).
# ---------------------------------------------------------------------------
struct LHLSerial end
const _LHL_BACKEND = Ref{Any}(LHLSerial())

@inline function _lhl_foreach_chunk!(f::F, ::LHLSerial, nchunks::Int) where {F}
    for t in 1:nchunks
        f(t)
    end
    return nothing
end

_lhl_thread_flag(::Val{B}) where {B} = B
_lhl_thread_flag(b::Bool) = b
_lhl_thread_min(::Type{T}) where {T} =
    T <: Union{Float32, Float64, ComplexF32, ComplexF64} ? _lhl_block_min(T) : typemax(Int)
# Rows below which a panel step's row work stays on one thread.
_lhl_thread_rows(::Type{T}) where {T} = 128
_lhl_line(::Type{T}) where {T} = 64 ÷ sizeof(T)

# Chunks the reduction hands to `_lhl_foreach_chunk!`; 1 is the serial path.
function _lhl_nthreads(bk, thread, n::Int, ::Type{T}) where {T}
    _lhl_thread_flag(thread) || return 1
    bk isa LHLSerial && return 1
    nt = Threads.nthreads()
    (nt > 1 && n >= _lhl_thread_min(T)) || return 1
    return nt
end

# Chunk t of nt over r0:r1 in whole blocks of the absolute grid 1+g(b-1):gb (the first
# and last block clipped to r0:r1), balanced to a block; chunks past the block count are
# empty (ia > ib).
@inline function _lhl_chunk(r0::Int, r1::Int, t::Int, nt::Int, g::Int)
    b0 = (r0 - 1) ÷ g
    nb = (r1 - 1) ÷ g - b0 + 1
    q, r = divrem(nb, nt)
    a = (t - 1) * q + min(t - 1, r)
    b = t * q + min(t, r)
    a == b && return (1, 0)
    return (max(r0, 1 + (b0 + a) * g), min(r1, (b0 + b) * g))
end

function _lhl_reduce_blocked!(
        bk, A::AbstractMatrix{T}, ipiv, B0::AbstractMatrix{T},
        w::AbstractVector{T}, pack::AbstractArray{T}, nb::Int, nt::Int = 1
    ) where {T}
    n = size(A, 2)
    have = false
    @inbounds for k0 in 1:nb:(n - 2)
        kb = min(k0 + nb - 1, n - 2)
        have = _lhl_panel_steps!(bk, A, ipiv, B0, w, pack, k0, kb, nb, nt, have)
        if kb + 2 <= n
            _lhl_swap_rows!(bk, A, ipiv, k0, kb, kb + 2, n, nt)
            _lhl_trsm_block!(bk, A, k0, kb, pack, nt)
            _lhl_gemm!(bk, A, kb + 2, n, k0, kb, k0 + 1, kb + 2, kb + 2, n, -one(T), pack, nt)
        end
        _lhl_top_gemm!(bk, A, k0, kb, nb, pack, nt)
    end
    _lhl_swap_deferred!(bk, A, ipiv, nb, nt)
    return A
end

# The steps k0:kb of one panel; returns whether the pivot search of step kb+1 has been done
# ahead (`_lhl_pivot_blocks!`, pointer path only).
function _lhl_panel_steps!(
        bk, A::AbstractMatrix{T}, ipiv, B0, w, pack, k0::Int, kb::Int, nb::Int, nt::Int, have::Bool
    ) where {T}
    n = size(A, 2)
    @inbounds begin
        for j in (k0 + 1):(kb + 1)
            @simd for i in 1:n
                B0[i, j - k0] = A[i, j]
            end
        end
        for k in k0:kb
            p = k + 1
            amax = _lhl_pivmag(A[k + 1, k])
            for i in (k + 2):n
                a = _lhl_pivmag(A[i, k])
                if a > amax
                    amax = a
                    p = i
                end
            end
            ipiv[k] = p
            if p != k + 1
                A[k + 1, k], A[p, k] = A[p, k], A[k + 1, k]
                _lhl_step_swaps!(A, B0, ipiv, k0, kb, k, p)
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
            kb + 2 <= n || continue
            _lhl_gemv_panel!(bk, nt, w, A, pack, k, k0 + 1, n, kb + 2, n)
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
    return false
end

# The interchange of step k (rows/columns k+1 and p) with column k already swapped: the
# other panel columns' rows, then the columns — within the panel a plain exchange (with
# the B0 copies), otherwise the pulled column enters, the pushed one leaves in its
# pre-panel state, and the pulled column receives the panel's interchanges and left
# transforms it has missed.
function _lhl_step_swaps!(A, B0, ipiv, k0::Int, kb::Int, k::Int, p::Int)
    n = size(A, 2)
    @inbounds begin
        for j in k0:(kb + 1)
            j == k && continue
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
    return nothing
end

# Pointer path.  Per step the main thread only reads a few dozen elements and writes
# tables; two chunked regions (`_lhl_foreach_chunk!`) do the rest.  A: column groups of the trailing block scale
# their rows of the multiplier column and form their GEMV partial (`_lhl_gemv_partials!`;
# a pulled column is read from its B0 copy).  B: row chunks apply the interchange, the
# panel's left and right transforms and the sum of the partials to their rows, each also
# forms — redundantly, from the same partials — the ≤ 2nb w-values the correction's
# triangle needs, so it can finish the correction and add into column k+1 on its rows,
# and scans its rows of column k+1 for the next step's pivot (`_lhl_pivot_blocks!`), in
# blocks of eight rows the main thread combines in order.  Every partition is fixed by
# n, k0, k, so the bits do not depend on the thread count; nt = 1 runs the same code.
# Tables live in B0[:, nb+1:nb+8]:
#   nb+1 catch-up x's · nb+2 pivot row (post-swap) · nb+3/nb+4 rows k+1/p pre-swap ·
#   nb+5/nb+6 the pulled column's overridden values/rows · nb+7/nb+8 pivot blocks ·
#   nb+9 the rows of the correction's triangle and their swap partners.
# `pack` holds the GEMV partials, then per-chunk scratch, then the groups' gather table.
function _lhl_panel_steps!(
        bk, A::StridedMatrix{T}, ipiv::Vector{Int}, B0::Matrix{T}, w::Vector{T}, pack::Array{T},
        k0::Int, kb::Int, nb::Int, nt::Int, have::Bool
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 2)
    ldB = size(B0, 1)
    P = max(cld(n - kb - 1, _LHL_GEMV_GROUP), 1)
    ldp = _lhl_ld(n - k0, T)
    if stride(A, 1) != 1 || size(B0, 2) < nb + 9 || ldB < n || length(pack) < P * ldp + 4nb * nt + 2nb * P
        return invoke(
            _lhl_panel_steps!, Tuple{Any, AbstractMatrix{T}, Any, Any, Any, Any, Int, Int, Int, Int, Bool},
            bk, A, ipiv, B0, w, pack, k0, kb, nb, nt, have
        )
    end
    sz = sizeof(T)
    ld = stride(A, 2)
    g = _lhl_line(T)
    par = nt > 1 && n - k0 >= _lhl_thread_rows(T)
    GC.@preserve A B0 w pack ipiv begin
        pA = pointer(A)
        pB = pointer(B0)
        pw = pointer(w)
        pip = pointer(ipiv)
        pT = pB + nb * ldB * sz
        pCX = pT
        pPJ = pT + ldB * sz
        pR1 = pT + 2ldB * sz
        pRP = pT + 3ldB * sz
        pOV = pT + 4ldB * sz
        pOR = pT + 5ldB * sz
        pPM = pT + 6ldB * sz
        pPI = pT + 7ldB * sz
        pSR = pT + 8ldB * sz
        pP = pointer(pack) - k0 * sz
        pQ = pointer(pack) + P * ldp * sz
        pG = pQ + 4nb * nt * sz
        if par
            _lhl_foreach_chunk!(bk, nt) do t
                ia, ib = _lhl_chunk(1, n, t, nt, g)
                ia <= ib && _lhl_b0_copy!(pA, ld, pB, ldB, k0, kb, ia, ib)
            end
        else
            _lhl_b0_copy!(pA, ld, pB, ldB, k0, kb, 1, n)
        end
        @inbounds for k in k0:kb
            p = have ? _lhl_pivot_combine(pPM, pPI, k, n) : _lhl_pivot_scan(pA, ld, k, n)
            ipiv[k] = p
            for j in k0:(kb + 1)
                unsafe_store!(pR1, A[k + 1, j], j - k0 + 1)
                unsafe_store!(pRP, A[p, j], j - k0 + 1)
            end
            if p != k + 1
                A[k + 1, k], A[p, k] = A[p, k], A[k + 1, k]
            end
            piv = A[k + 1, k]
            pulled = p > kb + 1
            if iszero(piv)
                p != k + 1 && _lhl_step_swaps!(A, B0, ipiv, k0, kb, k, p)
                have = false
                continue
            end
            for i in (k + 2):(kb + 1)
                A[i, k] /= piv
            end
            nov = pulled ? _lhl_pulled_tables!(pA, ld, pRP, pOV, pOR, pCX, ipiv, k0, k, p) : 0
            for j in (k + 1):(kb + 1)
                unsafe_store!(pPJ, unsafe_load(pRP, j - k0 + 1), j - k)
            end
            if pulled
                unsafe_store!(pPJ, unsafe_load(pOV, k - k0 + 1), 1)
            elseif p != k + 1
                unsafe_store!(pPJ, unsafe_load(pRP, p - k0 + 1), 1)
                unsafe_store!(pPJ, unsafe_load(pRP, k - k0 + 2), p - k)
            end
            trailing = kb + 2 <= n
            nsp = _lhl_special_rows!(pSR, ipiv, k0, k)
            pSub = pulled ? pB + (k - k0) * ldB * sz : Ptr{T}(0)
            Pw = trailing ?
                _lhl_gemv_partials!(
                    bk, w, A, k, k0 + 1, n, kb + 2, n, pack, ldp, nt, piv, pulled ? p : 0, pSub, pSR, nsp, pG, 2nb
                ) : 0
            if par && (Pw > 0 || !trailing)
                _lhl_foreach_chunk!(bk, nt) do t
                    ia, ib = _lhl_chunk(k0 + 1, n, t, nt, g)
                    ia <= ib && _lhl_step_rows!(
                        pA, ld, pB, ldB, pw, pP, ldp, Pw, pQ + (t - 1) * 4nb * sz, pG, 2nb, pT, pip,
                        k0, kb, k, p, nov, trailing, ia, ib
                    )
                    ja, jb = _lhl_chunk(1, k0, t, nt, g)
                    ja <= jb && p != k + 1 && _lhl_exchange_rows!(pA, ld, pB, ldB, k0, kb, k, p, ja, jb)
                end
            else
                _lhl_step_rows!(pA, ld, pB, ldB, pw, pP, ldp, Pw, pQ, pG, 2nb, pT, pip, k0, kb, k, p, nov, trailing, k0 + 1, n)
                p != k + 1 && _lhl_exchange_rows!(pA, ld, pB, ldB, k0, kb, k, p, 1, k0)
            end
            have = true
        end
    end
    return have
end

@inline function _lhl_b0_copy!(pA::Ptr{T}, ld::Int, pB::Ptr{T}, ldB::Int, k0::Int, kb::Int, ia::Int, ib::Int) where {T}
    sz = sizeof(T)
    for j in (k0 + 1):(kb + 1)
        pa = pA + (j - 1) * ld * sz
        pb = pB + (j - k0 - 1) * ldB * sz
        @simd ivdep for i in ia:ib
            unsafe_store!(pb, unsafe_load(pa, i), i)
        end
    end
    return nothing
end

# Serial pivot search over column k: the first row of the largest |A[i, k]|, i ≥ k+1.
@inline function _lhl_pivot_scan(pA::Ptr{T}, ld::Int, k::Int, n::Int) where {T}
    pc = pA + (k - 1) * ld * sizeof(T)
    p = k + 1
    amax = abs(unsafe_load(pc, k + 1))
    for i in (k + 2):n
        a = abs(unsafe_load(pc, i))
        if a > amax
            amax = a
            p = i
        end
    end
    return p
end

# The same over the blocks of eight rows (block b = rows 8b-7:8b) a step's row chunks
# scanned ahead: block b holds its first-largest |A[i, k]| and row, rows < k+1 excluded,
# empty blocks -1.
@inline function _lhl_pivot_combine(pPM::Ptr{T}, pPI::Ptr{T}, k::Int, n::Int) where {T}
    b0 = k ÷ 8 + 1
    b1 = (n - 1) ÷ 8 + 1
    amax = unsafe_load(pPM, b0)
    p = Int(unsafe_load(pPI, b0))
    for b in (b0 + 1):b1
        a = unsafe_load(pPM, b)
        if a > amax
            amax = a
            p = Int(unsafe_load(pPI, b))
        end
    end
    return p
end

# Row chunk ia:ib of column c = k+1 (rows ≥ k+2, the first as the initial candidate)
# into the pivot blocks.
@inline function _lhl_pivot_blocks!(pA::Ptr{T}, ld::Int, pPM::Ptr{T}, pPI::Ptr{T}, k::Int, ia::Int, ib::Int) where {T}
    pc = pA + k * ld * sizeof(T)
    lo = max(ia, k + 2)
    b = (lo - 1) ÷ 8 + 1
    i = lo
    while i <= ib
        ie = min(8b, ib)
        amax = -one(T)
        idx = 0
        if i == k + 2
            amax = abs(unsafe_load(pc, i))
            idx = i
            i += 1
        end
        while i <= ie
            a = abs(unsafe_load(pc, i))
            if a > amax
                amax = a
                idx = i
            end
            i += 1
        end
        unsafe_store!(pPM, amax, b)
        unsafe_store!(pPI, T(idx), b)
        b += 1
    end
    return nothing
end

# The pulled column p's values at the rows the panel's interchanges touch, after those
# interchanges and the left transforms of steps k0:k-1 on rows ≤ k+1 (rows k+1 of the
# multiplier columns come from the pre-swap row p); writes the value/row table and the
# catch-up x's, returns the table length.
function _lhl_pulled_tables!(
        pA::Ptr{T}, ld::Int, pRP::Ptr{T}, pOV::Ptr{T}, pOR::Ptr{T}, pCX::Ptr{T},
        ipiv::Vector{Int}, k0::Int, k::Int, p::Int
    ) where {T}
    sz = sizeof(T)
    lds = ld * sz
    pc = pA + (p - 1) * lds
    m = k - k0 + 1
    @inbounds for kk in k0:k
        unsafe_store!(pOV, unsafe_load(pc, kk + 1), kk - k0 + 1)
        unsafe_store!(pOR, T(kk + 1), kk - k0 + 1)
    end
    nov = m
    @inbounds for kk in k0:k
        q = ipiv[kk]
        q == kk + 1 && continue
        s = 0
        for r in 1:nov
            if Int(unsafe_load(pOR, r)) == q
                s = r
                break
            end
        end
        if s == 0
            nov += 1
            s = nov
            unsafe_store!(pOV, unsafe_load(pc, q), s)
            unsafe_store!(pOR, T(q), s)
        end
        a = unsafe_load(pOV, kk - k0 + 1)
        unsafe_store!(pOV, unsafe_load(pOV, s), kk - k0 + 1)
        unsafe_store!(pOV, a, s)
    end
    for kk in k0:(k - 1)
        x = unsafe_load(pOV, kk - k0 + 1)
        unsafe_store!(pCX, x, kk - k0 + 1)
        iszero(x) && continue
        pkk = pA + (kk - 1) * lds
        for i in (kk + 2):(k + 1)
            aik = i == k + 1 ? unsafe_load(pRP, kk - k0 + 1) : unsafe_load(pkk, i)
            unsafe_store!(pOV, unsafe_load(pOV, i - k0) - aik * x, i - k0)
        end
    end
    return nov
end

# Rows ia:ib (⊂ 1:k0) of the interchange's column part.
@inline function _lhl_exchange_rows!(
        pA::Ptr{T}, ld::Int, pB::Ptr{T}, ldB::Int, k0::Int, kb::Int, k::Int, p::Int, ia::Int, ib::Int
    ) where {T}
    sz = sizeof(T)
    pk1 = pA + k * ld * sz
    pcp = pA + (p - 1) * ld * sz
    pb1 = pB + (k - k0) * ldB * sz
    if p <= kb + 1
        pbp = pB + (p - k0 - 1) * ldB * sz
        @simd ivdep for i in ia:ib
            a = unsafe_load(pk1, i)
            unsafe_store!(pk1, unsafe_load(pcp, i), i)
            unsafe_store!(pcp, a, i)
            b = unsafe_load(pb1, i)
            unsafe_store!(pb1, unsafe_load(pbp, i), i)
            unsafe_store!(pbp, b, i)
        end
    else
        @simd ivdep for i in ia:ib
            t = unsafe_load(pcp, i)
            unsafe_store!(pcp, unsafe_load(pb1, i), i)
            unsafe_store!(pb1, t, i)
            unsafe_store!(pk1, t, i)
        end
    end
    return nothing
end

# Batch B's row kernel, rows ia:ib ⊂ k0+1:n; see `_lhl_panel_steps!`.  pQ is this chunk's
# private scratch (4nb values), pT the table base (B0[:, nb+1]).  Row k+1 of the
# multiplier columns is read from the pre-swap row p table: its owner may be swapping it.
function _lhl_step_rows!(
        pA::Ptr{T}, ld::Int, pB::Ptr{T}, ldB::Int, pw::Ptr{T}, pP::Ptr{T}, ldp::Int, P::Int,
        pQ::Ptr{T}, pG::Ptr{T}, ldg::Int, pT::Ptr{T}, pip::Ptr{Int}, k0::Int, kb::Int, k::Int,
        p::Int, nov::Int, trailing::Bool, ia::Int, ib::Int
    ) where {T}
    sz = sizeof(T)
    lds = ld * sz
    ldBs = ldB * sz
    pCX = pT
    pPJ = pT + ldBs
    pR1 = pT + 2ldBs
    pRP = pT + 3ldBs
    pOV = pT + 4ldBs
    pOR = pT + 5ldBs
    pPM = pT + 6ldBs
    pPI = pT + 7ldBs
    pk = pA + (k - 1) * lds
    pk1 = pk + lds
    if p != k + 1
        for j in k0:(kb + 1)
            j == k && continue
            pc = pA + (j - 1) * lds
            ia <= k + 1 <= ib && unsafe_store!(pc, unsafe_load(pRP, j - k0 + 1), k + 1)
            ia <= p <= ib && unsafe_store!(pc, unsafe_load(pR1, j - k0 + 1), p)
        end
        _lhl_exchange_rows!(pA, ld, pB, ldB, k0, kb, k, p, ia, ib)
    end
    il = max(ia, k + 2)
    if p > kb + 1
        for r in 1:nov
            i = Int(unsafe_load(pOR, r))
            ia <= i <= ib && unsafe_store!(pk1, unsafe_load(pOV, r), i)
        end
        for kk in k0:(k - 1)
            x = unsafe_load(pCX, kk - k0 + 1)
            iszero(x) && continue
            pkk = pA + (kk - 1) * lds
            @simd ivdep for i in il:ib
                unsafe_store!(pk1, unsafe_load(pk1, i) - unsafe_load(pkk, i) * x, i)
            end
        end
    end
    for j in (k + 1):(kb + 1)
        pj = unsafe_load(pPJ, j - k)
        iszero(pj) && continue
        pc = pA + (j - 1) * lds
        @simd ivdep for i in il:ib
            unsafe_store!(pc, unsafe_load(pc, i) - unsafe_load(pk, i) * pj, i)
        end
    end
    for j in (k + 2):(kb + 1)
        vj = unsafe_load(pk, j)
        iszero(vj) && continue
        pc = pA + (j - 1) * lds
        @simd ivdep for i in ia:ib
            unsafe_store!(pk1, unsafe_load(pk1, i) + vj * unsafe_load(pc, i), i)
        end
    end
    if trailing
        P > 0 && _lhl_sum_partials!(pw, pP, ldp, P, ia, ib)
        m = k - k0 + 1
        pLW = pQ                        # w at rows k0+1:k+1
        pLQ = pQ + m * sz               # w at the swap partners outside k0+1:k+1
        pLR = pLQ + m * sz              # their rows
        pXS = pLR + m * sz              # the correction's x's
        for i in (k0 + 1):(k + 1)
            unsafe_store!(pLW, _lhl_special_sum(pw, pG, ldg, P, i - k0, i), i - k0)
        end
        nq = 0
        for kk in k0:k
            q = unsafe_load(pip, kk)
            q == kk + 1 && continue
            if q <= k + 1
                a = unsafe_load(pLW, kk - k0 + 1)
                unsafe_store!(pLW, unsafe_load(pLW, q - k0), kk - k0 + 1)
                unsafe_store!(pLW, a, q - k0)
            else
                s = 0
                for r in 1:nq
                    if Int(unsafe_load(pLR, r)) == q
                        s = r
                        break
                    end
                end
                if s == 0
                    nq += 1
                    s = nq
                    unsafe_store!(pLQ, _lhl_special_sum(pw, pG, ldg, P, m + s, q), s)
                    unsafe_store!(pLR, T(q), s)
                end
                a = unsafe_load(pLW, kk - k0 + 1)
                unsafe_store!(pLW, unsafe_load(pLQ, s), kk - k0 + 1)
                unsafe_store!(pLQ, a, s)
            end
        end
        for kk in k0:k
            x = unsafe_load(pLW, kk - k0 + 1)
            unsafe_store!(pXS, x, kk - k0 + 1)
            iszero(x) && continue
            pkk = pA + (kk - 1) * lds
            for i in (kk + 2):k
                unsafe_store!(pLW, unsafe_load(pLW, i - k0) - unsafe_load(pkk, i) * x, i - k0)
            end
            if kk + 2 <= k + 1
                unsafe_store!(pLW, unsafe_load(pLW, k + 1 - k0) - unsafe_load(pRP, kk - k0 + 1) * x, k + 1 - k0)
            end
        end
        for i in max(ia, k0 + 1):min(ib, k + 1)
            v = unsafe_load(pLW, i - k0)
            unsafe_store!(pw, v, i)
            unsafe_store!(pk1, unsafe_load(pk1, i) + v, i)
        end
        for r in 1:nq
            i = Int(unsafe_load(pLR, r))
            ia <= i <= ib && unsafe_store!(pw, unsafe_load(pLQ, r), i)
        end
        for kk in k0:k
            x = unsafe_load(pXS, kk - k0 + 1)
            iszero(x) && continue
            pkk = pA + (kk - 1) * lds
            @simd ivdep for i in il:ib
                unsafe_store!(pw, unsafe_load(pw, i) - unsafe_load(pkk, i) * x, i)
            end
        end
        @simd ivdep for i in il:ib
            unsafe_store!(pk1, unsafe_load(pk1, i) + unsafe_load(pw, i), i)
        end
    end
    _lhl_pivot_blocks!(pA, ld, pPM, pPI, k, ia, ib)
    return nothing
end

# Rows k0+1:k+1, then the interchange partners of steps k0:k outside them, first
# occurrence order; returns the count.
@inline function _lhl_special_rows!(pSR::Ptr{T}, ipiv::Vector{Int}, k0::Int, k::Int) where {T}
    m = k - k0 + 1
    for r in 1:m
        unsafe_store!(pSR, T(k0 + r), r)
    end
    nsp = m
    @inbounds for kk in k0:k
        q = ipiv[kk]
        q <= k + 1 && continue
        found = false
        for r in (m + 1):nsp
            if Int(unsafe_load(pSR, r)) == q
                found = true
                break
            end
        end
        if !found
            nsp += 1
            unsafe_store!(pSR, T(q), nsp)
        end
    end
    return nsp
end

# w at special row i (slot r of the gather table): the P groups' values summed in order,
# or w itself when the serial path already holds the sum (P = 0).
@inline function _lhl_special_sum(pw::Ptr{T}, pG::Ptr{T}, ldg::Int, P::Int, r::Int, i::Int) where {T}
    P == 0 && return unsafe_load(pw, i)
    s = unsafe_load(pG, r)
    for p in 2:P
        s += unsafe_load(pG + (p - 1) * ldg * sizeof(T), r)
    end
    return s
end

# The row interchanges of steps k0:kb applied to columns c0:c1.
function _lhl_swap_rows!(bk, A::AbstractMatrix, ipiv, k0::Int, kb::Int, c0::Int, c1::Int, nt::Int)
    @inbounds for c in c0:c1, k in k0:kb
        p = ipiv[k]
        p != k + 1 && ((A[k + 1, c], A[p, c]) = (A[p, c], A[k + 1, c]))
    end
    return nothing
end

function _lhl_swap_rows!(
        bk, A::StridedMatrix{T}, ipiv::Vector{Int}, k0::Int, kb::Int, c0::Int, c1::Int, nt::Int
    ) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    if nt == 1 || stride(A, 1) != 1 || c1 - c0 + 1 <= _LHL_GEMV_GROUP
        return invoke(
            _lhl_swap_rows!, Tuple{Any, AbstractMatrix, Any, Int, Int, Int, Int, Int},
            bk, A, ipiv, k0, kb, c0, c1, nt
        )
    end
    GC.@preserve A ipiv begin
        pA = pointer(A)
        pip = pointer(ipiv)
        ld = stride(A, 2)
        P = cld(c1 - c0 + 1, _LHL_GEMV_GROUP)
        _lhl_foreach_chunk!(bk, P) do p
            ca, cb = _lhl_group(c0, c1, p)
            _lhl_swap_cols_ptr!(pA, ld, pip, k0, kb, ca, cb)
        end
    end
    return nothing
end

@inline function _lhl_swap_cols_ptr!(pA::Ptr{T}, ld::Int, pip::Ptr{Int}, k0::Int, kb::Int, ca::Int, cb::Int) where {T}
    for c in ca:cb
        pc = pA + (c - 1) * ld * sizeof(T)
        for k in k0:kb
            p = unsafe_load(pip, k)
            if p != k + 1
                x = unsafe_load(pc, k + 1)
                unsafe_store!(pc, unsafe_load(pc, p), k + 1)
                unsafe_store!(pc, x, p)
            end
        end
    end
    return nothing
end

# The interchanges deferred on the multipliers: column c of panel k0:kb receives those of
# every later step.
function _lhl_swap_deferred!(bk, A::AbstractMatrix, ipiv, nb::Int, nt::Int)
    n = size(A, 2)
    @inbounds for k0 in 1:nb:(n - 2)
        kb = min(k0 + nb - 1, n - 2)
        for c in k0:kb, k in (kb + 1):(n - 2)
            p = ipiv[k]
            p != k + 1 && ((A[k + 1, c], A[p, c]) = (A[p, c], A[k + 1, c]))
        end
    end
    return nothing
end

function _lhl_swap_deferred!(
        bk, A::StridedMatrix{T}, ipiv::Vector{Int}, nb::Int, nt::Int
    ) where {T <: Union{Float32, Float64, ComplexF32, ComplexF64}}
    n = size(A, 2)
    if nt == 1 || stride(A, 1) != 1 || n - 2 < nb * nt
        return invoke(_lhl_swap_deferred!, Tuple{Any, AbstractMatrix, Any, Int, Int}, bk, A, ipiv, nb, nt)
    end
    GC.@preserve A ipiv begin
        pA = pointer(A)
        pip = pointer(ipiv)
        ld = stride(A, 2)
        _lhl_foreach_chunk!(bk, nt) do t
            ca, cb = _lhl_chunk(1, n - 2, t, nt, nb)
            for c in ca:cb
                kbc = min(((c - 1) ÷ nb + 1) * nb, n - 2)
                _lhl_swap_cols_ptr!(pA, ld, pip, kbc + 1, n - 2, c, c)
            end
        end
    end
    return nothing
end

# w[r0:r1] = Σ_{j=c0}^{c1} A[j, k] * A[r0:r1, j]
function _lhl_gemv_cols!(w, A::AbstractMatrix{T}, k::Int, r0::Int, r1::Int, c0::Int, c1::Int) where {T}
    @inbounds for i in r0:r1
        w[i] = zero(T)
    end
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

# The pointer path cuts the columns into groups of `_LHL_GEMV_GROUP`, forms each group's
# product separately and adds the group sums in order, w = ((g₁ + g₂) + g₃) + ….  Threads
# take groups and leave the P partials (rows r0:r1, stride ldp, in `pack`) for the caller's
# row chunks to sum, returning P; the serial path sums group by group into `w` through
# one scratch vector and returns 0.  A group's column set depends on the sizes only, so
# both give the same bits.  Each group first scales its rows of the multiplier column by 1/piv;
# column psub (a column pulled into the panel this step) is read from pSub instead.  The
# threads also copy their partial at the nsp rows listed in pSR into their row of the
# gather table pG (stride ldg), for the row chunks' redundant triangle.
const _LHL_GEMV_GROUP = 64
@inline _lhl_group(c0::Int, c1::Int, p::Int) =
    (c0 + (p - 1) * _LHL_GEMV_GROUP, min(c0 + p * _LHL_GEMV_GROUP - 1, c1))
function _lhl_gemv_partials!(
        bk, w::Vector{T}, A::StridedMatrix{T}, k::Int, r0::Int, r1::Int, c0::Int, c1::Int,
        pack::Array{T}, ldp::Int, nt::Int, piv::T, psub::Int, pSub::Ptr{T}, pSR::Ptr{T},
        nsp::Int, pG::Ptr{T}, ldg::Int
    ) where {T <: Union{Float32, Float64}}
    P = cld(c1 - c0 + 1, _LHL_GEMV_GROUP)
    sz = sizeof(T)
    GC.@preserve w A pack begin
        pw = pointer(w)
        pA = pointer(A)
        ld = stride(A, 2)
        pP = pointer(pack) - (r0 - 1) * sz
        if nt > 1 && P > 1
            _lhl_foreach_chunk!(bk, P) do p
                ga, gb = _lhl_group(c0, c1, p)
                pp = pP + (p - 1) * ldp * sz
                _lhl_gemv_group!(pp, pA, ld, k, r0, r1, ga, gb, piv, psub, pSub)
                pg = pG + (p - 1) * ldg * sz
                for r in 1:nsp
                    unsafe_store!(pg, unsafe_load(pp, Int(unsafe_load(pSR, r))), r)
                end
            end
            return P
        end
        ca, cb = _lhl_group(c0, c1, 1)
        _lhl_gemv_group!(pw, pA, ld, k, r0, r1, ca, cb, piv, psub, pSub)
        for p in 2:P
            ca, cb = _lhl_group(c0, c1, p)
            _lhl_gemv_group!(pP, pA, ld, k, r0, r1, ca, cb, piv, psub, pSub)
            @simd ivdep for i in r0:r1
                unsafe_store!(pw, unsafe_load(pw, i) + unsafe_load(pP, i), i)
            end
        end
    end
    return 0
end

@inline _lhl_colp(pA::Ptr{T}, lds::Int, j::Int, psub::Int, pSub::Ptr{T}) where {T} =
    j == psub ? pSub : pA + (j - 1) * lds

# A[c0:c1, k] /= piv, then w[r0:r1] = Σ_{j=c0}^{c1} A[j, k] * A[r0:r1, j], c0 ≤ c1, four
# columns per pass; the first pass assigns.
@inline function _lhl_gemv_group!(
        pw::Ptr{T}, pA::Ptr{T}, ld::Int, k::Int, r0::Int, r1::Int, c0::Int, c1::Int,
        piv::T, psub::Int, pSub::Ptr{T}
    ) where {T}
    sz = sizeof(T)
    lds = ld * sz
    pl = pA + (k - 1) * lds
    for j in c0:c1
        unsafe_store!(pl, unsafe_load(pl, j) / piv, j)
    end
    j = c0
    if j + 3 <= c1
        l1 = unsafe_load(pl, j); l2 = unsafe_load(pl, j + 1)
        l3 = unsafe_load(pl, j + 2); l4 = unsafe_load(pl, j + 3)
        p1 = _lhl_colp(pA, lds, j, psub, pSub); p2 = _lhl_colp(pA, lds, j + 1, psub, pSub)
        p3 = _lhl_colp(pA, lds, j + 2, psub, pSub); p4 = _lhl_colp(pA, lds, j + 3, psub, pSub)
        @simd ivdep for i in r0:r1
            unsafe_store!(
                pw, (l1 * unsafe_load(p1, i) + l2 * unsafe_load(p2, i)) +
                    (l3 * unsafe_load(p3, i) + l4 * unsafe_load(p4, i)), i
            )
        end
        j += 4
    else
        lj = unsafe_load(pl, j)
        p1 = _lhl_colp(pA, lds, j, psub, pSub)
        @simd ivdep for i in r0:r1
            unsafe_store!(pw, lj * unsafe_load(p1, i), i)
        end
        j += 1
    end
    while j + 3 <= c1
        l1 = unsafe_load(pl, j); l2 = unsafe_load(pl, j + 1)
        l3 = unsafe_load(pl, j + 2); l4 = unsafe_load(pl, j + 3)
        p1 = _lhl_colp(pA, lds, j, psub, pSub); p2 = _lhl_colp(pA, lds, j + 1, psub, pSub)
        p3 = _lhl_colp(pA, lds, j + 2, psub, pSub); p4 = _lhl_colp(pA, lds, j + 3, psub, pSub)
        @simd ivdep for i in r0:r1
            unsafe_store!(
                pw, unsafe_load(pw, i) + (
                    (l1 * unsafe_load(p1, i) + l2 * unsafe_load(p2, i)) +
                        (l3 * unsafe_load(p3, i) + l4 * unsafe_load(p4, i))
                ), i
            )
        end
        j += 4
    end
    while j <= c1
        lj = unsafe_load(pl, j)
        if !iszero(lj)
            p1 = _lhl_colp(pA, lds, j, psub, pSub)
            @simd ivdep for i in r0:r1
                unsafe_store!(pw, unsafe_load(pw, i) + lj * unsafe_load(p1, i), i)
            end
        end
        j += 1
    end
    return nothing
end

# w[ia:ib] = ((g₁ + g₂) + g₃) + … over the P partials at stride ldp, one vector of rows
# at a time so that the P loads of a row are in flight together (the partials sit in other
# cores' caches).
@inline function _lhl_sum_partials!(pw::Ptr{T}, pP::Ptr{T}, ldp::Int, P::Int, ia::Int, ib::Int) where {T}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    lps = ldp * sz
    i = ia
    while i + W - 1 <= ib
        q = pP + (i - 1) * sz
        acc = _lhl_vload(V, q)
        for p in 2:P
            acc = _lhl_vadd(acc, _lhl_vload(V, q + (p - 1) * lps))
        end
        _lhl_vstore!(pw + (i - 1) * sz, acc)
        i += W
    end
    while i <= ib
        q = pP + (i - 1) * sz
        s = unsafe_load(q)
        for p in 2:P
            s += unsafe_load(q + (p - 1) * lps)
        end
        unsafe_store!(pw, s, i)
        i += 1
    end
    return nothing
end

# The generic panel's per-step GEMV.  The fallback is the plain column sweep; complex
# float goes through fixed 64-column groups whose partials are combined in group order —
# the partition and the order depend on the sizes only, so any thread count (including 1)
# gives the same bits.
function _lhl_gemv_panel!(
        bk, nt::Int, w, A::AbstractMatrix{T}, pack, k::Int, r0::Int, r1::Int, c0::Int, c1::Int
    ) where {T}
    _lhl_gemv_cols!(w, A, k, r0, r1, c0, c1)
    return nothing
end

function _lhl_gemv_panel!(
        bk, nt::Int, w::Vector{T}, A::StridedMatrix{T}, pack::Vector{T}, k::Int,
        r0::Int, r1::Int, c0::Int, c1::Int
    ) where {T <: _LHL_CFloat}
    if stride(A, 1) != 1
        _lhl_gemv_cols!(w, A, k, r0, r1, c0, c1)
        return nothing
    end
    Tr = real(T)
    P = cld(c1 - c0 + 1, _LHL_GEMV_GROUP)
    ldq = size(A, 2) + 8
    sz = sizeof(Tr)
    GC.@preserve w A pack begin
        pw = Ptr{Tr}(pointer(w)) + 2 * (r0 - 1) * sz
        pA = Ptr{Tr}(pointer(A))
        ld = 2 * stride(A, 2)
        pP = Ptr{Tr}(pointer(pack))
        if nt > 1 && P > 1
            _lhl_foreach_chunk!(bk, P) do p
                ca, cb = _lhl_group(c0, c1, p)
                _lhl_cgemv_group!(pP + 2 * (p - 1) * ldq * sz, pA, ld, k, r0, r1, ca, cb)
            end
        else
            for p in 1:P
                ca, cb = _lhl_group(c0, c1, p)
                _lhl_cgemv_group!(pP + 2 * (p - 1) * ldq * sz, pA, ld, k, r0, r1, ca, cb)
            end
        end
        # w = ((g₁ + g₂) + g₃) + … in group order
        m = 2 * (r1 - r0 + 1)
        @inbounds @simd ivdep for t in 1:m
            unsafe_store!(pw, unsafe_load(pP, t), t)
        end
        for p in 2:P
            pq = pP + 2 * (p - 1) * ldq * sz
            @inbounds @simd ivdep for t in 1:m
                unsafe_store!(pw, unsafe_load(pw, t) + unsafe_load(pq, t), t)
            end
        end
    end
    return nothing
end

# One group's partial q[1:r1-r0+1] = Σ_{j=ca}^{cb} A[j, k] · A[r0:r1, j] on the real view,
# four columns per pass, the first pass assigning.  The four multipliers broadcast as
# re/im scalars; the imaginary halves accumulate separately and are swap-negated into the
# result once per vector (see the complex-kernel header comment).
@inline function _lhl_cgemv_group!(
        pq::Ptr{T}, pA::Ptr{T}, ld::Int, k::Int, r0::Int, r1::Int, ca::Int, cb::Int
    ) where {T}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    lds = ld * sz
    pk = pA + (k - 1) * lds
    rl = 2 * (r1 - r0 + 1)
    mb = (rl - rl % W) * sz
    off = 2 * (r0 - 1) * sz
    j = ca
    if j + 3 <= cb
        _lhl_cgemv_quad!(Val(true), V, pq, pA, lds, pk, j, off, mb, rl)
        j += 4
    else
        _lhl_cgemv_one!(Val(true), V, pq, pA, lds, pk, j, off, mb, rl)
        j += 1
    end
    while j + 3 <= cb
        _lhl_cgemv_quad!(Val(false), V, pq, pA, lds, pk, j, off, mb, rl)
        j += 4
    end
    while j <= cb
        _lhl_cgemv_one!(Val(false), V, pq, pA, lds, pk, j, off, mb, rl)
        j += 1
    end
    return nothing
end

@inline _lhl_cmul(j::Int, pk::Ptr{T}) where {T} =
    Complex(unsafe_load(pk, 2j - 1), unsafe_load(pk, 2j))

@inline function _lhl_cgemv_quad!(
        ::Val{F}, ::Type{V}, pq::Ptr{T}, pA::Ptr{T}, lds::Int, pk::Ptr{T}, j::Int,
        off::Int, mb::Int, rl::Int
    ) where {F, W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    l1 = _lhl_cmul(j, pk); l2 = _lhl_cmul(j + 1, pk)
    l3 = _lhl_cmul(j + 2, pk); l4 = _lhl_cmul(j + 3, pk)
    br1 = _lhl_bcast(V, real(l1)); bi1 = _lhl_bcast(V, imag(l1))
    br2 = _lhl_bcast(V, real(l2)); bi2 = _lhl_bcast(V, imag(l2))
    br3 = _lhl_bcast(V, real(l3)); bi3 = _lhl_bcast(V, imag(l3))
    br4 = _lhl_bcast(V, real(l4)); bi4 = _lhl_bcast(V, imag(l4))
    p1 = pA + (j - 1) * lds + off
    p2 = p1 + lds
    p3 = p2 + lds
    p4 = p3 + lds
    z = _lhl_bcast(V, zero(T))
    d = 0
    while d < mb
        a1 = _lhl_vload(V, p1 + d); a2 = _lhl_vload(V, p2 + d)
        a3 = _lhl_vload(V, p3 + d); a4 = _lhl_vload(V, p4 + d)
        t1 = _lhl_fma(a2, br2, _lhl_fma(a1, br1, z))
        t1 = _lhl_fma(a4, br4, _lhl_fma(a3, br3, t1))
        t2 = _lhl_fma(a2, bi2, _lhl_fma(a1, bi1, z))
        t2 = _lhl_fma(a4, bi4, _lhl_fma(a3, bi3, t2))
        t1 = _lhl_vadd(t1, _lhl_vswapneg(t2))
        F || (t1 = _lhl_vadd(_lhl_vload(V, pq + d), t1))
        _lhl_vstore!(pq + d, t1)
        d += W * sz
    end
    t = mb ÷ (2 * sz) + 1
    while 2t <= rl
        a = _lhl_cmulld(p1, t) * l1 + _lhl_cmulld(p2, t) * l2 +
            _lhl_cmulld(p3, t) * l3 + _lhl_cmulld(p4, t) * l4
        F || (a += _lhl_cmulld(pq, t))
        unsafe_store!(pq, real(a), 2t - 1)
        unsafe_store!(pq, imag(a), 2t)
        t += 1
    end
    return nothing
end

@inline function _lhl_cgemv_one!(
        ::Val{F}, ::Type{V}, pq::Ptr{T}, pA::Ptr{T}, lds::Int, pk::Ptr{T}, j::Int,
        off::Int, mb::Int, rl::Int
    ) where {F, W, T, V <: NTuple{W, VecElement{T}}}
    sz = sizeof(T)
    l1 = _lhl_cmul(j, pk)
    if !F && iszero(l1)
        return nothing
    end
    br1 = _lhl_bcast(V, real(l1))
    bi1 = _lhl_bcast(V, imag(l1))
    p1 = pA + (j - 1) * lds + off
    z = _lhl_bcast(V, zero(T))
    d = 0
    while d < mb
        a1 = _lhl_vload(V, p1 + d)
        t1 = _lhl_fma(a1, br1, z)
        t2 = _lhl_fma(a1, bi1, z)
        t1 = _lhl_vadd(t1, _lhl_vswapneg(t2))
        F || (t1 = _lhl_vadd(_lhl_vload(V, pq + d), t1))
        _lhl_vstore!(pq + d, t1)
        d += W * sz
    end
    t = mb ÷ (2 * sz) + 1
    while 2t <= rl
        a = _lhl_cmulld(p1, t) * l1
        F || (a += _lhl_cmulld(pq, t))
        unsafe_store!(pq, real(a), 2t - 1)
        unsafe_store!(pq, imag(a), 2t)
        t += 1
    end
    return nothing
end

@inline _lhl_cmulld(p::Ptr{T}, t::Int) where {T} =
    Complex(unsafe_load(p, 2t - 1), unsafe_load(p, 2t))

# The deferred right updates of rows 1:k0 of the panel columns: from the panel columns
# themselves, then A[1:k0, k0+1:kb+1] += A[1:k0, kb+2:n] * A[kb+2:n, k0:kb], one nb-wide K
# chunk at a time.
function _lhl_top_gemm!(bk, A::AbstractMatrix{T}, k0::Int, kb::Int, nb::Int, pack, nt::Int) where {T}
    n = size(A, 2)
    _lhl_top_fix!(A, k0, kb, 1, k0)
    for kk in (kb + 2):nb:n
        ke = min(kk + nb - 1, n)
        _lhl_gemm!(bk, A, 1, k0, kk, ke, kk, k0, k0 + 1, kb + 1, one(T), pack, nt)
    end
    return nothing
end

@inline function _lhl_top_fix!(A::AbstractMatrix, k0::Int, kb::Int, ia::Int, ib::Int)
    @inbounds for k in k0:kb, j in (k + 2):(kb + 1)
        vj = A[j, k]
        iszero(vj) && continue
        @simd for i in ia:ib
            A[i, k + 1] += vj * A[i, j]
        end
    end
    return nothing
end

# Same in K chunks of 64: with only nb output columns, packing the k0×K left operand would
# cost as much as the product, so the tile reads it in place in row blocks small enough
# that a block's chunk stays in L1/L2.  Threads take row chunks of whole tiles, each doing
# its rows' panel-column part first.
function _lhl_top_gemm!(
        bk, A::StridedMatrix{T}, k0::Int, kb::Int, nb::Int, pack::Array{T}, nt::Int
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 2)
    if stride(A, 1) != 1
        return invoke(_lhl_top_gemm!, Tuple{Any, AbstractMatrix{T}, Int, Int, Int, Any, Int}, bk, A, k0, kb, nb, pack, nt)
    end
    mr = 3 * (_LHL_VEC_BYTES ÷ sizeof(T))
    ld = stride(A, 2)
    GC.@preserve A begin
        pA = pointer(A)
        if nt > 1 && k0 >= 4mr
            _lhl_foreach_chunk!(bk, nt) do t
                ia, ib = _lhl_chunk(1, k0, t, nt, mr)
                ia <= ib && _lhl_top_gemm_rows!(pA, ld, k0, kb, n, ia, ib)
            end
        else
            _lhl_top_gemm_rows!(pA, ld, k0, kb, n, 1, k0)
        end
    end
    return nothing
end

@inline function _lhl_top_gemm_rows!(pA::Ptr{T}, ld::Int, k0::Int, kb::Int, n::Int, ia::Int, ie0::Int) where {T}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    mr = 3W
    rowblock = 4mr
    sz = sizeof(T)
    lds = ld * sz
    for k in k0:kb
        pk = pA + (k - 1) * lds
        for j in (k + 2):(kb + 1)
            vj = unsafe_load(pk, j)
            iszero(vj) && continue
            pc = pA + (j - 1) * lds
            @simd ivdep for i in ia:ie0
                unsafe_store!(pk + lds, unsafe_load(pk + lds, i) + vj * unsafe_load(pc, i), i)
            end
        end
    end
    kb + 2 <= n || return nothing
    j = kb + 2
    while j <= n
        K = min(64, n - j + 1)
        pP = pA + (j - 1) * ld * sz
        pB = pA + (j - 1 - ld) * sz
        ib = ia
        while ib <= ie0
            ie = min(ib + rowblock - 1, ie0)
            rfull = ib + ((ie - ib + 1) ÷ mr) * mr - 1
            ct = _lhl_micro_tile!(V, pA, ld, pP, ld, pB, ld, K, 1, ib, rfull, k0 + 1, kb + 1)
            _lhl_micro_edge!(V, pA, ld, pP, ld, pB, ld, K, 1, rfull + 1, ie, k0 + 1, kb + 1)
            _lhl_micro_edge!(V, pA, ld, pP, ld, pB, ld, K, 1, ib, rfull, ct, kb + 1)
            ib = ie + 1
        end
        j += K
    end
    return nothing
end

# A[i0:i1, c0:c1] += sgn * A[i0:i1, j0:j1] * A[r0:r0+K-1, cB:cB+(c1-c0)],  K = j1 - j0 + 1.
# The three blocks must not overlap.
function _lhl_gemm!(
        bk, A::AbstractMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack, nt::Int
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
        bk, A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}, nt::Int
    ) where {T <: Union{Float32, Float64}}
    (i1 < i0 || j1 < j0 || c1 < c0) && return nothing
    if stride(A, 1) == 1
        _lhl_gemm_micro!(bk, A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack, nt)
    else
        invoke(
            _lhl_gemm!, Tuple{Any, AbstractMatrix{T}, Int, Int, Int, Int, Int, Int, Int, Int, T, Any, Int},
            bk, A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack, nt
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

# Threads take the GEMV's column groups (whole 4-column tiles); every thread reads all of P.
function _lhl_gemm_micro!(
        bk, A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}, nt::Int
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
    ld = stride(A, 2)
    sz = sizeof(T)
    GC.@preserve A pack begin
        pA = pointer(A)
        pP = pointer(pack)
        pB = pA + ((cB - c0) * ld + (r0 - 1)) * sz
        if nt > 1 && c1 - c0 + 1 > _LHL_GEMV_GROUP
            P = cld(c1 - c0 + 1, _LHL_GEMV_GROUP)
            _lhl_foreach_chunk!(bk, P) do p
                ca, cb = _lhl_group(c0, c1, p)
                _lhl_gemm_tiles!(pA, ld, pP, ldp, pB, K, i0, i1, ca, cb)
            end
        else
            _lhl_gemm_tiles!(pA, ld, pP, ldp, pB, K, i0, i1, c0, c1)
        end
    end
    return nothing
end

@inline function _lhl_gemm_tiles!(
        pA::Ptr{T}, ld::Int, pP::Ptr{T}, ldp::Int, pB::Ptr{T}, K::Int, i0::Int, i1::Int,
        c0::Int, c1::Int
    ) where {T}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    mr = 3W
    rowblock = 384
    ib = i0
    while ib <= i1
        ie = min(ib + rowblock - 1, i1)
        rfull = ib + ((ie - ib + 1) ÷ mr) * mr - 1
        ct = _lhl_micro_tile!(V, pA, ld, pP, ldp, pB, ld, K, i0, ib, rfull, c0, c1)
        _lhl_micro_edge!(V, pA, ld, pP, ldp, pB, ld, K, i0, rfull + 1, ie, c0, c1)
        _lhl_micro_edge!(V, pA, ld, pP, ldp, pB, ld, K, i0, ib, rfull, ct, c1)
        ib = ie + 1
    end
    return nothing
end


function _lhl_gemm!(
        bk, A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}, nt::Int
    ) where {T <: _LHL_CFloat}
    (i1 < i0 || j1 < j0 || c1 < c0) && return nothing
    if stride(A, 1) == 1
        _lhl_gemm_cmicro!(bk, A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack, nt)
    else
        invoke(
            _lhl_gemm!, Tuple{Any, AbstractMatrix{T}, Int, Int, Int, Int, Int, Int, Int, Int, T, Any, Int},
            bk, A, i0, i1, j0, j1, r0, cB, c0, c1, sgn, pack, nt
        )
    end
    return nothing
end

# The complex product through the real microkernel: column j of the packed multiplier
# panel becomes two real columns — the interleaved values and their swap-negation — whose
# matching two B rows are the re/im lanes of B's complex row j, which the interleaved
# storage already lays out contiguously.  So C_rv += P_packed · B_rv runs the unmodified
# real tiles with 2·rows rows and 2K inner length, B read from A in place: the same
# 8·rows·cols·K real flops as four real GEMMs, with no shuffles and one pass over C.
function _lhl_gemm_cmicro!(
        bk, A::StridedMatrix{T}, i0::Int, i1::Int, j0::Int, j1::Int, r0::Int,
        cB::Int, c0::Int, c1::Int, sgn::T, pack::Array{T}, nt::Int
    ) where {T <: _LHL_CFloat}
    Tr = real(T)
    K = j1 - j0 + 1
    rows = i1 - i0 + 1
    rr = 2 * rows
    ldp = rr % 256 == 0 ? rr + 8 : rr
    2 * length(pack) >= ldp * 2K || throw(ArgumentError("LHL gemm scratch too small"))
    sz = sizeof(Tr)
    ld = 2 * stride(A, 2)
    GC.@preserve A pack begin
        pA = Ptr{Tr}(pointer(A))
        pP = Ptr{Tr}(pointer(pack))
        @inbounds for (jj, j) in enumerate(j0:j1)
            pa = pP + (2jj - 2) * ldp * sz
            pb = pP + (2jj - 1) * ldp * sz
            for t in 1:rows
                v = sgn * A[i0 + t - 1, j]
                unsafe_store!(pa, real(v), 2t - 1)
                unsafe_store!(pa, imag(v), 2t)
                unsafe_store!(pb, -imag(v), 2t - 1)
                unsafe_store!(pb, real(v), 2t)
            end
        end
        pB = pA + ((cB - c0) * ld + 2 * (r0 - 1)) * sz
        ia = 2 * i0 - 1
        ib = 2 * i1
        # Each C element takes its 2K FMAs in a fixed order whatever the partition, so
        # both threaded splits are bit-identical to the serial call.
        if nt > 1 && c1 - c0 + 1 > _LHL_GEMV_GROUP
            P = cld(c1 - c0 + 1, _LHL_GEMV_GROUP)
            _lhl_foreach_chunk!(bk, P) do p
                ca, cb = _lhl_group(c0, c1, p)
                _lhl_gemm_tiles!(pA, ld, pP, ldp, pB, 2K, ia, ib, ca, cb)
            end
        elseif nt > 1 && rr >= 4 * 384
            _lhl_foreach_chunk!(bk, nt) do t
                ra, rb = _lhl_chunk(ia, ib, t, nt, 384)
                ra <= rb && _lhl_gemm_tiles!(pA, ld, pP + (ra - ia) * sz, ldp, pB, 2K, ra, rb, c0, c1)
            end
        else
            _lhl_gemm_tiles!(pA, ld, pP, ldp, pB, 2K, ia, ib, c0, c1)
        end
    end
    return nothing
end

# Rows k0+1:kb+1 of the trailing block T = A[:, kb+2:n] ← M⁻¹ · rows, M the unit lower
# triangle of the panel's multipliers (M[i, k+1] = A[i, k] for i ≥ k+2, i.e. the panel's left
# transforms restricted to its own rows).  Row k0+1 is unchanged.
function _lhl_trsm_block!(bk, A::AbstractMatrix{T}, k0::Int, kb::Int, pack, nt::Int = 1) where {T}
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
# live in `pack`, in the microkernel's packed-P / B layouts.  Threads take the GEMV's
# column groups, each copying and solving its own columns.
function _lhl_trsm_block!(
        bk, A::StridedMatrix{T}, k0::Int, kb::Int, pack::Array{T}, nt::Int = 1
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 2)
    nb = kb - k0 + 1
    ncol = n - kb - 1
    if stride(A, 1) != 1 || nb < 2 || length(pack) < nb * nb + nb * ncol
        return invoke(_lhl_trsm_block!, Tuple{Any, AbstractMatrix{T}, Int, Int, Any, Int}, bk, A, k0, kb, pack, nt)
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
    sz = sizeof(T)
    ld = stride(A, 2)
    GC.@preserve A pack begin
        pA = pointer(A)
        pP = pointer(pack)
        pB = pointer(pack) + (off - (kb + 1) * nb) * sz
        if nt > 1 && ncol > _LHL_GEMV_GROUP
            P = cld(ncol, _LHL_GEMV_GROUP)
            _lhl_foreach_chunk!(bk, P) do p
                ca, cb = _lhl_group(kb + 2, n, p)
                _lhl_trsm_cols!(pA, ld, pP, pB, nb, k0, kb, ca, cb)
            end
        else
            _lhl_trsm_cols!(pA, ld, pP, pB, nb, k0, kb, kb + 2, n)
        end
    end
    return A
end

@inline function _lhl_trsm_cols!(
        pA::Ptr{T}, ld::Int, pP::Ptr{T}, pB::Ptr{T}, nb::Int, k0::Int, kb::Int, ca::Int, cb::Int
    ) where {T}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    for c in ca:cb
        pc = pA + (c - 1) * ld * sz
        po = pB + ((c - 1) * nb - k0) * sz
        for i in (k0 + 1):(kb + 1)
            unsafe_store!(po, unsafe_load(pc, i), i)
        end
    end
    ib = k0 + 1
    ie = kb + 1
    rfull = ib + (nb ÷ (3W)) * 3W - 1
    ct = _lhl_micro_tile!(V, pA, ld, pP, nb, pB, nb, nb, ib, ib, rfull, ca, cb)
    _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, ib, rfull, ct, cb)
    r = rfull + 1
    while r + W - 1 <= ie
        ct = _lhl_micro_tile1!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, ca, cb)
        _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, r + W - 1, ct, cb)
        r += W
    end
    _lhl_micro_edge!(V, pA, ld, pP, nb, pB, nb, nb, ib, r, ie, ca, cb)
    return nothing
end

# Ht[j, i] = A[i, j] for i ≤ j + 1: row j of Ht is column j of A, transposed a line of
# columns at a time; threads take column chunks (Float32/Float64 on `Matrix` storage).
function _lhl_ht_fill!(bk, Ht::Matrix{T}, A::AbstractMatrix{T}, n::Int, nt::Int) where {T}
    g = 8
    if nt > 1 && n >= 4g * nt && T <: Union{Float32, Float64, ComplexF32, ComplexF64} && A isa Matrix{T}
        GC.@preserve Ht A begin
            pH = pointer(Ht)
            pA = pointer(A)
            ldh = size(Ht, 1)
            lda = size(A, 1)
            _lhl_foreach_chunk!(bk, nt) do t
                ja, jb = _lhl_chunk(1, n, t, nt, g)
                ja <= jb && _lhl_ht_fill_cols!(pH, ldh, pA, lda, n, ja, jb)
            end
        end
    else
        _lhl_ht_fill_cols!(Ht, A, n, 1, n)
    end
    return Ht
end

@inline function _lhl_ht_fill_cols!(Ht, A, n::Int, ja::Int, jb::Int)
    @inbounds for j0 in ja:8:jb
        j1 = min(j0 + 7, jb)
        for i in 1:min(j0 + 1, n)
            for j in j0:j1
                Ht[j, i] = A[i, j]
            end
        end
        for i in (j0 + 2):min(j1 + 1, n)
            for j in (i - 1):j1
                Ht[j, i] = A[i, j]
            end
        end
    end
    return nothing
end

@inline function _lhl_ht_fill_cols!(pH::Ptr{T}, ldh::Int, pA::Ptr{T}, lda::Int, n::Int, ja::Int, jb::Int) where {T}
    sz = sizeof(T)
    for j0 in ja:8:jb
        j1 = min(j0 + 7, jb)
        for i in 1:min(j0 + 1, n)
            ph = pH + (i - 1) * ldh * sz
            pa = pA + (i - 1) * sz
            for j in j0:j1
                unsafe_store!(ph, unsafe_load(pa + (j - 1) * lda * sz), j)
            end
        end
        for i in (j0 + 2):min(j1 + 1, n)
            ph = pH + (i - 1) * ldh * sz
            pa = pA + (i - 1) * sz
            for j in (i - 1):j1
                unsafe_store!(ph, unsafe_load(pa + (j - 1) * lda * sz), j)
            end
        end
    end
    return nothing
end

"""
    applyZ!(x, ws)

`x ← Z x`.  `n²/2` multiply–adds.  Uses `ws.xbuf` (for a complex workspace, the shift's
planar buffer) as scratch, so not thread-safe on a shared workspace.
"""
function applyZ!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    if T <: _LHL_CFloat && eltype(x) === T
        sh = getfield(ws, :shift)
        y = sh.xbuf
        o = sh.po
        @inbounds for i in 1:n
            v = x[i]
            y[i] = real(v)
            y[o + i] = imag(v)
        end
        _lhl_zero_planes!(y, n, o)
        _lhl_zsweep_bufc!(y, 0, o, ws.Lpp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            p = ip[i]
            x[i] = Complex(y[p], y[o + p]) * d[i]
        end
    elseif eltype(x) === T
        y = ws.xbuf
        @inbounds for i in 1:n
            y[i] = x[i]
        end
        _lhl_zero_pad!(y, n)
        _lhl_zsweep_buf!(y, 0, ws.Lp, n)
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

`x ← Z⁻¹x`.  `n²/2` multiply–adds.  Uses `ws.xbuf` (for a complex workspace, the shift's
planar buffer) as scratch, so not thread-safe on a shared workspace.
"""
function applyZinv!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    if T <: _LHL_CFloat && eltype(x) === T
        sh = getfield(ws, :shift)
        y = sh.xbuf
        o = sh.po
        perm = ws.perm
        isc = ws.iscale
        @inbounds for i in 1:n
            p = perm[i]
            v = x[p] * isc[p]
            y[i] = real(v)
            y[o + i] = imag(v)
        end
        _lhl_zero_planes!(y, n, o)
        _lhl_zinvsweep_bufc!(y, 0, o, ws.Lpp, n)
        @inbounds for i in 1:n
            x[i] = Complex(y[i], y[o + i])
        end
    elseif eltype(x) === T
        y = ws.xbuf
        _lhl_gather!(y, x, ws)
        _lhl_zinvsweep_buf!(y, 0, ws.Lp, n)
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
# is a full vector, so there is no remainder loop and no branch on the row count.  `o` is
# the offset of the vector inside `y` (a plane of a complex buffer); the two-plane forms
# sweep the planes at `oa` and `ob` together, reading each multiplier once.
_lhl_zinvsweep_buf!(y::AbstractVector, o::Int, Lp::AbstractVector, n::Int) =
    _lhl_zinvsweep!(view(y, (o + 1):length(y)), Lp, n)
_lhl_zsweep_buf!(y::AbstractVector, o::Int, Lp::AbstractVector, n::Int) =
    _lhl_zsweep!(view(y, (o + 1):length(y)), Lp, n)
function _lhl_zinvsweep_buf!(y::AbstractVector, oa::Int, ob::Int, Lp::AbstractVector, n::Int)
    _lhl_zinvsweep_buf!(y, oa, Lp, n)
    return _lhl_zinvsweep_buf!(y, ob, Lp, n)
end
function _lhl_zsweep_buf!(y::AbstractVector, oa::Int, ob::Int, Lp::AbstractVector, n::Int)
    _lhl_zsweep_buf!(y, oa, Lp, n)
    return _lhl_zsweep_buf!(y, ob, Lp, n)
end

# The head of a group of four steps (its 3×3 triangle, `h` pointing at the eight head
# slots): the serial recurrence on rows k+1:k+4 of the plane at `o`, returning the four
# values the body broadcasts.
@inline function _lhl_zinv_head!(y::Vector{T}, o::Int, k::Int, h::Ptr{T}) where {T}
    @inbounds begin
        x1 = y[o + k + 1]
        x2 = muladd(-unsafe_load(h), x1, y[o + k + 2])
        y[o + k + 2] = x2
        x3 = muladd(-unsafe_load(h, 3), x2, muladd(-unsafe_load(h, 2), x1, y[o + k + 3]))
        y[o + k + 3] = x3
        x4 = muladd(-unsafe_load(h, 6), x3, muladd(-unsafe_load(h, 5), x2, muladd(-unsafe_load(h, 4), x1, y[o + k + 4])))
        y[o + k + 4] = x4
    end
    return x1, x2, x3, x4
end
@inline function _lhl_z_head!(y::Vector{T}, o::Int, k::Int, h::Ptr{T}) where {T}
    @inbounds begin
        x1 = y[o + k + 1]
        x2 = y[o + k + 2]
        x3 = y[o + k + 3]
        x4 = y[o + k + 4]
        y[o + k + 2] = muladd(unsafe_load(h), x1, x2)
        y[o + k + 3] = muladd(unsafe_load(h, 3), x2, muladd(unsafe_load(h, 2), x1, x3))
        y[o + k + 4] = muladd(unsafe_load(h, 6), x3, muladd(unsafe_load(h, 5), x2, muladd(unsafe_load(h, 4), x1, x4)))
    end
    return x1, x2, x3, x4
end
@inline _lhl_bcast4(::Type{V}, x) where {V} =
    (_lhl_bcast(V, x[1]), _lhl_bcast(V, x[2]), _lhl_bcast(V, x[3]), _lhl_bcast(V, x[4]))
@inline _lhl_neg4(x) = (-x[1], -x[2], -x[3], -x[4])

# The body of a group: rows k+5:n of the four columns against the broadcasts `b`, one plane
# (`q`) or two (`q` and `q + od`).  `p` walks the multipliers, `pa` per vector, `cs` between
# the four columns' rows.
@inline function _lhl_zbody!(::Type{V}, p::Ptr{T}, pend::Ptr{T}, pa::Int, cs::Int, q::Ptr{T}, b) where {W, T, V <: NTuple{W, VecElement{T}}}
    while p < pend
        v = _lhl_vload(V, q)
        v = _lhl_fma(_lhl_vload(V, p), b[1], v)
        v = _lhl_fma(_lhl_vload(V, p + cs), b[2], v)
        v = _lhl_fma(_lhl_vload(V, p + 2cs), b[3], v)
        v = _lhl_fma(_lhl_vload(V, p + 3cs), b[4], v)
        _lhl_vstore!(q, v)
        p += pa
        q += W * sizeof(T)
    end
    return nothing
end
@inline function _lhl_zbody!(::Type{V}, p::Ptr{T}, pend::Ptr{T}, pa::Int, cs::Int, q::Ptr{T}, od::Int, b, c) where {W, T, V <: NTuple{W, VecElement{T}}}
    while p < pend
        v = _lhl_vload(V, q)
        w = _lhl_vload(V, q + od)
        l = _lhl_vload(V, p)
        v = _lhl_fma(l, b[1], v)
        w = _lhl_fma(l, c[1], w)
        l = _lhl_vload(V, p + cs)
        v = _lhl_fma(l, b[2], v)
        w = _lhl_fma(l, c[2], w)
        l = _lhl_vload(V, p + 2cs)
        v = _lhl_fma(l, b[3], v)
        w = _lhl_fma(l, c[3], w)
        l = _lhl_vload(V, p + 3cs)
        v = _lhl_fma(l, b[4], v)
        w = _lhl_fma(l, c[4], w)
        _lhl_vstore!(q, v)
        _lhl_vstore!(q + od, w)
        p += pa
        q += W * sizeof(T)
    end
    return nothing
end
# A single step (rows k+2:n of one column) on one or two planes.
@inline function _lhl_zstep!(::Type{V}, p::Ptr{T}, pend::Ptr{T}, q::Ptr{T}, b::V) where {W, T, V <: NTuple{W, VecElement{T}}}
    while p < pend
        _lhl_vstore!(q, _lhl_fma(_lhl_vload(V, p), b, _lhl_vload(V, q)))
        p += W * sizeof(T)
        q += W * sizeof(T)
    end
    return nothing
end
@inline function _lhl_zstep!(::Type{V}, p::Ptr{T}, pend::Ptr{T}, q::Ptr{T}, od::Int, b::V, c::V) where {W, T, V <: NTuple{W, VecElement{T}}}
    while p < pend
        l = _lhl_vload(V, p)
        _lhl_vstore!(q, _lhl_fma(l, b, _lhl_vload(V, q)))
        _lhl_vstore!(q + od, _lhl_fma(l, c, _lhl_vload(V, q + od)))
        p += W * sizeof(T)
        q += W * sizeof(T)
    end
    return nothing
end

for P in (1, 2)
    # plane offsets and the per-plane head/broadcast statements
    os = P == 1 ? (:o,) : (:oa, :ob)
    args = [:($(o)::Int) for o in os]
    heads(f) = Expr(:block, [:($(Symbol(:x, i)) = $f(y, $(os[i]), k, h)) for i in 1:P]...)
    bcs(neg) = Expr(:block, [:($(Symbol(:b, i)) = _lhl_bcast4(V, $(neg ? :(_lhl_neg4($(Symbol(:x, i)))) : Symbol(:x, i)))) for i in 1:P]...)
    bc1(neg) = Expr(:block, [:($(Symbol(:b, i)) = _lhl_bcast(V, $(neg ? :(-y[$(os[i]) + k + 1]) : :(y[$(os[i]) + k + 1])))) for i in 1:P]...)
    body = P == 1 ? :(_lhl_zbody!(V, p, pend, pa, cs, q, b1)) : :(_lhl_zbody!(V, p, pend, pa, cs, q, od, b1, b2))
    step = P == 1 ? :(_lhl_zstep!(V, h, pend, q, b1)) : :(_lhl_zstep!(V, h, pend, q, od, b1, b2))
    od = P == 1 ? :(nothing) : :(od = (ob - oa) * sz)
    o1 = os[1]
    @eval begin
        function _lhl_zinvsweep_buf!(y::Vector{T}, $(args...), Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
            V = _lhl_vectype(T)
            W = _LHL_VEC_BYTES ÷ sizeof(T)
            sz = sizeof(T)
            tiled = _lhl_tiled(n, T)
            G = max(n - 2, 0) >> 2
            $od
            GC.@preserve y Lp begin
                py = pointer(y) + $o1 * sz
                h = pointer(Lp)
                @inbounds for g in 1:G
                    k = 4g - 3
                    $(heads(:_lhl_zinv_head!))
                    $(bcs(true))
                    p = h + _LHL_HEAD * sz
                    q = py + (k + 4) * sz
                    mpb = cld(n - k - 4, W) * W * sz
                    h = p + 4mpb
                    # byte distance between the four columns' rows, and the advance per vector
                    cs = tiled ? W * sz : mpb
                    pa = tiled ? 4W * sz : W * sz
                    pend = tiled ? h : p + mpb
                    $body
                end
                @inbounds for k in (4G + 1):(n - 2)
                    $(bc1(true))
                    q = py + (k + 1) * sz
                    pend = h + cld(n - k - 1, W) * W * sz
                    $step
                    h = pend
                end
            end
            return y
        end

        function _lhl_zsweep_buf!(y::Vector{T}, $(args...), Lp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
            V = _lhl_vectype(T)
            W = _LHL_VEC_BYTES ÷ sizeof(T)
            sz = sizeof(T)
            tiled = _lhl_tiled(n, T)
            G = max(n - 2, 0) >> 2
            $od
            GC.@preserve y Lp begin
                py = pointer(y) + $o1 * sz
                h = pointer(Lp) + length(Lp) * sz
                @inbounds for k in (n - 2):-1:(4G + 1)
                    $(bc1(false))
                    m = cld(n - k - 1, W) * W
                    h -= m * sz
                    q = py + (k + 1) * sz
                    pend = h + m * sz
                    $step
                end
                @inbounds for g in G:-1:1
                    k = 4g - 3
                    h -= _lhl_group_size(n, k, W) * sz
                    $(heads(:_lhl_z_head!))
                    $(bcs(false))
                    p = h + _LHL_HEAD * sz
                    q = py + (k + 4) * sz
                    mpb = cld(n - k - 4, W) * W * sz
                    cs = tiled ? W * sz : mpb
                    pa = tiled ? 4W * sz : W * sz
                    pend = tiled ? p + 4mpb : p + mpb
                    $body
                end
            end
            return y
        end
    end
end

# ---------------------------------------------------------------------------
# The same sweeps for a *complex* workspace: `y` holds the vector on two planes (offsets
# `oa`, `ob`) and the multipliers come planar from `Lpp` (im plane `Lh` reals behind the
# re plane, see `_lhl_lpack2!`), so every complex multiply–add is four real FMAs on plane
# vectors with no lane shuffling.
# ---------------------------------------------------------------------------

# Group heads: the 3×3 triangle as scalar complex arithmetic on the planes; `h` points at
# the re head slots, the im slots sit `Lo` bytes behind.
@inline function _lhl_zinv_headc!(y::Vector{T}, oa::Int, ob::Int, k::Int, h::Ptr{T}, Lo::Int) where {T}
    C = Complex{T}
    @inbounds begin
        x1 = C(y[oa + k + 1], y[ob + k + 1])
        x2 = C(y[oa + k + 2], y[ob + k + 2]) - C(unsafe_load(h), unsafe_load(h + Lo)) * x1
        y[oa + k + 2] = real(x2)
        y[ob + k + 2] = imag(x2)
        x3 = C(y[oa + k + 3], y[ob + k + 3]) - C(unsafe_load(h, 2), unsafe_load(h + Lo, 2)) * x1 -
            C(unsafe_load(h, 3), unsafe_load(h + Lo, 3)) * x2
        y[oa + k + 3] = real(x3)
        y[ob + k + 3] = imag(x3)
        x4 = C(y[oa + k + 4], y[ob + k + 4]) -
            (C(unsafe_load(h, 4), unsafe_load(h + Lo, 4)) * x1 + C(unsafe_load(h, 5), unsafe_load(h + Lo, 5)) * x2) -
            C(unsafe_load(h, 6), unsafe_load(h + Lo, 6)) * x3
        y[oa + k + 4] = real(x4)
        y[ob + k + 4] = imag(x4)
    end
    return x1, x2, x3, x4
end

@inline function _lhl_z_headc!(y::Vector{T}, oa::Int, ob::Int, k::Int, h::Ptr{T}, Lo::Int) where {T}
    C = Complex{T}
    @inbounds begin
        x1 = C(y[oa + k + 1], y[ob + k + 1])
        x2 = C(y[oa + k + 2], y[ob + k + 2])
        x3 = C(y[oa + k + 3], y[ob + k + 3])
        x4 = C(y[oa + k + 4], y[ob + k + 4])
        v = x2 + C(unsafe_load(h), unsafe_load(h + Lo)) * x1
        y[oa + k + 2] = real(v)
        y[ob + k + 2] = imag(v)
        v = x3 + C(unsafe_load(h, 2), unsafe_load(h + Lo, 2)) * x1 +
            C(unsafe_load(h, 3), unsafe_load(h + Lo, 3)) * x2
        y[oa + k + 3] = real(v)
        y[ob + k + 3] = imag(v)
        v = x4 + (C(unsafe_load(h, 4), unsafe_load(h + Lo, 4)) * x1 + C(unsafe_load(h, 5), unsafe_load(h + Lo, 5)) * x2) +
            C(unsafe_load(h, 6), unsafe_load(h + Lo, 6)) * x3
        y[oa + k + 4] = real(v)
        y[ob + k + 4] = imag(v)
    end
    return x1, x2, x3, x4
end

# The body of a group: rows k+5:n of the four columns, planar multipliers against the
# complex coefficients `c` (already sign-folded), both `y` planes updated per vector.
@inline function _lhl_zbodyc!(
        ::Type{V}, p::Ptr{T}, Lo::Int, pend::Ptr{T}, pa::Int, cs::Int, q::Ptr{T}, od::Int,
        c::NTuple{4, Complex{T}}
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    br1 = _lhl_bcast(V, real(c[1])); bi1 = _lhl_bcast(V, imag(c[1])); bn1 = _lhl_bcast(V, -imag(c[1]))
    br2 = _lhl_bcast(V, real(c[2])); bi2 = _lhl_bcast(V, imag(c[2])); bn2 = _lhl_bcast(V, -imag(c[2]))
    br3 = _lhl_bcast(V, real(c[3])); bi3 = _lhl_bcast(V, imag(c[3])); bn3 = _lhl_bcast(V, -imag(c[3]))
    br4 = _lhl_bcast(V, real(c[4])); bi4 = _lhl_bcast(V, imag(c[4])); bn4 = _lhl_bcast(V, -imag(c[4]))
    while p < pend
        v = _lhl_vload(V, q)
        w = _lhl_vload(V, q + od)
        lr = _lhl_vload(V, p)
        li = _lhl_vload(V, p + Lo)
        v = _lhl_fma(li, bn1, _lhl_fma(lr, br1, v))
        w = _lhl_fma(li, br1, _lhl_fma(lr, bi1, w))
        lr = _lhl_vload(V, p + cs)
        li = _lhl_vload(V, p + cs + Lo)
        v = _lhl_fma(li, bn2, _lhl_fma(lr, br2, v))
        w = _lhl_fma(li, br2, _lhl_fma(lr, bi2, w))
        lr = _lhl_vload(V, p + 2cs)
        li = _lhl_vload(V, p + 2cs + Lo)
        v = _lhl_fma(li, bn3, _lhl_fma(lr, br3, v))
        w = _lhl_fma(li, br3, _lhl_fma(lr, bi3, w))
        lr = _lhl_vload(V, p + 3cs)
        li = _lhl_vload(V, p + 3cs + Lo)
        v = _lhl_fma(li, bn4, _lhl_fma(lr, br4, v))
        w = _lhl_fma(li, br4, _lhl_fma(lr, bi4, w))
        _lhl_vstore!(q, v)
        _lhl_vstore!(q + od, w)
        p += pa
        q += W * sizeof(T)
    end
    return nothing
end

# A single step (rows k+2:n of one column) on the planes.
@inline function _lhl_zstepc!(
        ::Type{V}, p::Ptr{T}, Lo::Int, pend::Ptr{T}, q::Ptr{T}, od::Int, c::Complex{T}
    ) where {W, T, V <: NTuple{W, VecElement{T}}}
    br = _lhl_bcast(V, real(c))
    bi = _lhl_bcast(V, imag(c))
    bn = _lhl_bcast(V, -imag(c))
    while p < pend
        v = _lhl_vload(V, q)
        w = _lhl_vload(V, q + od)
        lr = _lhl_vload(V, p)
        li = _lhl_vload(V, p + Lo)
        v = _lhl_fma(li, bn, _lhl_fma(lr, br, v))
        w = _lhl_fma(li, br, _lhl_fma(lr, bi, w))
        _lhl_vstore!(q, v)
        _lhl_vstore!(q + od, w)
        p += W * sizeof(T)
        q += W * sizeof(T)
    end
    return nothing
end

function _lhl_zinvsweep_bufc!(y::Vector{T}, oa::Int, ob::Int, Lpp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    Lo = (length(Lpp) >> 1) * sz
    od = (ob - oa) * sz
    GC.@preserve y Lpp begin
        py = pointer(y) + oa * sz
        h = pointer(Lpp)
        @inbounds for g in 1:G
            k = 4g - 3
            x = _lhl_zinv_headc!(y, oa, ob, k, h, Lo)
            c = (-x[1], -x[2], -x[3], -x[4])
            p = h + _LHL_HEAD * sz
            q = py + (k + 4) * sz
            mpb = cld(n - k - 4, W) * W * sz
            h = p + 4mpb
            cs = tiled ? W * sz : mpb
            pa = tiled ? 4W * sz : W * sz
            pend = tiled ? h : p + mpb
            _lhl_zbodyc!(V, p, Lo, pend, pa, cs, q, od, c)
        end
        @inbounds for k in (4G + 1):(n - 2)
            ck = -Complex(y[oa + k + 1], y[ob + k + 1])
            q = py + (k + 1) * sz
            pend = h + cld(n - k - 1, W) * W * sz
            _lhl_zstepc!(V, h, Lo, pend, q, od, ck)
            h = pend
        end
    end
    return y
end

function _lhl_zsweep_bufc!(y::Vector{T}, oa::Int, ob::Int, Lpp::Vector{T}, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    tiled = _lhl_tiled(n, T)
    G = max(n - 2, 0) >> 2
    Lo = (length(Lpp) >> 1) * sz
    od = (ob - oa) * sz
    GC.@preserve y Lpp begin
        py = pointer(y) + oa * sz
        h = pointer(Lpp) + (length(Lpp) >> 1) * sz
        @inbounds for k in (n - 2):-1:(4G + 1)
            ck = Complex(y[oa + k + 1], y[ob + k + 1])
            m = cld(n - k - 1, W) * W
            h -= m * sz
            q = py + (k + 1) * sz
            _lhl_zstepc!(V, h, Lo, h + m * sz, q, od, ck)
        end
        @inbounds for g in G:-1:1
            k = 4g - 3
            h -= _lhl_group_size(n, k, W) * sz
            c = _lhl_z_headc!(y, oa, ob, k, h, Lo)
            p = h + _LHL_HEAD * sz
            q = py + (k + 4) * sz
            mpb = cld(n - k - 4, W) * W * sz
            cs = tiled ? W * sz : mpb
            pa = tiled ? 4W * sz : W * sz
            pend = tiled ? p + 4mpb : p + mpb
            _lhl_zbodyc!(V, p, Lo, pend, pa, cs, q, od, c)
        end
    end
    return y
end

# ---------------------------------------------------------------------------
# The γ-dependent half
# ---------------------------------------------------------------------------

"""
    lhl_shift!(ws, σ, τ) -> ws
    lhl_shift!(sh::LHLShift, ws, σ, τ) -> sh

Form `G = σI + τH` and LU-factorize it (partial pivoting) into `ws.shift` — or into the
separately held `sh`, resized to `ws` if needed — transposed, in `Gt`.  `≈n²`
multiply–adds.  `(σ, τ) = (1, -γ)` gives `I - γJ`; `(0, 1)` gives `J` itself.  The shift is
converted to the `LHLShift`'s element type: a complex `σ` or `τ` needs one built for
complex shifts (`LHLWorkspace{T}(n; shift = Complex{T})`, `lhl(J; shift = Complex{T})` or
`LHLShift{Complex{T}}(ws)`), and throws an `ArgumentError` otherwise.  A zero pivot is
reported in `info` (the index of the first), not thrown.

An upper Hessenberg matrix has one subdiagonal, so each elimination step chooses between
two candidates and the element growth factor is bounded by `n` — far tighter than the
`2ⁿ⁻¹` of general partial pivoting.
"""
function lhl_shift!(ws::LHLWorkspace, σ, τ)
    lhl_shift!(ws.shift, ws, σ, τ)
    return ws
end

function lhl_shift!(sh::LHLShift{TG}, ws::LHLWorkspace{T}, σ, τ) where {TG, T}
    n = ws.n
    (TG === T || TG === Complex{real(T)}) || throw(
        ArgumentError("an LHLShift{$TG} cannot serve a workspace of element type $T")
    )
    if TG <: Real && !(isreal(σ) && isreal(τ))
        throw(
            ArgumentError(
                "complex shift ($σ, $τ) on an LHLShift{$TG}; build the workspace with " *
                    "`shift = Complex{$T}` or use an `LHLShift{Complex{$T}}`"
            )
        )
    end
    σ = convert(TG, σ)
    τ = convert(TG, τ)
    _lhl_resize!(sh, n)
    sh.σ = σ
    sh.τ = τ
    n == 0 && return sh
    if real(T) <: Union{Float32, Float64} && n >= _lhl_shift_fused_min(TG)
        # Four rows per pass; `Gt`'s padded leading dimension keeps its columns out of the
        # L1 set the four `Ht` columns share when the stride is a multiple of 4 KiB.
        info = _lhl_shift_fused!(Val(4), sh, ws, σ, τ)
    else
        info = _lhl_shift_rows!(sh, ws, σ, τ)
    end
    sh.info = info
    return sh
end

# Below these sizes the one-step passes win: their inner loops are shorter and the fused
# passes' triangles cost more per step; a complex shift, whose passes store twice as much,
# gains from fusing earlier.
_lhl_shift_fused_min(::Type{TG}) where {TG} = TG <: Complex ? 128 : 512

# Element access on the shift's storage: for a complex `TG` the value at (row j[, column c])
# is the pair at rows j and o+j of the two planes.
@inline _lhl_get(::Type{TG}, A::AbstractVector, o::Int, j::Int) where {TG} = @inbounds A[j]
@inline _lhl_get(::Type{Complex{Tr}}, A::AbstractVector, o::Int, j::Int) where {Tr} =
    @inbounds Complex(A[j], A[o + j])
@inline _lhl_get(::Type{TG}, A::AbstractMatrix, o::Int, j::Int, c::Int) where {TG} = @inbounds A[j, c]
@inline _lhl_get(::Type{Complex{Tr}}, A::AbstractMatrix, o::Int, j::Int, c::Int) where {Tr} =
    @inbounds Complex(A[j, c], A[o + j, c])
@inline _lhl_set!(::Type{TG}, A::AbstractVector, o::Int, j::Int, v) where {TG} = @inbounds A[j] = v
@inline function _lhl_set!(::Type{Complex{Tr}}, A::AbstractVector, o::Int, j::Int, v) where {Tr}
    @inbounds A[j] = real(v)
    @inbounds A[o + j] = imag(v)
    return v
end
@inline _lhl_set!(::Type{TG}, A::AbstractMatrix, o::Int, j::Int, c::Int, v) where {TG} = @inbounds A[j, c] = v
@inline function _lhl_set!(::Type{Complex{Tr}}, A::AbstractMatrix, o::Int, j::Int, c::Int, v) where {Tr}
    @inbounds A[j, c] = real(v)
    @inbounds A[o + j, c] = imag(v)
    return v
end

# Pivot magnitude (reduction and shift) and balance norm: |re| + |im| for complex values,
# as LAPACK's izamax/CABS1 — no square root and no overflow, and within √2 of the modulus.
@inline _lhl_pivmag(x::Real) = abs(x)
@inline _lhl_pivmag(x::Complex) = abs(real(x)) + abs(imag(x))

# Row k of G is formed from Ht only when it enters the elimination, and the row that has
# not yet been chosen as a pivot row lives in `r`: at step k the candidates are the
# pending row (`r`, currently row k) and the fresh row k+1, whichever wins is written to
# Gt[:, k] as row k of U, and the loser minus its multiple becomes the new pending row.
# No row of G is ever copied twice and no interchange is ever performed on storage.
# `TG` real or complex: with a complex shift every product is written out on the planes,
# in the same term order as `Complex` arithmetic, so the inner loops stay real ones.
@inline function _lhl_shift_rows!(sh::LHLShift{TG}, ws::LHLWorkspace, σ::TG, τ::TG) where {TG}
    Ht = ws.Ht
    Gt = sh.Gt
    swap = sh.swap
    r = sh.work
    n = ws.n
    o = sh.po
    @inbounds begin
        @simd for j in 1:n
            _lhl_set!(TG, r, o, j, τ * Ht[j, 1])
        end
        _lhl_set!(TG, r, o, 1, _lhl_get(TG, r, o, 1) + σ)
        info = 0
        for k in 1:(n - 1)
            a = _lhl_get(TG, r, o, k)
            b = τ * Ht[k, k + 1]
            if _lhl_pivmag(b) > _lhl_pivmag(a)
                swap[k] = true
                _lhl_set!(TG, Gt, o, k, k, b)
                l = a / b
                _lhl_set!(TG, Gt, o, k, k + 1, l)
                @simd for j in (k + 1):n
                    g = τ * Ht[j, k + 1]
                    _lhl_set!(TG, Gt, o, j, k, g)
                    _lhl_set!(TG, r, o, j, _lhl_get(TG, r, o, j) - l * g)
                end
                _lhl_set!(TG, Gt, o, k + 1, k, _lhl_get(TG, Gt, o, k + 1, k) + σ)
                _lhl_set!(TG, r, o, k + 1, _lhl_get(TG, r, o, k + 1) - l * σ)
            else
                swap[k] = false
                _lhl_set!(TG, Gt, o, k, k, a)
                if iszero(a)
                    info == 0 && (info = k)
                    l = zero(TG)
                else
                    l = b / a
                end
                _lhl_set!(TG, Gt, o, k, k + 1, l)
                @simd for j in (k + 1):n
                    rj = _lhl_get(TG, r, o, j)
                    _lhl_set!(TG, Gt, o, j, k, rj)
                    _lhl_set!(TG, r, o, j, τ * Ht[j, k + 1] - l * rj)
                end
                _lhl_set!(TG, r, o, k + 1, _lhl_get(TG, r, o, k + 1) + σ)
            end
        end
        return _lhl_shift_finish!(sh, info)
    end
end

@inline function _lhl_shift_finish!(sh::LHLShift{TG}, info::Int) where {TG}
    Gt = sh.Gt
    n = sh.n
    o = sh.po
    @inbounds begin
        rn = _lhl_get(TG, sh.work, o, n)
        _lhl_set!(TG, Gt, o, n, n, rn)
        sh.swap[n] = false
        iszero(rn) && info == 0 && (info = n)
        rd = sh.rdiag
        @simd for j in 1:n
            rd[j] = inv(_lhl_get(TG, Gt, o, j, j))
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
    s = _lhl_pivmag(b) > _lhl_pivmag(a)
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

@inline function _lhl_shift_step(s::Bool, l::T, τ::T, h, rj::T) where {T}
    g = τ * h
    return ifelse(s, g, rj), ifelse(s, rj - l * g, g - l * rj)
end

# Steps k..k+R-1 with decisions S on elements j0:n; S is a compile-time tuple so that each
# pivot pattern gets its own branch-free loop.  All loads precede all stores in the body:
# a store to Gt[j, k] followed by a load of Ht[j, k+2] a 4 KiB multiple away stalls otherwise.
@generated function _lhl_shift_pass!(
        ::Val{S}, L::NTuple{R, T}, Ht, Gt, r, o::Int, n::Int, τ::T, k::Int, j0::Int
    ) where {S, R, T}
    loads = Expr(:block)
    steps = Expr(:block)
    stores = Expr(:block)
    for i in 1:R
        h = Symbol(:h_, i)
        u = Symbol(:u_, i)
        push!(loads.args, :($h = Ht[j, k + $i]))
        push!(steps.args, :(($u, rj) = _lhl_shift_step($(S[i]), L[$i], τ, $h, rj)))
        push!(stores.args, :(_lhl_set!($T, Gt, o, j, k + $(i - 1), $u)))
    end
    return quote
        @inbounds @simd ivdep for j in j0:n
            rj = _lhl_get($T, r, o, j)
            $loads
            $steps
            $stores
            _lhl_set!($T, r, o, j, rj)
        end
        return nothing
    end
end

@generated function _lhl_shift_pass!(S::NTuple{R, Bool}, L, Ht, Gt, r, o, n, τ, k, j0) where {R}
    ex = :(_lhl_shift_pass!(Val($(ntuple(_ -> false, R))), L, Ht, Gt, r, o, n, τ, k, j0))
    for idx in 1:(2^R - 1)
        pat = ntuple(i -> (idx >> (i - 1)) & 1 == 1, R)
        ex = :(idx == $idx ? _lhl_shift_pass!(Val($pat), L, Ht, Gt, r, o, n, τ, k, j0) : $ex)
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

# `Base.setindex(::Tuple, v, i)` is not public API.  The result must keep its `NTuple{R}`
# type so the `@generated` pass above still specializes on it.
@inline _lhl_setindex(t::Tuple{Vararg{Any, N}}, v, i::Int) where {N} =
    ntuple(j -> ifelse(j == i, v, t[j]), Val(N))

@inline function _lhl_shift_block!(
        ::Val{R}, Ht, Gt, swap, r, o::Int, n::Int, σ::T, τ::T, k::Int, info::Int
    ) where {R, T}
    @inbounds begin
        b = τ * Ht[k, k + 1]
        rk = _lhl_get(T, r, o, k)
        s, l, info = _lhl_shift_decide(rk, b, info, k)
        swap[k] = s
        _lhl_set!(T, Gt, o, k, k, ifelse(s, b, rk))
        _lhl_set!(T, Gt, o, k, k + 1, l)
        S = _lhl_fill(Val(R), s)
        L = _lhl_fill(Val(R), l)
        for m in 1:R
            j = k + m
            rj = _lhl_get(T, r, o, j)
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
                _lhl_set!(T, Gt, o, j, k + i - 1, u)
            end
            _lhl_set!(T, r, o, j, rj)
            if m < R
                b = τ * Ht[j, j + 1]
                s, l, info = _lhl_shift_decide(rj, b, info, j)
                swap[j] = s
                _lhl_set!(T, Gt, o, j, j, ifelse(s, b, rj))
                _lhl_set!(T, Gt, o, j, j + 1, l)
                S = _lhl_setindex(S, s, m + 1)
                L = _lhl_setindex(L, l, m + 1)
            end
        end
        _lhl_shift_pass!(S, L, Ht, Gt, r, o, n, τ, k, k + R + 1)
    end
    return info
end

function _lhl_shift_fused!(::Val{R}, sh::LHLShift{TG}, ws::LHLWorkspace, σ::TG, τ::TG) where {R, TG}
    Ht = ws.Ht
    Gt = sh.Gt
    swap = sh.swap
    r = sh.work
    n = ws.n
    o = sh.po
    @inbounds begin
        @simd for j in 1:n
            _lhl_set!(TG, r, o, j, τ * Ht[j, 1])
        end
        _lhl_set!(TG, r, o, 1, _lhl_get(TG, r, o, 1) + σ)
        info = 0
        k = 1
        while k + R - 1 <= n - 1
            info = _lhl_shift_block!(Val(R), Ht, Gt, swap, r, o, n, σ, τ, k, info)
            k += R
        end
        while k <= n - 1
            info = _lhl_shift_block!(Val(1), Ht, Gt, swap, r, o, n, σ, τ, k, info)
            k += 1
        end
        return _lhl_shift_finish!(sh, info)
    end
end

function _hessenberg_solve!(x::AbstractVector, sh::LHLShift{T, T}) where {T}
    Gt = sh.Gt
    rd = sh.rdiag
    n = sh.n
    _lhl_hess_forward!(x, Gt, sh.swap, n)
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

@inline _lhl_cget(Gt, o::Int, i::Int, c::Int) = Complex(Gt[i, c], Gt[o + i, c])

# The complex solve on the planar buffer (`y[1:n]` real parts, `y[o+1:o+n]` imaginary
# parts, planar `Gt`): the same pipelined back substitution as `_hessenberg_solve!`, each
# complex product written out on the two planes.
function _hessenberg_solve_planar!(y::Vector{Tr}, sh::LHLShift{Complex{Tr}, Tr}) where {Tr}
    Gt = sh.Gt
    rd = sh.rdiag
    swap = sh.swap
    n = sh.n
    o = sh.po
    C = Complex{Tr}
    @inbounds for k in 1:(n - 1)
        s = swap[k]
        a = C(y[k], y[o + k])
        b = C(y[k + 1], y[o + k + 1])
        xk = ifelse(s, b, a)
        y[k] = real(xk)
        y[o + k] = imag(xk)
        v = ifelse(s, a, b) - C(Gt[k, k + 1], Gt[o + k, k + 1]) * xk
        y[k + 1] = real(v)
        y[o + k + 1] = imag(v)
    end
    j = n
    s1 = s2 = s3 = s4 = zero(C)
    @inbounds while j - 3 >= 1
        xj = (C(y[j], y[o + j]) - s1) * rd[j]
        y[j] = real(xj)
        y[o + j] = imag(xj)
        s2 += C(Gt[j, j - 1], Gt[o + j, j - 1]) * xj
        s3 += C(Gt[j, j - 2], Gt[o + j, j - 2]) * xj
        s4 += C(Gt[j, j - 3], Gt[o + j, j - 3]) * xj
        xj1 = (C(y[j - 1], y[o + j - 1]) - s2) * rd[j - 1]
        y[j - 1] = real(xj1)
        y[o + j - 1] = imag(xj1)
        s3 += C(Gt[j - 1, j - 2], Gt[o + j - 1, j - 2]) * xj1
        s4 += C(Gt[j - 1, j - 3], Gt[o + j - 1, j - 3]) * xj1
        xj2 = (C(y[j - 2], y[o + j - 2]) - s3) * rd[j - 2]
        y[j - 2] = real(xj2)
        y[o + j - 2] = imag(xj2)
        s4 += C(Gt[j - 2, j - 3], Gt[o + j - 2, j - 3]) * xj2
        xj3 = (C(y[j - 3], y[o + j - 3]) - s4) * rd[j - 3]
        y[j - 3] = real(xj3)
        y[o + j - 3] = imag(xj3)
        jn = j - 4
        if jn - 3 >= 1
            s1, s2, s3, s4 = _lhl_pdot4(y, Gt, o, jn, jn, n)
        end
        j = jn
    end
    @inbounds while j >= 1
        sr = si = zero(Tr)
        @simd for i in (j + 1):n
            xr = y[i]
            xi = y[o + i]
            gr = Gt[i, j]
            gi = Gt[o + i, j]
            sr += gr * xr - gi * xi
            si += gr * xi + gi * xr
        end
        xj = (C(y[j], y[o + j]) - C(sr, si)) * rd[j]
        y[j] = real(xj)
        y[o + j] = imag(xj)
        j -= 1
    end
    return y
end

# The four complex dot products Σᵢ Gt[i, jn-c]·y[i], i = j+1:n, c = 0:3, on the planes.
@inline function _lhl_pdot4(y::AbstractVector{Tr}, Gt::AbstractMatrix{Tr}, o::Int, j::Int, jn::Int, n::Int) where {Tr}
    t1r = t1i = t2r = t2i = t3r = t3i = t4r = t4i = zero(Tr)
    @inbounds @simd for i in (j + 1):n
        xr = y[i]
        xi = y[o + i]
        g1r = Gt[i, jn]
        g1i = Gt[o + i, jn]
        g2r = Gt[i, jn - 1]
        g2i = Gt[o + i, jn - 1]
        g3r = Gt[i, jn - 2]
        g3i = Gt[o + i, jn - 2]
        g4r = Gt[i, jn - 3]
        g4i = Gt[o + i, jn - 3]
        t1r += g1r * xr - g1i * xi
        t1i += g1r * xi + g1i * xr
        t2r += g2r * xr - g2i * xi
        t2i += g2r * xi + g2i * xr
        t3r += g3r * xr - g3i * xi
        t3i += g3r * xi + g3i * xr
        t4r += g4r * xr - g4i * xi
        t4i += g4r * xi + g4i * xr
    end
    return Complex(t1r, t1i), Complex(t2r, t2i), Complex(t3r, t3i), Complex(t4r, t4i)
end

# Explicit vectors, running to a full vector past n (the pads of both planes are zero):
# eight accumulators, one real and one imaginary per column.
@inline function _lhl_pdot4(y::Vector{T}, Gt::Matrix{T}, o::Int, j::Int, jn::Int, n::Int) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    ldg = size(Gt, 1)
    GC.@preserve y Gt begin
        c1 = pointer(Gt) + ((jn - 1) * ldg + j) * sz
        c2 = c1 - ldg * sz
        c3 = c2 - ldg * sz
        c4 = c3 - ldg * sz
        q = pointer(y) + j * sz
        ob = o * sz
        mb = cld(n - j, W) * W * sz
        z = _lhl_bcast(V, zero(T))
        a1r = a1i = a2r = a2i = a3r = a3i = a4r = a4i = z
        d = 0
        while d < mb
            xr = _lhl_vload(V, q + d)
            xi = _lhl_vload(V, q + ob + d)
            nxi = _lhl_vneg(xi)
            g = _lhl_vload(V, c1 + d)
            a1r = _lhl_fma(g, xr, a1r)
            a1i = _lhl_fma(g, xi, a1i)
            g = _lhl_vload(V, c1 + ob + d)
            a1r = _lhl_fma(g, nxi, a1r)
            a1i = _lhl_fma(g, xr, a1i)
            g = _lhl_vload(V, c2 + d)
            a2r = _lhl_fma(g, xr, a2r)
            a2i = _lhl_fma(g, xi, a2i)
            g = _lhl_vload(V, c2 + ob + d)
            a2r = _lhl_fma(g, nxi, a2r)
            a2i = _lhl_fma(g, xr, a2i)
            g = _lhl_vload(V, c3 + d)
            a3r = _lhl_fma(g, xr, a3r)
            a3i = _lhl_fma(g, xi, a3i)
            g = _lhl_vload(V, c3 + ob + d)
            a3r = _lhl_fma(g, nxi, a3r)
            a3i = _lhl_fma(g, xr, a3i)
            g = _lhl_vload(V, c4 + d)
            a4r = _lhl_fma(g, xr, a4r)
            a4i = _lhl_fma(g, xi, a4i)
            g = _lhl_vload(V, c4 + ob + d)
            a4r = _lhl_fma(g, nxi, a4r)
            a4i = _lhl_fma(g, xr, a4i)
            d += W * sz
        end
    end
    return Complex(_lhl_vsum(a1r), _lhl_vsum(a1i)), Complex(_lhl_vsum(a2r), _lhl_vsum(a2i)),
        Complex(_lhl_vsum(a3r), _lhl_vsum(a3i)), Complex(_lhl_vsum(a4r), _lhl_vsum(a4i))
end

# On the padded buffer: same pipelining, the four dot products as vector accumulators
# running to a full vector past n (Gt's pad rows and y's pad are zero).
_hessenberg_solve_buf!(y::AbstractVector, sh::LHLShift) = _hessenberg_solve!(y, sh)

function _hessenberg_solve_buf!(y::Vector{T}, sh::LHLShift{T, T}) where {T <: Union{Float32, Float64}}
    V = _lhl_vectype(T)
    W = _LHL_VEC_BYTES ÷ sizeof(T)
    sz = sizeof(T)
    Gt = sh.Gt
    rd = sh.rdiag
    n = sh.n
    ldg = size(Gt, 1)
    _lhl_hess_forward!(y, Gt, sh.swap, n)
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

@inline _lhl_vsum(v::_LHLVec{T, W}) where {T, W} = sum(ntuple(w -> v[w].value, Val(W + 1)))

"""
    lhl_ldiv!(x, ws)
    lhl_ldiv!(x, sh::LHLShift, ws)

`x ← W⁻¹x` for the `W` currently loaded by [`lhl_shift!`](@ref) into `ws.shift` (or into
`sh`): `Z⁻¹`, Hessenberg solve, `Z`.  `3n²/2` multiply–adds.  `x` must be able to hold the
shift's element type (a real `x` cannot take a complex shift).  Uses the shift's `xbuf` as
scratch, so concurrent solves must each have their own.
"""
lhl_ldiv!(x::AbstractVector, ws::LHLWorkspace) = lhl_ldiv!(x, ws.shift, ws)

function lhl_ldiv!(x::AbstractVector, sh::LHLShift{TG}, ws::LHLWorkspace) where {TG}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    sh.n == n || throw(DimensionMismatch("the LHLShift is $(sh.n)×$(sh.n), the workspace $n×$n"))
    if eltype(x) === TG
        y = sh.xbuf
        _lhl_gather!(y, x, ws)
        _lhl_zinvsweep_buf!(y, 0, ws.Lp, n)
        _hessenberg_solve_buf!(y, sh)
        _lhl_zsweep_buf!(y, 0, ws.Lp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            x[i] = y[ip[i]] * d[i]
        end
    else
        applyZinv!(x, ws)
        _hessenberg_solve!(x, sh)
        applyZ!(x, ws)
    end
    return x
end

# A complex shift on a real reduction: gather into the two planes, sweep each with the real
# multipliers, solve on the planes, scatter.  On a complex reduction the sweeps run on `x`
# itself (the multipliers are complex) and only the Hessenberg solve goes through the planes.
function lhl_ldiv!(x::AbstractVector, sh::LHLShift{Complex{Tr}, Tr}, ws::LHLWorkspace{T}) where {Tr, T}
    n = ws.n
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), the workspace is $n×$n"))
    sh.n == n || throw(DimensionMismatch("the LHLShift is $(sh.n)×$(sh.n), the workspace $n×$n"))
    eltype(x) <: Real &&
        throw(ArgumentError("a real x cannot hold the solution of a complex shift; use a Vector{$(Complex{Tr})}"))
    y = sh.xbuf
    o = sh.po
    Lp = ws.Lp
    if T <: Real
        perm = ws.perm
        isc = ws.iscale
        @inbounds for i in 1:n
            p = perm[i]
            v = x[p] * isc[p]
            y[i] = real(v)
            y[o + i] = imag(v)
        end
        _lhl_zero_planes!(y, n, o)
        _lhl_zinvsweep_buf!(y, 0, o, Lp, n)
        _hessenberg_solve_planar!(y, sh)
        _lhl_zsweep_buf!(y, 0, o, Lp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            p = ip[i]
            x[i] = Complex(y[p], y[o + p]) * d[i]
        end
    elseif T <: _LHL_CFloat && eltype(x) === Complex{Tr}
        perm = ws.perm
        isc = ws.iscale
        @inbounds for i in 1:n
            p = perm[i]
            v = x[p] * isc[p]
            y[i] = real(v)
            y[o + i] = imag(v)
        end
        _lhl_zero_planes!(y, n, o)
        _lhl_zinvsweep_bufc!(y, 0, o, ws.Lpp, n)
        _hessenberg_solve_planar!(y, sh)
        _lhl_zsweep_bufc!(y, 0, o, ws.Lpp, n)
        d = ws.scale
        ip = ws.iperm
        @inbounds for i in 1:n
            p = ip[i]
            x[i] = Complex(y[p], y[o + p]) * d[i]
        end
    else
        applyZinv!(x, ws)
        @inbounds for i in 1:n
            v = x[i]
            y[i] = real(v)
            y[o + i] = imag(v)
        end
        _lhl_zero_planes!(y, n, o)
        _hessenberg_solve_planar!(y, sh)
        @inbounds for i in 1:n
            x[i] = Complex(y[i], y[o + i])
        end
        applyZ!(x, ws)
    end
    return x
end

function _lhl_zero_planes!(y::Vector{T}, n::Int, o::Int) where {T}
    @inbounds for i in (n + 1):o
        y[i] = zero(T)
        y[o + i] = zero(T)
    end
    return y
end


"""
    lhl_refine!(x, A, b, ws, steps)
    lhl_refine!(x, A, b, sh::LHLShift, ws, steps)

Apply `steps` rounds of fixed-precision iterative refinement to a solve of `A x = b`, where
`ws` (with `ws.shift`, or with `sh`) holds the factorization of `A`.  `Z` is not orthogonal,
so the raw solve's backward error carries a factor `κ(Z)`; refinement buys it back for the
price of a matvec and a second `O(n²)` solve (Skeel 1980).  One step is enough to match
LU's backward error even on matrices where `κ(Z)` reaches `10¹⁰`.
"""
lhl_refine!(x::AbstractVector, A, b::AbstractVector, ws::LHLWorkspace, steps::Int) =
    lhl_refine!(x, A, b, ws.shift, ws, steps)

function lhl_refine!(x::AbstractVector, A, b::AbstractVector, sh::LHLShift, ws::LHLWorkspace, steps::Int)
    steps <= 0 && return x
    r = sh.resid
    for _ in 1:steps
        mul!(r, A, x)
        r .= b .- r
        lhl_ldiv!(r, sh, ws)
        x .+= r
    end
    return x
end


"""
    lhl(J; balance = true, shift = eltype(J), thread = Val(true)) -> LHLWorkspace

Reduce `J` to upper Hessenberg form by Gaussian similarity with partial pivoting.  `J` is
not modified; [`lhl!`](@ref) reuses an existing workspace instead of allocating one.
`shift` is the element type of the shifts and solves; `Complex{eltype(J)}` on a real `J`
keeps the reduction real and makes only the shifted half complex (see [`LHLShift`](@ref)).

`thread = Val(true)` lets the blocked reduction (`n ≥ 500` for `Float64`, `512` for
`ComplexF64`, `1024` for `Float32` and `ComplexF32`; other element types stay serial) run
on Polyester threads; threading requires
`using Polyester` (which loads the `LHLFactorizationPolyesterExt` extension) and
`julia -t N` — without either, `Val(true)` silently runs the serial code.  `Val(false)`
(or `false`) keeps it single-threaded.  The threaded work is partitioned independently of
the thread count, so `factors`, `Lp`, `Ht` and the solves are bit-identical with and
without threads.  The shifts and solves are serial and work per right-hand side; nothing
here reads or sets the BLAS thread count.

Follow with [`lhl_shift!`](@ref) to load a shift and [`lhl_ldiv!`](@ref) to solve.
"""
function lhl(J::AbstractMatrix; balance::Bool = true, shift::Type = eltype(J), thread = Val(true))
    ws = LHLWorkspace{eltype(J)}(checksquare(J); shift)
    lhl_reduce!(ws, J, balance, thread)
    return ws
end

"""
    lhl!(ws, J; balance = true, thread = Val(true)) -> ws

Reduce `J` into the existing workspace `ws`, resizing it if needed, and return `ws`.  `J`
is not modified.  The in-place counterpart of [`lhl`](@ref), which documents `balance` and
`thread`; the shift element type is fixed when `ws` is built and is not a keyword here.
"""
function lhl!(ws::LHLWorkspace, J::AbstractMatrix; balance::Bool = true, thread = Val(true))
    lhl_reduce!(ws, J, balance, thread)
    return ws
end

# Compile the serial paths for both float types into the package image
# (`ccall(:jl_generating_output)`: only while precompiling): the unblocked and the blocked
# reduction, real and complex shifts, solve and refinement.  The Polyester extension
# compiles the chunked paths.
if ccall(:jl_generating_output, Cint, ()) == 1
    let
        for T in (Float64, Float32)
            for n in (48, _lhl_block_min(T) + 8)
                J = T[1 / (i + j) + (i == j) for i in 1:n, j in 1:n]
                ws = lhl(J)
                lhl!(ws, J; thread = Val(false))
                _lhl_reduce_blocked!(LHLSerial(), ws.fstore, ws.ipiv, ws.Ht, ws.work, ws.pack, 16, 2)
                lhl_shift!(ws, one(T), -one(T) / 8)
                b = ones(T, n)
                x = lhl_ldiv!(copy(b), ws)
                lhl_refine!(x, J, b, ws, 1)
                sh = LHLShift{Complex{T}}(ws)
                lhl_shift!(sh, ws, one(T), Complex{T}(0, 1) / 8)
                lhl_ldiv!(Complex{T}.(b), sh, ws)
            end
        end
        for T in (ComplexF64, ComplexF32)
            n = 48
            J = T[1 / (i + j) + (i == j) * (1 + im) for i in 1:n, j in 1:n]
            ws = lhl(J)
            lhl!(ws, J; thread = Val(false))
            _lhl_reduce_blocked!(LHLSerial(), ws.fstore, ws.ipiv, ws.Ht, ws.work, ws.pack, 16, 2)
            lhl_shift!(ws, one(T), T(-1 // 8, 1 // 16))
            b = ones(T, n)
            x = lhl_ldiv!(copy(b), ws)
            lhl_refine!(x, J, b, ws, 1)
            applyZinv!(x, ws)
            applyZ!(x, ws)
        end
    end
end

using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    @compile_workload begin
        J = [1.0 0.2 0.0 0.0; 0.0 2.0 0.3 0.0; 0.0 0.0 3.0 0.4; 0.1 0.0 0.0 4.0]
        ws = lhl(J; thread = Val(false))
        lhl_shift!(ws, 1.0, -0.1)
        b = ones(4)
        x = lhl_ldiv!(copy(b), ws)
        lhl_refine!(x, Matrix{Float64}(LinearAlgebra.I, 4, 4) - 0.1 * J, b, ws, 1)
        sh = LHLShift{ComplexF64}(ws)
        lhl_shift!(sh, ws, 1.0, 0.1im)
        lhl_ldiv!(ComplexF64.(b), sh, ws)
    end
end

end # module
