"""
    LHL

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
using LHL, LinearAlgebra

J = randn(400, 400)
ws = lhl(J)                      # O(n³), once
for γ in (0.01, 0.013, 0.02)
    lhl_shift!(ws, 1, -γ)        # O(n²), per shift: builds I - γJ
    x = lhl_ldiv!(copy(b), ws)   # O(n²), per right-hand side
end
```

`ShiftedJacobian(J, γ)` wraps the same idea as an `AbstractMatrix` equal to `I - γJ`, for
handing to a linear-solver interface. LinearSolve.jl's `LHLFactorization` does exactly that.

## Stability

`Z` is not orthogonal, so the backward error carries a factor `κ(Z)` an LU does not have.
Partial pivoting is not optional (without it the method loses every digit on ordinary
matrices), and one step of iterative refinement — see `lhl_refine!` — restores a backward
error comparable to LU's even on the near-nilpotent matrices where `κ(Z)` reaches `10¹⁰`.
"""
module LHL

using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare

export LHLWorkspace, lhl, lhl!, lhl_reduce!, lhl_shift!, lhl_ldiv!, lhl_refine!,
    applyZ!, applyZinv!, ShiftedJacobian, mark_jacobian_updated!, set_shift!

"""
    ShiftedJacobian(J, γ)

Lazy representation of `I - γ*J` as an `AbstractMatrix`: the "W matrix" of an implicit
ODE/DAE solver, kept in the split form so that a solver can react to a change of `γ`
without touching `J`.

Indexing, `mul!`, `Matrix`, `\\` and everything else generic behave exactly as they would
for the assembled `I - γ*J`, so a `ShiftedJacobian` may be handed to any LinearSolve
algorithm.  LinearSolve.jl's `LHLFactorization` is the one that exploits the split: it
factorizes `J` once and then absorbs a new `γ` in `O(n²)`.

More generally a `ShiftedJacobian(J, α, β)` stands for `α*J + β*I`, which is the form an
implicit ODE solver using the W-transform `W = J - M/(dt·γ)` needs.

The shift is mutable — set it with [`set_shift!`](@ref) (or, through a LinearSolve cache,
`update_gamma!`/`update_shift!`).  When the *contents* of `J` change, call
[`mark_jacobian_updated!`](@ref) so that solvers caching a factorization of `J` know to
rebuild it.

!!! note
    A general mass matrix is not supported: only `M = I`.  `M = μI` can be folded by
    solving `I - (γ/μ)J` and scaling the right-hand side by `1/μ`.

## Example

```julia
J = randn(200, 200)
A = ShiftedJacobian(J, 0.01)          # == I - 0.01J
@assert A ≈ I - 0.01J
set_shift!(A, -0.013, 1)              # now == I - 0.013J
```
"""
mutable struct ShiftedJacobian{T, JT <: AbstractMatrix} <: AbstractMatrix{T}
    J::JT
    α::T
    β::T
    jac_version::Int
end

ShiftedJacobian(J::AbstractMatrix, γ::Number) = ShiftedJacobian(J, -γ, oneunit(γ))

function ShiftedJacobian(J::AbstractMatrix, α::Number, β::Number)
    T = promote_type(eltype(J), typeof(α), typeof(β))
    return ShiftedJacobian{T, typeof(J)}(J, convert(T, α), convert(T, β), 0)
end

Base.size(W::ShiftedJacobian) = size(W.J)
Base.@propagate_inbounds function Base.getindex(
        W::ShiftedJacobian{T}, i::Int, j::Int
    ) where {T}
    return W.α * W.J[i, j] + (i == j ? W.β : zero(T))
end
Base.copy(W::ShiftedJacobian) = ShiftedJacobian(copy(W.J), W.α, W.β, W.jac_version)

# A lazy view of `α*J + β*I` has nowhere to put an element. Say so plainly: the default
# `CanonicalIndexError` gives no hint about what to do instead, and the usual way to hit
# this is handing a `ShiftedJacobian` to a solver that factorizes in place.
@noinline function Base.setindex!(W::ShiftedJacobian, args...)
    throw(
        ArgumentError(
            "a ShiftedJacobian is a lazy view of α*J + β*I and cannot be written to elementwise. Use `Matrix(W)` for an algorithm that factorizes in place, or an algorithm that consumes the split form (LinearSolve.jl's `LHLFactorization`)."
        )
    )
end

# Split by rank rather than using `AbstractVecOrMat`: stdlib has separate `mul!` methods
# for the vector and matrix cases, so a single VecOrMat method is ambiguous with both.
for XT in (:AbstractVector, :AbstractMatrix)
    @eval function LinearAlgebra.mul!(
            y::$XT, W::ShiftedJacobian, x::$XT, a::Number, b::Number
        )
        iszero(b) ? fill!(y, zero(eltype(y))) : rmul!(y, b)
        mul!(y, W.J, x, a * W.α, true)
        return y .+= (a * W.β) .* x
    end
end

"""
    mark_jacobian_updated!(A) -> A

Tell `A` that the Jacobian it wraps has new contents, invalidating any cached
factorization of it.  A no-op for anything that is not a [`ShiftedJacobian`](@ref), so it
is safe to call unconditionally.
"""
mark_jacobian_updated!(A) = A
function mark_jacobian_updated!(W::ShiftedJacobian)
    W.jac_version += 1
    return W
end

"""
    LHLWorkspace

Storage for one LHL factorization: the reduction of `J` and the LU of the current shifted
Hessenberg.  Build one with [`lhl`](@ref) or [`lhl!`](@ref).

`factors` holds `H` in `triu(factors, -1)` and the step-`k` multipliers in the annihilated
positions `factors[k+2:n, k]`, exactly the way an LU packs its own.  `Ht` and `Gt` hold
`H` and the LU of `I - γH` **transposed**: the Hessenberg elimination sweeps rows, and in
a column-major array the transposed layout turns every inner loop of the per-γ work
(rebuild, elimination, interchange, back substitution) into a contiguous one.
"""
mutable struct LHLWorkspace{T, Tr}
    factors::Matrix{T}
    ipiv::Vector{Int}
    scale::Vector{Tr}
    Ht::Matrix{T}
    Gt::Matrix{T}
    swap::Vector{Bool}
    resid::Vector{T}
    σ::T
    τ::T
    jac_version::Int
    n::Int
    info::Int
end

function LHLWorkspace{T}(n::Integer) where {T}
    Tr = real(T)
    return LHLWorkspace{T, Tr}(
        Matrix{T}(undef, n, n), Vector{Int}(undef, max(n - 2, 0)), ones(Tr, n),
        Matrix{T}(undef, n, n), Matrix{T}(undef, n, n), Vector{Bool}(undef, n),
        Vector{T}(undef, n), zero(T), zero(T), -1, n, 0
    )
end

function _lhl_resize!(ws::LHLWorkspace{T}, n::Int) where {T}
    n == ws.n && size(ws.factors, 1) == n && return ws
    ws.factors = Matrix{T}(undef, n, n)
    ws.Ht = Matrix{T}(undef, n, n)
    ws.Gt = Matrix{T}(undef, n, n)
    resize!(ws.ipiv, max(n - 2, 0))
    resize!(ws.scale, n)
    resize!(ws.swap, n)
    resize!(ws.resid, n)
    ws.n = n
    ws.jac_version = -1
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
    ipiv = ws.ipiv
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
        # Nₖ⁻¹A (rowᵢ ← rowᵢ - lᵢ·row_{k+1}) and then (Nₖ⁻¹A)Nₖ
        # (col_{k+1} ← col_{k+1} + Σ lᵢ·colᵢ).  The right update must see the columns the
        # left one already transformed, but it needs each column only *after* that column
        # is done — so the two sweep the trailing submatrix together, halving the memory
        # traffic that dominates this loop at any size past cache.
        pk = A[k + 1, k + 1]
        if !iszero(pk)
            @simd for i in (k + 2):n
                A[i, k + 1] -= A[i, k] * pk
            end
        end
        for j in (k + 2):n
            pj = A[k + 1, j]
            if !iszero(pj)
                @simd for i in (k + 2):n
                    A[i, j] -= A[i, k] * pj
                end
            end
            vj = A[j, k]
            if !iszero(vj)
                @simd for i in 1:n
                    A[i, k + 1] += vj * A[i, j]
                end
            end
        end
    end
    Ht = ws.Ht
    @inbounds for j in 1:n, i in 1:min(j + 1, n)
        Ht[j, i] = A[i, j]
    end
    ws.jac_version = -1
    return ws
end

"""
    applyZ!(x, ws)

`x ← Z x`.  `n²/2` multiply–adds.
"""
function applyZ!(x::AbstractVector, ws::LHLWorkspace)
    A = ws.factors
    n = ws.n
    @inbounds for k in (n - 2):-1:1
        xk = x[k + 1]
        if !iszero(xk)
            @simd for i in (k + 2):n
                x[i] += A[i, k] * xk
            end
        end
    end
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
    A = ws.factors
    n = ws.n
    d = ws.scale
    @inbounds @simd for i in 1:n
        x[i] /= d[i]
    end
    @inbounds for k in 1:(n - 2)
        p = ws.ipiv[k]
        p != k + 1 && ((x[k + 1], x[p]) = (x[p], x[k + 1]))
    end
    @inbounds for k in 1:(n - 2)
        xk = x[k + 1]
        if !iszero(xk)
            @simd for i in (k + 2):n
                x[i] -= A[i, k] * xk
            end
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
    Ht = ws.Ht
    Gt = ws.Gt
    swap = ws.swap
    n = ws.n
    σ = convert(T, σ)
    τ = convert(T, τ)
    @inbounds for i in 1:n
        @simd for j in max(i - 1, 1):n
            Gt[j, i] = τ * Ht[j, i]
        end
        Gt[i, i] += σ
    end
    info = 0
    @inbounds for k in 1:(n - 1)
        if abs(Gt[k, k + 1]) > abs(Gt[k, k])
            swap[k] = true
            for j in k:n
                Gt[j, k], Gt[j, k + 1] = Gt[j, k + 1], Gt[j, k]
            end
        else
            swap[k] = false
        end
        pivot = Gt[k, k]
        if iszero(pivot)
            info == 0 && (info = k)
            Gt[k, k + 1] = zero(T)
            continue
        end
        l = Gt[k, k + 1] / pivot
        Gt[k, k + 1] = l
        if !iszero(l)
            @simd for j in (k + 1):n
                Gt[j, k + 1] -= l * Gt[j, k]
            end
        end
    end
    swap[n] = false
    iszero(Gt[n, n]) && info == 0 && (info = n)
    ws.info = info
    return ws
end

function _hessenberg_solve!(x::AbstractVector, ws::LHLWorkspace{T}) where {T}
    Gt = ws.Gt
    swap = ws.swap
    n = ws.n
    @inbounds for k in 1:(n - 1)
        swap[k] && ((x[k], x[k + 1]) = (x[k + 1], x[k]))
        x[k + 1] -= Gt[k, k + 1] * x[k]
    end
    @inbounds for j in n:-1:1
        s = zero(T)
        @simd for i in (j + 1):n
            s += Gt[i, j] * x[i]
        end
        x[j] = (x[j] - s) / Gt[j, j]
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
    set_shift!(W::ShiftedJacobian, α, β) -> W

Set the wrapped shift so that `W == α*J + β*I`.  `set_shift!(W, γ)` is the `I - γ*J` form.
"""
function set_shift!(W::ShiftedJacobian, α::Number, β::Number)
    T = eltype(W)
    W.α = convert(T, α)
    W.β = convert(T, β)
    return W
end
set_shift!(W::ShiftedJacobian, γ::Number) = set_shift!(W, -γ, oneunit(γ))

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
