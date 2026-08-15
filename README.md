# LHLFactorization.jl

[![CI](https://github.com/SciML/LHLFactorization.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SciML/LHLFactorization.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Solve a whole family of shifted linear systems

```
(σI + τJ) x = b
```

for many `(σ, τ)` against **one** `O(n³)` factorization of `J`. Each new shift costs
`O(n²)`; each solve costs `O(n²)`.

The trick is old and underused: reduce `J` once to upper Hessenberg form by a similarity
transform,

```
J = Z H Z⁻¹
```

The shift never reaches `Z`, so

```
σI + τJ = Z (σI + τH) Z⁻¹
```

and `σI + τH` is Hessenberg — its LU is `O(n²)`, not `O(n³)`. LHLFactorization.jl does the reduction by
Gaussian elimination-style similarity transformations with partial pivoting (Wilkinson's
elimination method, EISPACK's `ELMHES`), giving `Z = D·P·L` with `L` unit lower triangular,
`P` a permutation and `D` a balancing diagonal — hence the name.

## Install

```julia
using Pkg; Pkg.add("LHLFactorization")
```

## Use

```julia
using LHLFactorization, LinearAlgebra

J = randn(400, 400)
b = randn(400)

ws = lhl(J)                       # O(n³), once

for γ in (0.01, 0.013, 0.021)
    lhl_shift!(ws, 1, -γ)         # O(n²): load W = I - γJ
    x = lhl_ldiv!(copy(b), ws)    # O(n²): solve
    lhl_refine!(x, I - γ * J, b, ws, 1)   # optional, see Stability
end
```

`lhl_shift!(ws, σ, τ)` loads `σI + τJ`. `(1, -γ)` gives `I - γJ`; `(0, 1)` gives `J`
itself; `(-1/(dt*γ), 1)` gives the W-transform `J - I/(dt·γ)` an implicit ODE solver uses.

To drive this from a solver interface rather than by hand, LinearSolve.jl's
`LHLFactorization` consumes a `SciMLOperators.WOperator` — the split `J - M/γ` an implicit
ODE solver already builds — and moves the shift with `update_gamma!`.

```julia
using LinearSolve, SciMLOperators
W = WOperator{true}(I, 0.01, J, similar(b))
cache = init(LinearProblem(W, b), LHLFactorization())
u1 = solve!(cache).u
update_gamma!(cache, 0.013)       # O(n²): reuses the reduction of J
u2 = solve!(cache).u
```

!!! note
    A user loading both this package and LinearSolve.jl sees the module name
    `LHLFactorization` shadow LinearSolve's algorithm type of the same name. Reach for
    `LinearSolve.LHLFactorization` in that case; `using LinearSolve` alone is unambiguous.

## Who this is for

Anything that solves the same matrix shifted many ways:

- **Stiff ODE/DAE solvers.** The iteration matrix is `W = I - γJ` with `γ = c·dt`. Adaptive
  step-size control moves `γ` every step while `J` is held fixed for tens of them, so the
  conventional solver pays `O(n³)` per step to absorb a scalar. Via LinearSolve.jl's
  `LHLFactorization` this is a drop-in change; on a dense 800-unknown Brusselator it is
  2.6–3.7× faster end to end at matched accuracy.
- **Resolvents and transfer functions**, `(sI - A)⁻¹` swept over `s`.
- **Shift-and-invert eigenvalue iterations** with a moving shift.

It is *not* a general-purpose `Ax = b` solver. Against a single system an LU wins on every
axis; the reduction here costs `5/3 n³` against an LU's `2/3 n³`.

## Cost

|            | reduction | new shift | solve |
|------------|-----------|-----------|-------|
| fresh LU per shift | — | `2/3 n³` | `2n²` |
| LHL        | `5/3 n³`  | `n²`      | `3n²` |

Measured against LAPACK on one thread, a new shift is 25× cheaper than a refactorization at
`n = 50`, 50× at `n = 256` and 80–95× at `n = 1600–2000`. The solve costs about what an
LU's triangular solves do (0.6× at `n ≤ 64`, 1.0–1.2× at `n = 256–2000`), so the trade
pays whenever a factorization serves more than a handful of shifts.

The reduction is unblocked below `n ≈ 250` (Float64; 500 for Float32) and blocked above:
delayed panel updates with rank-`nb` GEMMs in a register-blocked pure-Julia microkernel, no
BLAS. On one thread it runs at 1× a LAPACK LU for `n ≤ 64`, 3× at `n = 256`, 4× at
`n = 1024` and 7–8× at `n = 2000`, where the trailing GEMV that dominates it is DRAM-bound.
It is faster than LAPACK's blocked *orthogonal* Hessenberg reduction
(`LinearAlgebra.hessenberg`, `dgehrd`) at every size measured — 3.6× at `n = 64`, 2× at
`n = 256–1024`, 1.3–1.6× at `n = 2000` — so `hessenberg` is not a better front end at any
size; its stdlib shifted solve also re-factorizes the Hessenberg on *every* solve rather
than caching one per shift.

## Stability

`Z` is not orthogonal, so unlike an LU the backward error carries a factor `κ(Z)`.

**Partial pivoting is not optional.** It bounds every multiplier by 1. Without it the
method loses every digit on ordinary matrices — a random Gaussian matrix already gives
`10⁻¹` backward error. It is always on; there is no switch.

**`κ(Z)` is usually small but not always.** On typical Jacobians it is `10²`–`10⁴`. An
adversarial search over six matrix families found it reaching `8×10¹⁰` on near-nilpotent
matrices, where every pivot of the reduction is near zero, costing eight digits.

**One refinement step removes the penalty.** `lhl_refine!` costs a matvec and a second
`O(n²)` solve, and restores a backward error comparable to LU's:

| matrix | κ(Z) | no refinement | 1 step | LU |
|---|---|---|---|---|
| randn | 2.3e3 | 1.3e-15 | 1.9e-16 | 4.5e-16 |
| Wilkinson growth | 9.9e3 | 1.4e-14 | 6.5e-17 | 1.9e-16 |
| near-nilpotent | 7.5e10 | **5.3e-09** | **1.3e-16** | 4.0e-16 |
| clustered spectrum | 3.7e3 | 1.0e-14 | 2.0e-16 | 5.4e-16 |

Inside a Newton loop refinement is usually the wrong trade: it roughly triples a solve, and
an implicit solver runs on the order of fourteen solves per shift, so it is charged against
exactly the quantity the method is saving. Newton is itself a correction loop, so an
inexact linear solve costs at most an extra iteration.

Matrices that are notoriously bad for LU — Wilkinson's `2ⁿ⁻¹` growth matrix, the Kahan
matrix — cause no trouble here. There is a structural reason the shifted half is safe: an
upper Hessenberg matrix has one subdiagonal, so each elimination step chooses between two
candidates and the growth factor of the Hessenberg LU is bounded by `n`, against `2ⁿ⁻¹` for
general partial pivoting. The whole exposure sits in `Z`.

## Limits

- Dense matrices only — the reduction fills in, so sparsity buys nothing.
- Real or complex `AbstractMatrix` with a scalar element type.
- No generalized (pencil) form: `M - γJ` for a general mass matrix `M` would need a
  Hessenberg–triangular reduction, which is not implemented. `M = μI` folds into the shift.

## References

- J.H. Wilkinson, *The Algebraic Eigenvalue Problem*, Oxford, 1965 — the elimination method
  and its growth analysis.
- W.H. Enright, "Improving the efficiency of matrix operations in the numerical solution of
  stiff ODEs", *ACM TOMS* 4(2), 1978 — the Hessenberg-reuse idea for stiff solvers.
- R.D. Skeel, "Iterative refinement implies numerical stability for Gaussian elimination",
  *Math. Comp.* 35, 1980 — why one refinement step suffices.
