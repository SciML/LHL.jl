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
  2.6–3.7× faster end to end at matched accuracy. Radau methods, whose stages split into a
  real and a complex shift of the same `J`, hold both against one reduction.
- **Resolvents and transfer functions**, `(sI - A)⁻¹` swept over `s` — complex `s` on a real
  `A` keeps the reduction real (see *Complex shifts on a real matrix*).
- **Shift-and-invert eigenvalue iterations** with a moving shift.

It is *not* a general-purpose `Ax = b` solver. Against a single system an LU wins on every
axis; the reduction here costs `5/3 n³` against an LU's `2/3 n³`.

## Cost

|            | reduction | new shift | solve |
|------------|-----------|-----------|-------|
| fresh LU per shift | — | `2/3 n³` | `2n²` |
| LHL        | `5/3 n³`  | `n²`      | `3n²` |

Measured against LAPACK on one thread, a new shift is 25× cheaper than a refactorization at
`n = 50`, 50× at `n = 256` and 100–130× at `n = 1024–2000`. The solve costs about what an
LU's triangular solves do (0.5× at `n ≤ 64`, 0.8× at `n = 256`, 1.0–1.25× at
`n = 512–2000`), so the trade pays whenever a factorization serves more than a handful of
shifts.

The reduction is unblocked below `n ≈ 500` (Float64; 1000 for Float32) and blocked above:
delayed panel updates with rank-`nb` GEMMs in a register-blocked pure-Julia microkernel, no
BLAS. On one thread it runs at 1× a LAPACK LU for `n ≤ 64`, 2.7× at `n = 256`, 3.8× at
`n = 1024` and 5–6× at `n = 1600–2000`, where the trailing GEMV that dominates it is DRAM-bound.
It is faster than LAPACK's blocked *orthogonal* Hessenberg reduction
(`LinearAlgebra.hessenberg`, `dgehrd`) at every size measured — 3.6× at `n = 64`, 2× at
`n = 256–1024`, 1.3–1.6× at `n = 2000` — so `hessenberg` is not a better front end at any
size; its stdlib shifted solve also re-factorizes the Hessenberg on *every* solve rather
than caching one per shift.

### Threading

`lhl(J; thread = Val(true))` (the default; `Val(false)` or `false` to disable) runs the
blocked reduction on [Polyester](https://github.com/JuliaSIMD/Polyester.jl) threads.
Threading requires `using Polyester` (which loads the `LHLFactorizationPolyesterExt`
extension; Polyester is a weak dependency) and `julia -t N`; without either,
`thread = Val(true)` silently runs the serial code.  Threaded: the per-step trailing GEMV in column groups, the panel's row work
in row chunks, and the panel-end GEMMs, triangular solve and interchange sweeps in the same
column groups.  Every partition depends on the sizes only, so `factors`, `Lp`, `Ht` and the
solves are bit-identical for any thread count.  Measured on an EPYC 7502 (Float64, Julia
1.10, min over repeats, threads pinned to as few 4-core CCXs as possible), the reduction
runs 1.5×/2.2×/2.0×/1.7× faster at `n = 512`, 1.7×/2.6×/2.8×/3.0× at `n = 1024` and
1.6×/1.9×/5.3×/8.4× at `n = 2000` with 2/4/8/16 threads (a step's trailing GEMV moves from
50 GB/s out of one core's L3 to 200–350 GB/s out of two to four CCXs' caches; the per-step
row work and the balancing are what remains serial-bound).  The unblocked path (`n < 500`)
and the shifts and solves stay serial (a row-split unblocked sweep measured slower than
serial: the per-step row interchange moves two rows' worth of cache lines between the
cores).  `using Polyester` adds about 0.15–0.35 s of load time; precompile workloads in the
package and the extension keep the first `lhl`/`lhl_shift!`/`lhl_ldiv!` under ~0.1 s in
both configurations.

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

## Complex shifts on a real matrix

Two consumers want a real `J` with complex shifts: resolvents and transfer functions
`(sI - A)⁻¹` swept over complex `s`, and Radau-type implicit Runge–Kutta methods, whose
stage system splits into a real `(I - γhJ)` and a complex `(I - (α + βi)hJ)` against the
same Jacobian. Promoting `J` to complex would make the `O(n³)` reduction complex — about
6–7× the flops of the real one — and every solve a fully complex one. Instead the reduction
stays real and only the shifted half is complex:

```julia
ws = lhl(J; shift = ComplexF64)      # real reduction, complex shifts (also lhl! on it)
for s in svals                       # resolvent sweep
    lhl_shift!(ws, s, -1)            # loads sI - J
    x = lhl_ldiv!(copy(b), ws)       # b, x complex
end
```

The `shift` type is `eltype(J)` or `Complex{eltype(J)}`; a complex shift on a workspace built
for real ones throws an `ArgumentError`, as does a real `x` for a complex solve. To hold several
shifts against one reduction at once — the Radau case — build extra shift states and pass them
explicitly; the workspace's own `ws.shift` is what the two-argument forms use:

```julia
ws = lhl(J)                          # real reduction, real ws.shift
sh = LHLShift{ComplexF64}(ws)        # a second, complex shift state on the same reduction
lhl_shift!(ws, 1, -γ * h)            # (I - γhJ)
lhl_shift!(sh, ws, 1, -(α + β * im) * h)
x1 = lhl_ldiv!(copy(b1), ws)         # real
x2 = lhl_ldiv!(copy(b2), sh, ws)     # complex
lhl_refine!(x2, I - (α + β * im) * h * J, b2, sh, ws, 1)   # optional
```

`lhl!(ws, J)` re-reduces; a held `LHLShift` follows the workspace's size on its next
`lhl_shift!`. Internally the complex `Gt` and the solve buffers are stored *planar* (real and
imaginary parts in separate rows), so the `Z` sweeps run the real multipliers over both planes
in one pass and every complex product is four real multiply–adds with no lane shuffling.

Measured on one thread (Float64, Julia 1.10, µs; `rc` = real reduction with a complex shift,
`rr` = real shift, `cc` = fully complex workspace on the same `J`):

| n | shift rr / rc / cc | solve rr / rc / cc | reduction real / complex |
|---|---|---|---|
| 16 | 0.28 / 1.21 / 1.26 | 0.25 / 0.44 / 0.71 | 2.2 / 7.9 |
| 64 | 1.5 / 6.1 / 7.3 | 1.2 / 2.5 / 5.5 | 44 / 287 |
| 256 | 16 / 41 / 52 | 15 / 28 / 70 | 2 180 / 13 940 |
| 1024 | 214 / 549 / 860 | 255 / 525 / 1 257 | 112 000 / 818 000 |

A complex shift costs 2.5–4.5× a real one (its `Gt` is twice the bytes) and its solve 1.8–2.1×
a real solve — the arithmetic ratio of complex-on-real work — where the fully complex workspace
pays 3–5× on the shift, 2.9–4.9× on the solve and 3.6–7.3× on the reduction, whose real form is
the part that is saved. Float32 ratios are lower still (shift 2.1–3×, solve 1.5–2×); Julia 1.12
matches 1.10 within a few percent on the shift and the solve. The real reduction's pivots are
real, and its forward error is within a factor of a few of the complex reduction's on the same
problem.

## Limits

- Dense matrices only — the reduction fills in, so sparsity buys nothing.
- Real or complex `AbstractMatrix` with a scalar element type; shifts of the matrix's element
  type or its complex extension (`Complex{Float32}` shifts on a `Float64` matrix, say, are not).
- No generalized (pencil) form: `M - γJ` for a general mass matrix `M` would need a
  Hessenberg–triangular reduction, which is not implemented. `M = μI` folds into the shift.

## References

- J.H. Wilkinson, *The Algebraic Eigenvalue Problem*, Oxford, 1965 — the elimination method
  and its growth analysis.
- W.H. Enright, "Improving the efficiency of matrix operations in the numerical solution of
  stiff ODEs", *ACM TOMS* 4(2), 1978 — the Hessenberg-reuse idea for stiff solvers.
- R.D. Skeel, "Iterative refinement implies numerical stability for Gaussian elimination",
  *Math. Comp.* 35, 1980 — why one refinement step suffices.
