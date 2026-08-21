# Release Notes

## Unreleased

### Added

  - **Adjoint solves from the existing reduction**: `lhl_ldivH!` solves `Wᴴ x = b`
    against the same reduction and the same shift LU as `lhl_ldiv!`
    (`Wᴴ = Z⁻ᴴ Gᴴ Zᴴ` — three `O(n²)` phases, no refactorization), with `lhl_refineH!`
    for iterative refinement against `Aᴴ` and `applyZH!`/`applyZinvH!` for the adjoint
    similarity transformations. A complex shift on a real reduction conjugates only the
    shifted half: `Zᴴ = Zᵀ` stays real.

## v2.0

Breaking. The changes below are mechanical to adopt; `lhl`, `lhl!`, `lhl_reduce!`,
`applyZ!`/`applyZinv!` and the `LHLWorkspace{T}(n)` constructor are all source compatible
with v1.

### Breaking

  - **`LHLWorkspace` gained a third type parameter**, `LHLWorkspace{T, Tr}` →
    `LHLWorkspace{T, Tr, TG}`, where `TG` is the element type the shifted Hessenberg is
    held in. Only code that *writes the parameters out* is affected: an annotation
    `ws::LHLWorkspace{Float64, Float64}` must become `LHLWorkspace{Float64, Float64,
    Float64}` or simply `ws::LHLWorkspace`. The constructor `LHLWorkspace{T}(n)` is
    unchanged, so most code needs no edit at all.

  - **`lhl_ldiv!` now throws `DimensionMismatch`** when `x`'s length does not match the
    workspace. v1 read past the end and returned a wrong answer silently, so anything this
    now catches was already broken.

  - **The shift moved out of the workspace into `LHLShift`.** The shift-taking methods gain
    a shift argument — `lhl_shift!(sh, ws, σ, τ)`, `lhl_ldiv!(x, sh, ws)`,
    `lhl_refine!(x, A, b, sh, ws, steps)`. The one-shift forms `lhl_shift!(ws, σ, τ)`,
    `lhl_ldiv!(x, ws)` and `lhl_refine!(x, A, b, ws, steps)` still work and operate on the
    workspace's own shift, `ws.shift`; `ws.σ`, `ws.τ` and `ws.info` still read and write
    through to it. Code using one shift per reduction needs no change.

### Added

  - **`LHLShift{TG}`** — a shift is now a first-class object, so **one reduction can serve
    several shifts**. Build extra ones with `LHLShift{TG}(ws)`.

  - **Complex shifts on a real reduction**: `lhl(J; shift = ComplexF64)`, or
    `LHLWorkspace{T}(n; shift = TG)`. A real `J` keeps a real `O(n³)` reduction while the
    shift alone goes complex — the shape RadauIIA needs, where a real and a complex stage
    matrix come from one Jacobian.

  - **Threading**: `lhl(J; thread = Val(true))` (the default) and
    `lhl_reduce!(ws, J, balance, thread)` run the blocked reduction on
    [Polyester](https://github.com/JuliaSIMD/Polyester.jl) threads when Polyester is loaded
    and `Threads.nthreads() > 1`. Deterministic: the result is bit-identical for any thread
    count. Pass `Val(false)` to disable.

### Performance

  - Rewritten kernels. Measured against LAPACK on one thread, a re-shift is 14× cheaper
    than an LU refactorization at `n = 25` and 108× at `n = 800`, and `lhl_ldiv!` is now
    about the cost of an LU's triangular solves rather than twice it.

### Notes

  - **One workspace, and one `LHLShift`, cannot serve concurrent solves** — both write
    scratch buffers. Give each thread its own.
