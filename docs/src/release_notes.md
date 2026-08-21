# Release Notes

## Unreleased

  - **Explicit-vector kernels for fully complex workspaces** (`lhl(J::Matrix{ComplexF64})`
    and `ComplexF32`). The reduction's trailing update, trailing GEMM and panel GEMV now
    run real explicit-vector kernels on the interleaved storage, and the solves' Z sweeps
    run on real planes (a planar copy of the packed multipliers). Measured on one Zen2
    core, a `ComplexF64` reduction runs at 2–3.8× the same-size real one (the flop ratio
    is 4) instead of 6.4–7.5×, and `lhl_ldiv!` at ≈2.5× instead of ≈4.7×. The threaded
    reduction now covers `ComplexF64` (`n ≥ 512`) and `ComplexF32` (`n ≥ 1024`), and is
    still bit-identical for any thread count. The `factors` layout is unchanged.

  - **Complex pivot magnitudes and balance norms are now `|re| + |im|`** (LAPACK's
    `CABS1`, as the shifted LU already used), replacing `abs`. Pivot *choices* of a
    complex reduction can therefore differ from v2.0.0 in near-ties; results differ at
    rounding level, both choices are equally valid partial pivoting, and complex
    multipliers are now bounded by `√2` in modulus (LAPACK's own bound for `zgetrf`)
    rather than 1. Real workspaces are bit-for-bit unaffected.

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
