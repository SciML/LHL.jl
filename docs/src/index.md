# LHLFactorization.jl

LHLFactorization reduces a dense square matrix `J` to upper Hessenberg form once and reuses
that reduction for a whole family of shifted systems `(σI + τJ) x = b`. The reduction is
`O(n³)` and happens once; loading a new shift and solving a right-hand side are both
`O(n²)`. That is the trade an implicit ODE solver wants, where `γ` moves every step while
`J` is held for tens of steps.

## Solve shifted systems

```@example quickstart
using LHLFactorization, LinearAlgebra

J = randn(40, 40)
b = randn(40)
ws = lhl(J)                        # O(n³), once

for γ in (0.01, 0.02, 0.04)
    lhl_shift!(ws, 1.0, -γ)        # O(n²), per shift: builds I - γJ
    x = lhl_ldiv!(copy(b), ws)     # O(n²), per right-hand side
    @assert (I - γ * J) * x ≈ b
end
```

A workspace is reused for a new matrix with [`lhl!`](@ref) and for many right-hand sides
with [`lhl_ldiv!`](@ref). [`lhl_reduce!`](@ref) is the non-keyword reduction entry point,
and [`applyZ!`](@ref) / [`applyZinv!`](@ref) apply the similarity transformation itself.

## Backward error and refinement

The similarity `Z` is not orthogonal, so the raw solve's backward error carries a factor
`κ(Z)` that an LU does not have. One step of [`lhl_refine!`](@ref) buys it back for the
price of a matvec and a second `O(n²)` solve.

```@example quickstart
γ = 0.05
A = I - γ * J
lhl_shift!(ws, 1.0, -γ)
x = lhl_ldiv!(copy(b), ws)
lhl_refine!(x, A, b, ws, 1)
@assert A * x ≈ b
```

## Complex shifts on a real reduction

A Radau-type implicit Runge–Kutta step needs one real and one complex shift against the
same real Jacobian. Passing `shift = Complex{T}` keeps the `O(n³)` reduction real and makes
only the `O(n²)` shifted half complex. An [`LHLShift`](@ref) is the shift-dependent half on
its own, so several shifts can be held against one reduction at the same time and passed
explicitly to the three-argument [`lhl_shift!`](@ref) and [`lhl_ldiv!`](@ref).

```@example quickstart
ws = lhl(J; shift = ComplexF64)        # real reduction, complex shifts
lhl_shift!(ws, 1.0, -0.05 + 0.02im)
z = lhl_ldiv!(ComplexF64.(b), ws)
@assert (I - (0.05 - 0.02im) * J) * z ≈ b

wr = lhl(J)                            # a real workspace ...
sh = LHLShift{ComplexF64}(wr)          # ... with an extra complex shift alongside it
lhl_shift!(wr, 1.0, -0.05)             # real shift, into wr.shift
lhl_shift!(sh, wr, 1.0, -0.05 + 0.02im)   # complex shift, into sh
xr = lhl_ldiv!(copy(b), wr)
xc = lhl_ldiv!(ComplexF64.(b), sh, wr)
@assert (I - 0.05 * J) * xr ≈ b
@assert (I - (0.05 - 0.02im) * J) * xc ≈ b
```

## Threading

The blocked reduction (`n ≥ 500` for `Float64`, `n ≥ 1024` for `Float32`; other element
types stay serial) runs on [Polyester](https://github.com/JuliaSIMD/Polyester.jl) threads
when `thread = Val(true)` — the default — *and* Polyester is loaded. Polyester is a weak
dependency: `using Polyester` loads the `LHLFactorizationPolyesterExt` extension, and
without it (or without `julia -t N`) `Val(true)` silently runs the serial code. `Val(false)`
keeps it single-threaded.

```@example threading
using LHLFactorization, Polyester       # loads LHLFactorizationPolyesterExt

J = randn(600, 600)
ws = lhl(J; thread = Val(true))         # threaded when julia runs with -t N
serial = lhl(J; thread = Val(false))
@assert ws.factors == serial.factors    # bit-identical, whatever the thread count
```

The threaded reduction partitions its work independently of the thread count, so
`factors`, `Lp`, `Ht` and every solve are bit-identical with and without threads. Shifts
and solves are serial and work per right-hand side; nothing in the package reads or sets
the BLAS thread count. A workspace uses its own scratch buffers during a solve, so
concurrent solves need one workspace each.

See the [API reference](api.md) for the complete workspace lifecycle.
