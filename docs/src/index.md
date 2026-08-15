# LHLFactorization.jl

LHLFactorization reduces a dense square matrix to upper Hessenberg form once and reuses the
reduction for a family of shifted linear systems.

```julia
using LHLFactorization

J = randn(40, 40)
b = randn(40)
ws = lhl(J)
lhl_shift!(ws, 1.0, -0.01)
x = lhl_ldiv!(copy(b), ws)
```

The reduction is `O(n^3)`. Loading a new shift and solving a right-hand side are both
`O(n^2)`. See the [API reference](api.md) for the workspace lifecycle and the
in-place transformation functions.
