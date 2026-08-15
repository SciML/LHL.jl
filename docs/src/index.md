# LHLFactorization.jl

LHLFactorization reduces a dense square matrix to upper Hessenberg form once and reuses the
reduction for a family of shifted linear systems.

## Solve shifted systems

```julia
using LHLFactorization, LinearAlgebra

J = randn(40, 40)
ws = lhl(J)

for γ in (0.01, 0.02, 0.04)
    rhs = randn(40)
    lhl_shift!(ws, 1.0, -γ)       # load I - γJ
    x = lhl_ldiv!(copy(rhs), ws)  # overwrite the copy with the solution
    @assert (I - γ * J) * x ≈ rhs
end
```

The reduction is `O(n^3)`. Loading a new shift and solving a right-hand side are both
`O(n^2)`. A workspace can be reused for a new matrix with `lhl!` and for many right-hand
sides with `lhl_ldiv!`. See the [API reference](api.md) for the complete workspace
lifecycle and the in-place transformation functions.
