using LHLFactorization, LinearAlgebra, Test

@testset "generic workspace lifecycle" begin
    J = randn(5, 5)
    ws = LHLWorkspace{Float64}(5)

    @test lhl_reduce!(ws, view(J, :, :), false) === ws
    @test lhl_shift!(ws, 1.0, -0.1) === ws

    b = randn(5)
    expected = Matrix(I - 0.1 * J) \ b
    x = lhl_ldiv!(view(copy(b), :), ws)
    @test x ≈ expected

    x0 = randn(5)
    transformed = copy(x0)
    @test applyZ!(transformed, ws) === transformed
    @test applyZinv!(transformed, ws) === transformed
    @test transformed ≈ x0
end

@testset "generic allocating and reusing entry points" begin
    J = randn(4, 4)
    ws = lhl(view(J, :, :); balance = false)
    @test ws isa LHLWorkspace

    J2 = randn(6, 6)
    @test lhl!(ws, J2; balance = true) === ws
    lhl_shift!(ws, 1.0, 0.2)
    b = randn(6)
    A = Matrix(I + 0.2 * J2)
    x = lhl_ldiv!(copy(b), ws)
    @test x ≈ A \ b
end

@testset "generic iterative refinement" begin
    J = randn(7, 7)
    b = randn(7)
    ws = lhl(J)
    A = Matrix(I - 0.05 * J)
    lhl_shift!(ws, 1.0, -0.05)
    x = lhl_ldiv!(copy(b), ws)
    @test lhl_refine!(x, A, b, ws, 1) === x
    @test A * x ≈ b rtol = 1.0e-10
end
