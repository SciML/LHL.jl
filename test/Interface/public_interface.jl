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

@testset "generic complex shifts against a real reduction" begin
    J = randn(9, 9)
    b = randn(9)
    τ = -0.05 + 0.02im

    ws = lhl(view(J, :, :); shift = ComplexF64)
    @test ws isa LHLWorkspace{Float64, Float64, ComplexF64}
    lhl_shift!(ws, 1.0, τ)
    z = lhl_ldiv!(ComplexF64.(b), ws)
    @test z ≈ Matrix(I + τ * J) \ b

    # a second, complex shift held alongside the workspace's own real one
    wr = lhl(J)
    sh = LHLShift{ComplexF64}(wr)
    @test sh isa LHLShift{ComplexF64}
    @test lhl_shift!(wr, 1.0, -0.05) === wr
    @test lhl_shift!(sh, wr, 1.0, τ) === sh
    @test lhl_ldiv!(copy(b), wr) ≈ Matrix(I - 0.05 * J) \ b
    zc = lhl_ldiv!(ComplexF64.(b), sh, wr)
    @test zc ≈ Matrix(I + τ * J) \ b
    @test lhl_refine!(zc, Matrix(I + τ * J), ComplexF64.(b), sh, wr, 1) === zc
    @test Matrix(I + τ * J) * zc ≈ b rtol = 1.0e-10

    @test LHLShift{ComplexF64}(9) isa LHLShift{ComplexF64}
    @test_throws ArgumentError LHLWorkspace{Float64}(9; shift = ComplexF32)
end

@testset "generic thread keyword and shift-field forwarding" begin
    J = randn(11, 11)
    for thread in (Val(true), Val(false), true, false)
        ws = LHLWorkspace{Float64}(11)
        @test lhl_reduce!(ws, J, true, thread) === ws
        @test lhl!(ws, J; thread) === ws
        @test lhl(J; thread) isa LHLWorkspace
    end

    ws = lhl(J)
    lhl_shift!(ws, 1.0, -0.25)
    # the shift-dependent fields are reachable through the workspace as well
    for name in (:Gt, :rdiag, :swap, :resid, :σ, :τ, :info)
        @test name in propertynames(ws)
        @test getproperty(ws, name) === getproperty(ws.shift, name)
    end
    @test ws.info == 0
end
