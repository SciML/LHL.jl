using LHLFactorization, LinearAlgebra, Random, Test

bwd(W, x, b) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))

@testset "reduction reconstructs J" begin
    for n in (1, 2, 3, 4, 17, 80), balance in (true, false)
        J = randn(MersenneTwister(n), n, n)
        ws = lhl(J; balance)
        Z = Matrix{Float64}(I, n, n)
        Zi = Matrix{Float64}(I, n, n)
        for j in 1:n
            applyZ!(view(Z, :, j), ws)
            applyZinv!(view(Zi, :, j), ws)
        end
        @test Z * Zi ≈ I
        H = triu(ws.factors, -1)
        @test Z * H * Zi ≈ J
        @test all(abs.(tril(ws.factors, -2)) .<= 1)   # partial pivoting bounds multipliers
    end
end

@testset "shifted solves" begin
    for n in (1, 2, 3, 5, 40, 129)
        J = randn(MersenneTwister(n), n, n)
        b = randn(MersenneTwister(n + 7), n)
        ws = lhl(J)
        for (σ, τ) in ((1.0, 0.0), (1.0, -1.0e-8), (1.0, -0.05), (1.0, 1.7), (0.0, 1.0), (-3.0, 2.0))
            iszero(σ) && iszero(τ) && continue
            W = σ * I + τ * J
            rank(Matrix(W)) == n || continue
            lhl_shift!(ws, σ, τ)
            @test lhl_ldiv!(copy(b), ws) ≈ Matrix(W) \ b rtol = 1.0e-9
        end
    end
end

@testset "workspace reuse across shifts and Jacobians" begin
    n = 50
    J = randn(MersenneTwister(3), n, n)
    b = randn(MersenneTwister(4), n)
    ws = lhl(J)
    for γ in (0.02, 1.0e-6, 5.0, 0.02)
        lhl_shift!(ws, 1, -γ)
        @test lhl_ldiv!(copy(b), ws) ≈ Matrix(I - γ * J) \ b rtol = 1.0e-9
    end
    J2 = randn(MersenneTwister(5), n, n)
    lhl!(ws, J2)
    lhl_shift!(ws, 1, -0.3)
    @test lhl_ldiv!(copy(b), ws) ≈ Matrix(I - 0.3 * J2) \ b rtol = 1.0e-9
    # a workspace can be resized onto a different n
    J3 = randn(MersenneTwister(6), 2n, 2n)
    lhl!(ws, J3)
    lhl_shift!(ws, 1, -0.3)
    b3 = randn(MersenneTwister(7), 2n)
    @test lhl_ldiv!(copy(b3), ws) ≈ Matrix(I - 0.3 * J3) \ b3 rtol = 1.0e-9
end

@testset "singular shift is reported, not thrown" begin
    ws = lhl([1.0 0.0; 0.0 2.0])
    lhl_shift!(ws, 1, -1.0)          # I - J is singular in the first coordinate
    @test ws.info != 0
end

@testset "complex" begin
    n = 30
    J = randn(MersenneTwister(13), ComplexF64, n, n)
    b = randn(MersenneTwister(14), ComplexF64, n)
    γ = 0.2 + 0.3im
    ws = lhl(J)
    lhl_shift!(ws, 1, -γ)
    @test lhl_ldiv!(copy(b), ws) ≈ Matrix(I - γ * J) \ b rtol = 1.0e-9
end

@testset "pivoting is what makes it work" begin
    # Every multiplier is bounded by 1, so the reduction cannot blow up the way an
    # unpivoted one does. Checked here on the matrix families that break without it.
    n = 100
    mats = Dict(
        "randn" => randn(MersenneTwister(1), n, n),
        "wilkinson" => [
            i == j ? 1.0 : (j == n ? 1.0 : (i > j ? -1.0 : 0.0))
                for i in 1:n, j in 1:n
        ],
        "graded" => Diagonal(10.0 .^ range(-8, 8, n)) * randn(MersenneTwister(3), n, n) *
            Diagonal(10.0 .^ range(8, -8, n)),
    )
    b = randn(MersenneTwister(42), n)
    for (name, J) in mats
        ws = lhl(J)
        @test maximum(abs, tril(ws.factors, -2)) <= 1 + eps()
        for γ in (1.0e-3, 1.0, 1.0e3)
            lhl_shift!(ws, 1, -γ)
            ws.info == 0 || continue
            x = lhl_ldiv!(copy(b), ws)
            W = Matrix(I - γ * J)
            lhl_refine!(x, W, b, ws, 1)
            @test bwd(W, x, b) < 1.0e-13
        end
    end
end

@testset "refinement recovers backward error where κ(Z) is enormous" begin
    # Strictly upper triangular with a tiny corner: every pivot of the reduction is near
    # zero, κ(Z) reaches 1e10, and the unrefined solve loses ~8 digits.
    n = 60
    J = triu(randn(MersenneTwister(17), n, n), 1)
    J[n, 1] = 1.0e-8
    b = randn(MersenneTwister(18), n)
    γ = 0.5
    W = Matrix(I - γ * J)
    ws = lhl(J)
    lhl_shift!(ws, 1, -γ)
    raw = lhl_ldiv!(copy(b), ws)
    refined = lhl_refine!(copy(raw), W, b, ws, 1)
    @test bwd(W, refined, b) <= bwd(W, raw, b)
    @test bwd(W, refined, b) < 1.0e-13
end

@testset "allocation-free once the workspace exists" begin
    n = 64
    J = randn(MersenneTwister(21), n, n)
    b = randn(MersenneTwister(22), n)
    ws = lhl(J)
    x = copy(b)
    lhl_shift!(ws, 1, -0.3)
    lhl_ldiv!(x, ws)
    @test @allocated(lhl_shift!(ws, 1, -0.4)) == 0
    @test @allocated(lhl_ldiv!(x, ws)) == 0
end
