using LHLFactorization, LinearAlgebra, Random, Test

bwd(W, x, b) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))

@testset "reduction reconstructs J" begin
    for n in (1, 2, 3, 4, 17, 80, 128, 200, 301), balance in (true, false)
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
    for n in (1, 2, 3, 5, 40, 129, 260)
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

@testset "explicit-vector solve kernels agree with the generic ones" begin
    for T in (Float64, Float32), n in (3, 7, 33, 130, 331, 500)
        J = randn(MersenneTwister(n), T, n, n)
        b = randn(MersenneTwister(n + 7), T, n)
        ws = lhl(J)
        lhl_shift!(ws, 1, -0.05)
        W = I - T(0.05) * J
        x = lhl_ldiv!(copy(b), ws)                    # Vector{T}: explicit kernels
        y = lhl_ldiv!(view(copy(b), :), ws)          # view: generic path
        for z in (x, y)
            @test z ≈ W \ b rtol = (T == Float32 ? 1.0e-3 : 1.0e-9)
            @test bwd(W, z, b) < 50 * eps(T)
        end
    end
end

@testset "blocked reduction agrees with the unblocked one" begin
    LHL = LHLFactorization
    function reduce_both(J::AbstractMatrix{T}, balance) where {T}
        n = size(J, 1)
        wb = LHLWorkspace{T}(n)
        copyto!(wb.factors, J)
        balance ? LHL._lhl_balance!(wb.factors, wb.scale) : fill!(wb.scale, 1)
        wu = deepcopy(wb)
        LHL._lhl_reduce_blocked!(wb.factors, wb.ipiv, wb.Ht, wb.resid, wb.Gt, LHL._lhl_panel_width(n))
        LHL._lhl_reduce_unblocked!(wu.factors, wu.ipiv)
        return wb, wu
    end
    for T in (Float64, ComplexF64), n in (160, 163, 200, 257), balance in (true, false)
        J = randn(MersenneTwister(n), T, n, n)
        wb, wu = reduce_both(J, balance)
        @test wb.ipiv == wu.ipiv
        @test wb.factors ≈ wu.factors
        @test all(abs.(tril(wb.factors, -2)) .<= 1)
        ws = lhl(J; balance)
        @test ws.ipiv == wb.ipiv
        @test ws.factors ≈ wb.factors
        b = randn(MersenneTwister(n + 1), T, n)
        for (σ, τ) in ((1.0, -0.05), (0.0, 1.0))
            lhl_shift!(ws, σ, τ)
            @test lhl_ldiv!(copy(b), ws) ≈ (σ * I + τ * J) \ b rtol = 1.0e-8
        end
    end
    # exact zeros and ties exercise the zero-pivot and column-interchange branches
    wb, wu = reduce_both(Float64.(rand(MersenneTwister(9), -1:1, 150, 150)), true)
    @test wb.ipiv == wu.ipiv
    @test wb.factors ≈ wu.factors
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

# The explicit-vector trailing update (Matrix{Float32/Float64}) must give the same
# reduction as the generic paired sweep, which a view of the same matrix dispatches to.
# Float32 results from the two differ by summation order alone, and at n ≈ 250 that is
# already ~n·eps in the factors, so each is compared to a Float64 reference instead: the
# explicit kernel must be as accurate as the generic one (within a factor 4 plus roundoff —
# a wrong loop bound, lane mask or dropped term shows up as O(1)).
wilkinson(n) = [i == j ? 1.0 : (j == n ? 1.0 : (i > j ? -1.0 : 0.0)) for i in 1:n, j in 1:n]
relerr(a, b) = norm(a - b) / norm(b)
function reduce_unblocked(J::AbstractMatrix{T}, generic::Bool) where {T}
    A = copy(J)
    ipiv = zeros(Int, max(size(A, 1) - 2, 0))
    LHLFactorization._lhl_reduce_unblocked!(generic ? view(A, :, :) : A, ipiv)
    return A, ipiv
end
@testset "explicit-vector trailing update agrees with the generic one: $T" for T in (Float64, Float32)
    for n in vcat(1:40, [64, 100, 127, 128, 129, 200, 257, 300])
        J = T.(randn(MersenneTwister(n), n, n))
        Ar, ipr = reduce_unblocked(Float64.(J), true)
        Ag, ipg = reduce_unblocked(J, true)
        Ae, ipe = reduce_unblocked(J, false)
        floor = 100n * eps(T)   # summation-order noise: O(n·eps) with a modest growth constant
        @test ipg == ipe == ipr
        @test relerr(Ae, Ar) <= 4 * relerr(Ag, Ar) + floor
        @test all(abs.(tril(Ae, -2)) .<= 1)
    end
    for J in (
            wilkinson(40), Matrix(1.0I, 30, 30), [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0],
            triu(randn(MersenneTwister(9), 50, 50), 1), Float64.(rand(MersenneTwister(9), -1:1, 150, 150)),
        )
        Ag, ipg = reduce_unblocked(T.(J), true)
        Ae, ipe = reduce_unblocked(T.(J), false)
        @test ipg == ipe
        @test Ag ≈ Ae
    end
end
