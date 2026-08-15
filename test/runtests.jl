using LHLFactorization, LinearAlgebra, Random, Test

bwd(W, x, b) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))

@testset "reduction reconstructs J" begin
    # 600 and 1030 take the blocked path by default; 1030 (and 128, 200) pad `fstore`
    for n in (1, 2, 3, 4, 17, 80, 128, 200, 301, 600, 1030), balance in (true, false)
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
    # 725 (Float64) and 1030 (both) put the packed multipliers above the 2 MiB tiling limit
    for T in (Float64, Float32), n in (3, 7, 33, 130, 331, 500, 725, 1030)
        J = randn(MersenneTwister(n), T, n, n)
        b = randn(MersenneTwister(n + 7), T, n)
        ws = lhl(J)
        lhl_shift!(ws, 1, -0.05)
        W = I - T(0.05) * J
        x = lhl_ldiv!(copy(b), ws)                    # Vector{T}: explicit kernels
        y = lhl_ldiv!(view(copy(b), :), ws)          # view, same eltype: buffered generic path
        z = lhl_ldiv!(complex.(b), ws)               # other eltype: in-place generic path
        for v in (x, y, z)
            # past n ≈ 500 the Float32 forward error (κ(W)·eps) exceeds 1e-3 for any solver
            # and the backward error carries κ(Z) past 50 eps; the three paths must still
            # agree with each other there
            (T == Float64 || n <= 500) && @test v ≈ W \ b rtol = (T == Float32 ? 1.0e-3 : 1.0e-9)
            @test bwd(W, v, b) < (n <= 500 ? 50 * eps(T) : 2 * bwd(W, x, b) + eps(T))
        end
        @test_throws DimensionMismatch lhl_ldiv!(zeros(T, n + 1), ws)
        @test_throws DimensionMismatch applyZ!(zeros(T, n - 1), ws)
    end
end

@testset "blocked reduction agrees with the unblocked one" begin
    LHL = LHLFactorization
    function reduce_both(J::AbstractMatrix{T}, balance) where {T}
        n = size(J, 1)
        wb = LHLWorkspace{T}(n)
        copyto!(wb.factors, J)
        balance ? LHL._lhl_balance!(wb.factors, wb.scale, wb.iscale) : fill!(wb.scale, 1)
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

# The shift as it was before the fused-row pass: one elimination step per pass over the
# pending row.  The fused kernel must reproduce it bit for bit — same pivots, same rounding —
# on random matrices and on ones full of exact ties and zero pivots.
function _lhl_shift_ref!(ws::LHLWorkspace{T}, σ, τ) where {T}
    Ht, Gt, swap, r, n = ws.Ht, ws.Gt, ws.swap, ws.resid, ws.n
    σ = convert(T, σ)
    τ = convert(T, τ)
    n == 0 && return ws
    @inbounds begin
        @simd for j in 1:n
            r[j] = τ * Ht[j, 1]
        end
        r[1] += σ
        info = 0
        for k in 1:(n - 1)
            a = r[k]
            b = τ * Ht[k, k + 1]
            if abs(b) > abs(a)
                swap[k] = true
                Gt[k, k] = b
                l = a / b
                Gt[k, k + 1] = l
                @simd for j in (k + 1):n
                    g = τ * Ht[j, k + 1]
                    Gt[j, k] = g
                    r[j] -= l * g
                end
                Gt[k + 1, k] += σ
                r[k + 1] -= l * σ
            else
                swap[k] = false
                Gt[k, k] = a
                if iszero(a)
                    info == 0 && (info = k)
                    l = zero(T)
                else
                    l = b / a
                end
                Gt[k, k + 1] = l
                @simd for j in (k + 1):n
                    rj = r[j]
                    Gt[j, k] = rj
                    r[j] = τ * Ht[j, k + 1] - l * rj
                end
                r[k + 1] += σ
            end
        end
        Gt[n, n] = r[n]
        swap[n] = false
        iszero(Gt[n, n]) && info == 0 && (info = n)
        rd = ws.rdiag
        @simd for j in 1:n
            rd[j] = inv(Gt[j, j])
        end
    end
    ws.info = info
    return ws
end
# Everything the solve reads: U (Gt[j, k], j ≥ k), the multipliers Gt[k, k+1], swap, rdiag, info.
function shift_state(ws)
    n = ws.n
    U = [ws.Gt[j, k] for k in 1:n for j in k:n]
    l = [ws.Gt[k, k + 1] for k in 1:(n - 1)]
    return (U, l, copy(ws.swap), copy(ws.rdiag), ws.info)
end
same_state(a, b) = all(x === y for (u, v) in zip(a, b) for (x, y) in zip(u, v))
tie_matrix(rng, T, n) = T.(rand(rng, -2:2, n, n))
@testset "shift matches the reference implementation: $T" for T in (Float64, Float32, ComplexF64)
    rng = MersenneTwister(31)
    ns = T <: Real ? vcat(1:300, [511, 512, 513, 520, 640, 768, 1000, 1024]) : 1:120
    for n in ns, tie in (false, true)
        J = tie ? tie_matrix(rng, T, n) : randn(rng, T, n, n)
        ws = lhl(J)
        wr = deepcopy(ws)
        for (σ, τ) in ((1, -0.1), (0, 1), (1, -3), (2, 0), (0, 0))
            _lhl_shift_ref!(wr, σ, τ)
            lhl_shift!(ws, σ, τ)
            @test same_state(shift_state(wr), shift_state(ws))
        end
    end
    # the fused kernels below their size threshold, all row counts, plus the tie/zero cases
    if T <: Real
        for n in 1:80, tie in (false, true), R in (1, 2, 3, 4)
            J = tie ? tie_matrix(rng, T, n) : randn(rng, T, n, n)
            ws = lhl(J)
            wr = deepcopy(ws)
            for (σ, τ) in ((1, -0.1), (0, 1), (2, 0))
                _lhl_shift_ref!(wr, σ, τ)
                ws.info = LHLFactorization._lhl_shift_fused!(Val(R), ws, T(σ), T(τ))
                @test same_state(shift_state(wr), shift_state(ws))
            end
        end
    end
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
    for (T, m) in ((Float64, 512), (Float64, 520), (Float32, 520), (Float32, 1024))   # fused shift
        wsm = lhl(randn(MersenneTwister(m), T, m, m))
        lhl_shift!(wsm, 1, -0.3)
        @test @allocated(lhl_shift!(wsm, 1, -0.4)) == 0
    end
end

# The explicit-vector trailing update (unit-stride Float32/Float64) must give the same
# reduction as the generic paired sweep, which a row-strided view dispatches to.
# Float32 results from the two differ by summation order alone, and at n ≈ 250 that is
# already ~n·eps in the factors, so each is compared to a Float64 reference instead: the
# explicit kernel must be as accurate as the generic one (within a factor 4 plus roundoff —
# a wrong loop bound, lane mask or dropped term shows up as O(1)).
wilkinson(n) = [i == j ? 1.0 : (j == n ? 1.0 : (i > j ? -1.0 : 0.0)) for i in 1:n, j in 1:n]
relerr(a, b) = norm(a - b) / norm(b)
function reduce_unblocked(J::AbstractMatrix{T}, generic::Bool) where {T}
    n = size(J, 1)
    ipiv = zeros(Int, max(n - 2, 0))
    generic || return LHLFactorization._lhl_reduce_unblocked!(copy(J), ipiv), ipiv
    P = zeros(T, 2n, n)
    P[1:2:(2n), :] .= J
    LHLFactorization._lhl_reduce_unblocked!(view(P, 1:2:(2n), 1:n), ipiv)
    return P[1:2:(2n), :], ipiv
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
