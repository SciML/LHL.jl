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
        LHL._lhl_reduce_blocked!(wb.factors, wb.ipiv, wb.Ht, wb.work, wb.pack, LHL._lhl_panel_width(n))
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
# pending row, on a plain n×n `Gt`.  The fused kernel and the planar complex storage must
# reproduce it bit for bit — same pivots, same rounding — on random matrices and on ones full
# of exact ties and zero pivots.
function _lhl_shift_ref!(Gt::Matrix{T}, swap, rdiag, Ht, σ, τ) where {T}
    n = size(Gt, 1)
    r = Vector{T}(undef, n)
    σ = convert(T, σ)
    τ = convert(T, τ)
    @inbounds begin
        @simd for j in 1:n
            r[j] = τ * Ht[j, 1]
        end
        r[1] += σ
        info = 0
        for k in 1:(n - 1)
            a = r[k]
            b = τ * Ht[k, k + 1]
            if LHLFactorization._lhl_pivmag(b) > LHLFactorization._lhl_pivmag(a)
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
        @simd for j in 1:n
            rdiag[j] = inv(Gt[j, j])
        end
    end
    return info
end
# Everything the solve reads: U (Gt[j, k], j ≥ k), the multipliers Gt[k, k+1], swap, rdiag, info.
function ref_state(Ht::AbstractMatrix, ::Type{T}, σ, τ) where {T}
    n = size(Ht, 1)
    Gt = zeros(T, n, n)
    swap = Vector{Bool}(undef, n)
    rdiag = Vector{T}(undef, n)
    info = _lhl_shift_ref!(Gt, swap, rdiag, Ht, σ, τ)
    U = [Gt[j, k] for k in 1:n for j in k:n]
    l = [Gt[k, k + 1] for k in 1:(n - 1)]
    return (U, l, swap, rdiag, info)
end
function shift_state(sh::LHLShift{TG}) where {TG}
    n = sh.n
    gt(j, k) = LHLFactorization._lhl_get(TG, sh.Gt, sh.po, j, k)
    U = [gt(j, k) for k in 1:n for j in k:n]
    l = [gt(k, k + 1) for k in 1:(n - 1)]
    return (U, l, copy(sh.swap), copy(sh.rdiag), sh.info)
end
same_state(a, b) = all(x === y for (u, v) in zip(a, b) for (x, y) in zip(u, v))
tie_matrix(rng, T, n) = T.(rand(rng, -2:2, n, n))
@testset "shift matches the reference implementation: $T" for T in (Float64, Float32, ComplexF64)
    rng = MersenneTwister(31)
    ns = T <: Real ? vcat(1:300, [511, 512, 513, 520, 640, 768, 1000, 1024]) : vcat(1:120, [511, 512, 520])
    for n in ns, tie in (false, true)
        J = tie ? tie_matrix(rng, T, n) : randn(rng, T, n, n)
        ws = lhl(J)
        for (σ, τ) in ((1, -0.1), (0, 1), (1, -3), (2, 0), (0, 0))
            lhl_shift!(ws, σ, τ)
            @test same_state(ref_state(ws.Ht, T, σ, τ), shift_state(ws.shift))
        end
    end
    # the fused kernels below their size threshold, all row counts, plus the tie/zero cases
    for n in 1:80, tie in (false, true), R in (1, 2, 3, 4)
        J = tie ? tie_matrix(rng, T, n) : randn(rng, T, n, n)
        ws = lhl(J)
        for (σ, τ) in ((1, -0.1), (0, 1), (2, 0))
            ws.info = LHLFactorization._lhl_shift_fused!(Val(R), ws.shift, ws, T(σ), T(τ))
            @test same_state(ref_state(ws.Ht, T, σ, τ), shift_state(ws.shift))
        end
    end
end

# A complex shift on a real reduction: `Gt` is planar, `Ht` real.  Against the reference
# on the same real `Ht` promoted to complex, bit for bit.
@testset "complex shift on a real reduction matches the reference" begin
    rng = MersenneTwister(32)
    for n in vcat(1:100, [255, 256, 511, 512, 520, 700]), tie in (false, true)
        J = tie ? tie_matrix(rng, Float64, n) : randn(rng, n, n)
        ws = lhl(J; shift = ComplexF64)
        for (σ, τ) in ((0.4 + 0.7im, -1), (1, -0.1 + 0.2im), (0, 1im), (2im, 0), (0, 0))
            lhl_shift!(ws, σ, τ)
            @test same_state(ref_state(ws.Ht, ComplexF64, σ, τ), shift_state(ws.shift))
        end
        if n <= 80
            for R in (1, 2, 3, 4)
                σ, τ = 0.4 + 0.7im, -1.0 + 0im
                ws.info = LHLFactorization._lhl_shift_fused!(Val(R), ws.shift, ws, σ, τ)
                @test same_state(ref_state(ws.Ht, ComplexF64, σ, τ), shift_state(ws.shift))
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

@testset "complex shifts on a real reduction" begin
    setprecision(BigFloat, 128) do
        for n in (1, 2, 3, 5, 40, 129, 260, 600)
            J = randn(MersenneTwister(n), n, n)
            b = randn(MersenneTwister(n + 7), ComplexF64, n)
            ws = lhl(J; shift = ComplexF64)
            @test ws isa LHLWorkspace{Float64, Float64, ComplexF64}
            wf = lhl(complex.(J))
            for (σ, τ) in ((0.4 + 0.7im, -1.0), (1.0, -0.05 + 0.3im), (0.0, 1im), (-3.0 + 2im, 2.0 - 1im), (1.0, -0.05))
                W = σ * I + τ * J
                lhl_shift!(ws, σ, τ)
                x = lhl_ldiv!(copy(b), ws)
                @test x ≈ Matrix(W) \ b rtol = 1.0e-9
                lhl_shift!(wf, σ, τ)
                xf = lhl_ldiv!(copy(b), wf)
                @test x ≈ xf rtol = 1.0e-9
                # forward errors of the real and the complex reduction are comparable (measured
                # 0.5–3× of each other on these matrices)
                if n <= 129
                    xref = (big(σ) * I + big(τ) * big.(J)) \ big.(b)
                    err(v) = Float64(norm(v - xref) / norm(xref))
                    @test err(x) <= 10 * err(xf) + 100 * eps()
                end
                # refinement with the complex system matrix
                y = lhl_refine!(copy(x), Matrix(W), b, ws, 1)
                @test bwd(Matrix(W), y, b) <= 2 * bwd(Matrix(W), x, b) + eps()
                @test bwd(Matrix(W), y, b) < 50 * eps()
            end
        end
    end
    # a resolvent sweep, (sI - A)⁻¹b over a set of s
    n = 80
    A = randn(MersenneTwister(80), n, n)
    b = randn(MersenneTwister(81), ComplexF64, n)
    ws = lhl(A; shift = ComplexF64)
    for s in (0.1 + 1im, 2.0 - 3im, 1im, -0.5 + 0.01im)
        lhl_shift!(ws, s, -1)
        @test lhl_ldiv!(copy(b), ws) ≈ (s * I - A) \ b rtol = 1.0e-9
    end
    # Radau-like use: a real and a complex shift held together against one reduction
    J = randn(MersenneTwister(90), n, n)
    br = randn(MersenneTwister(91), n)
    bc = randn(MersenneTwister(92), ComplexF64, n)
    ws = lhl(J)
    sh = LHLShift{ComplexF64}(ws)
    @test sh isa LHLShift{ComplexF64, Float64}
    for (γ, γc) in ((0.1, 0.05 + 0.1im), (0.2, 0.1 - 0.3im))
        lhl_shift!(ws, 1, -γ)
        lhl_shift!(sh, ws, 1, -γc)
        @test lhl_ldiv!(copy(br), ws) ≈ (I - γ * J) \ br rtol = 1.0e-9
        @test lhl_ldiv!(copy(bc), sh, ws) ≈ (I - γc * J) \ bc rtol = 1.0e-9
        @test lhl_ldiv!(copy(bc), ws) ≈ (I - γ * J) \ bc rtol = 1.0e-9    # complex rhs, real shift
        @test lhl_refine!(lhl_ldiv!(copy(bc), sh, ws), I - γc * J, bc, sh, ws, 1) ≈ (I - γc * J) \ bc rtol = 1.0e-12
        @test ws.info == 0 && sh.info == 0 && sh.σ == 1 && sh.τ == -γc
    end
    # the shift resizes with the workspace
    J2 = randn(MersenneTwister(93), 2n, 2n)
    b2 = randn(MersenneTwister(94), ComplexF64, 2n)
    lhl!(ws, J2)
    lhl_shift!(sh, ws, 1, -0.3im)
    @test lhl_ldiv!(copy(b2), sh, ws) ≈ (I - 0.3im * J2) \ b2 rtol = 1.0e-9
    # a real x cannot receive a complex solution; a real shift state cannot take a complex shift
    @test_throws ArgumentError lhl_ldiv!(randn(2n), sh, ws)
    @test_throws ArgumentError lhl_shift!(ws, 1, -0.3im)
    @test_throws ArgumentError lhl_shift!(LHLShift{ComplexF64}(3), lhl(randn(Float32, 3, 3)), 1, 1)
    @test_throws ArgumentError LHLWorkspace{Float64}(3; shift = Float32)
    @test_throws DimensionMismatch lhl_ldiv!(copy(b2), LHLShift{ComplexF64}(n), ws)
    lhl_shift!(ws, 1 + 0im, -0.3 + 0im)         # complex with zero imaginary parts is fine
    @test lhl_ldiv!(copy(b2), ws) ≈ (I - 0.3 * J2) \ b2 rtol = 1.0e-9
    # singular complex shift is reported
    wd = lhl([1.0 0.0; 0.0 2.0]; shift = ComplexF64)
    lhl_shift!(wd, -1im, 1im)                   # i(J - I)
    @test wd.info != 0
    # Float32 and a generic element type
    J32 = randn(MersenneTwister(95), Float32, 70, 70)
    b32 = randn(MersenneTwister(96), ComplexF32, 70)
    w32 = lhl(J32; shift = ComplexF32)
    lhl_shift!(w32, 0.4f0 + 0.7f0im, -1)
    @test lhl_ldiv!(copy(b32), w32) ≈ ((0.4f0 + 0.7f0im) * I - J32) \ b32 rtol = 1.0e-3
    Jb = big.(randn(MersenneTwister(97), 20, 20))
    bb = Complex{BigFloat}.(randn(MersenneTwister(98), ComplexF64, 20))
    wb = lhl(Jb; shift = Complex{BigFloat})
    lhl_shift!(wb, 0.4 + 0.7im, -1)
    @test lhl_ldiv!(copy(bb), wb) ≈ ((0.4 + 0.7im) * I - Jb) \ bb rtol = 1.0e-9
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
    for m in (64, 520)   # complex shift on a real reduction, built in and held separately
        Jm = randn(MersenneTwister(m), m, m)
        wc = lhl(Jm; shift = ComplexF64)
        sh = LHLShift{ComplexF64}(wc)
        Wc = (0.4 + 0.7im) * I - Jm
        xc = randn(MersenneTwister(m + 1), ComplexF64, m)
        bc = copy(xc)
        lhl_shift!(wc, 0.4 + 0.7im, -1)
        lhl_shift!(sh, wc, 0.4 + 0.7im, -1)
        lhl_ldiv!(xc, wc)
        lhl_ldiv!(xc, sh, wc)
        lhl_refine!(xc, Wc, bc, wc, 1)
        lhl_refine!(xc, Wc, bc, sh, wc, 1)
        sc = randn(MersenneTwister(m + 2), ComplexF64)   # a runtime value, not a constant-folded literal
        @test @allocated(lhl_shift!(wc, sc, -1)) == 0
        @test @allocated(lhl_shift!(sh, wc, sc, -1)) == 0
        @test @allocated(lhl_shift!(wc, sc, 1 - sc)) == 0
        @test @allocated(lhl_ldiv!(xc, wc)) == 0
        @test @allocated(lhl_ldiv!(xc, sh, wc)) == 0
        @test @allocated(lhl_refine!(xc, Wc, bc, wc, 1)) == 0
        @test @allocated(lhl_refine!(xc, Wc, bc, sh, wc, 1)) == 0
        @test @allocated(lhl!(wc, Jm)) == 0
    end
end

# The threaded reduction partitions its work independently of the thread count, so the
# threaded and serial paths must agree to the bit; with one thread the two paths coincide
# and this only checks the API (the CI workflow runs a single thread).
@testset "threaded reduction is bit-identical to the serial one" begin
    LHL = LHLFactorization
    nthr = Threads.nthreads()
    nthr == 1 && @info "Threads.nthreads() == 1: the threaded reduction runs on the serial path"
    alloc_lhl(w, J, balance) = @allocated lhl!(w, J; balance, thread = Val(true))
    function both(J::AbstractMatrix{T}, balance) where {T}
        n = size(J, 1)
        w1 = LHLWorkspace{T}(n)
        w2 = LHLWorkspace{T}(n)
        # unwritten slots of Lp/Ht are undef; zero them so that == compares only what the
        # reduction writes
        for w in (w1, w2)
            fill!(w.Lp, 0)
            fill!(w.Ht, 0)
        end
        lhl!(w1, J; balance, thread = Val(false))
        lhl!(w2, J; balance, thread = Val(true))
        return w1, w2
    end
    for (T, n) in ((Float64, 520), (Float64, 777), (Float32, 1030)), balance in (true, false)
        J = randn(MersenneTwister(n), T, n, n)
        w1, w2 = both(J, balance)
        @test w1.ipiv == w2.ipiv
        @test w1.factors == w2.factors
        @test w1.Lp == w2.Lp
        @test w1.Ht == w2.Ht
        b = randn(MersenneTwister(n + 1), T, n)
        for (σ, τ) in ((1, -0.05), (0, 1))
            lhl_shift!(w1, σ, τ)
            lhl_shift!(w2, σ, τ)
            x1 = lhl_ldiv!(copy(b), w1)
            x2 = lhl_ldiv!(copy(b), w2)
            @test x1 == x2
            @test bwd(σ * I + τ * J, x2, b) <= 20n * eps(T)
        end
        @test alloc_lhl(w2, J, balance) == 0
    end
    # small sizes and the zero-pivot / interchange branches on the blocked kernel directly,
    # with the thread count forced (chunks then hold few rows each)
    for J in (
            randn(MersenneTwister(1), 160, 160), randn(MersenneTwister(2), Float32, 300, 300),
            Float64.(rand(MersenneTwister(9), -1:1, 200, 200)),
        )
        T = eltype(J)
        n = size(J, 1)
        w1 = LHLWorkspace{T}(n)
        w2 = LHLWorkspace{T}(n)
        copyto!(w1.factors, J)
        copyto!(w2.factors, J)
        LHL._lhl_reduce_blocked!(w1.factors, w1.ipiv, w1.Ht, w1.work, w1.pack, 16, 1)
        LHL._lhl_reduce_blocked!(w2.factors, w2.ipiv, w2.Ht, w2.work, w2.pack, 16, max(nthr, 3))
        @test w1.ipiv == w2.ipiv
        @test w1.factors == w2.factors
        # against the unblocked reduction, in Float64 (the two Float32 paths differ by
        # summation order alone, ~n·eps in the factors)
        wu = LHLWorkspace{Float64}(n)
        copyto!(wu.factors, J)
        LHL._lhl_reduce_unblocked!(wu.factors, wu.ipiv)
        @test wu.ipiv == w2.ipiv
        @test norm(w2.factors - wu.factors) <= 100n * eps(T) * norm(wu.factors)
    end
    # a Bool works too, and other element types stay serial
    J = randn(MersenneTwister(5), ComplexF64, 800, 800)
    w1 = lhl(J; thread = false)
    w2 = lhl(J; thread = true)
    @test w1.factors == w2.factors
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
