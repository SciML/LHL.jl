using LHLFactorization, LinearAlgebra, Random, Test

bwd(W, x, b) = norm(b - W * x, Inf) / (opnorm(W, Inf) * norm(x, Inf) + norm(b, Inf))

# The threaded reduction partitions its work independently of the thread count, so the
# threaded and serial paths must agree to the bit.  Without Polyester loaded, or with one
# thread, the two paths coincide and this only checks the API (CI runs with four threads).
function threading_tests()
    LHL = LHLFactorization
    nthr = Threads.nthreads()
    poly = !(LHL._LHL_BACKEND[] isa LHL.LHLSerial)
    poly || @info "Polyester not loaded: `thread = Val(true)` runs the serial path"
    nthr == 1 && @info "Threads.nthreads() == 1: the threaded reduction runs on the serial path"
    threaded = poly && nthr > 1
    @testset "threaded reduction is bit-identical to the serial one (Polyester $(poly ? "loaded" : "absent"))" begin
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
            # Before Julia 1.12, `@batch` code compiled into the extension's image pays 112
            # bytes per region when run on more than one thread (Polyester's cfunction
            # trampoline is re-created per call there); 1.12 is allocation-free.
            if VERSION >= v"1.12" || !threaded
                @test alloc_lhl(w2, J, balance) == 0
            else
                @test alloc_lhl(w2, J, balance) <= 112 * 3n
            end
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
            LHL._lhl_reduce_blocked!(LHL._LHL_BACKEND[], w1.factors, w1.ipiv, w1.Ht, w1.work, w1.pack, 16, 1)
            LHL._lhl_reduce_blocked!(LHL._LHL_BACKEND[], w2.factors, w2.ipiv, w2.Ht, w2.work, w2.pack, 16, max(nthr, 3))
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
    return nothing
end

# The threading testset runs twice: without Polyester (`thread = Val(true)` must then be
# the serial path) and, after `using Polyester`, with the extension loaded.
threading_tests()
using Polyester
threading_tests()
