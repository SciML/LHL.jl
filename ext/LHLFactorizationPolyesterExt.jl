module LHLFactorizationPolyesterExt

using Polyester: @batch
using LHLFactorization
using LHLFactorization: LHLSerial, _LHL_BACKEND, _lhl_reduce_blocked!, _lhl_ht_fill!

# The chunk backend of the reduction; see the threading section of the package.
struct LHLPolyester end

function LHLFactorization._lhl_foreach_chunk!(f::F, ::LHLPolyester, nchunks::Int) where {F}
    @batch for t in 1:nchunks
        f(t)
    end
    return nothing
end

__init__() = (_LHL_BACKEND[] = LHLPolyester())

# Compile the chunked paths on the Polyester backend (blocked reduction on two chunks, both
# float types) into the extension's image while precompiling; the backend is set for the
# duration since `__init__` does not run then.  (Before Julia 1.12 a `@batch` compiled into
# an image allocates 112 bytes per call on more than one thread — Polyester's cfunction
# trampoline is re-created each time; not precompiling would cost ~6 s at the first `lhl`.)
if ccall(:jl_generating_output, Cint, ()) == 1
    let
        _LHL_BACKEND[] = LHLPolyester()
        for T in (Float64, Float32)
            n = 160
            J = T[1 / (i + j) + (i == j) for i in 1:n, j in 1:n]
            ws = lhl(J)
            lhl!(ws, J; thread = true)
            _lhl_reduce_blocked!(LHLPolyester(), ws.fstore, ws.ipiv, ws.Ht, ws.work, ws.pack, 16, 2)
            _lhl_ht_fill!(LHLPolyester(), ws.Ht, ws.fstore, n, 2)
        end
        _LHL_BACKEND[] = LHLSerial()
    end
end

end
