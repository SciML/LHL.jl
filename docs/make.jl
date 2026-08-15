using Documenter
using LHLFactorization

makedocs(
    modules = [LHLFactorization],
    sitename = "LHLFactorization.jl",
    format = Documenter.HTML(edit_link = "master"),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
)

deploydocs(repo = "github.com/SciML/LHLFactorization.jl.git")
