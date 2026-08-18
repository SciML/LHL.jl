using Documenter
using LHLFactorization

makedocs(
    modules = [LHLFactorization],
    sitename = "LHLFactorization.jl",
    format = Documenter.HTML(edit_link = "master"),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Release Notes" => "release_notes.md",
    ],
)

deploydocs(repo = "github.com/SciML/LHLFactorization.jl.git")
