# Use
#
#     DOCUMENTER_DEBUG=true julia --color=yes make.jl local [nonstrict] [fixdoctests]
#
# for local builds.

using Documenter
using AutoDiffOperators

# Doctest setup
DocMeta.setdocmeta!(
    AutoDiffOperators,
    :DocTestSetup,
    :(using AutoDiffOperators);
    recursive=true,
)

makedocs(
    sitename = "AutoDiffOperators",
    modules = [AutoDiffOperators],
    format = Documenter.HTML(
        prettyurls = !("local" in ARGS),
        canonical = "https://oschulz.github.io/AutoDiffOperators.jl/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "LICENSE" => "LICENSE.md",
    ],
    doctest = ("fixdoctests" in ARGS) ? :fix : true,
    linkcheck = !("nonstrict" in ARGS),
    strict = !("nonstrict" in ARGS),
)

deploydocs(
    repo = "github.com/oschulz/AutoDiffOperators.jl.git",
    forcepush = true,
    push_preview = true,
)
