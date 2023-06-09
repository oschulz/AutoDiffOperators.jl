# AutoDiffOperators.jl

[![Documentation for stable version](https://img.shields.io/badge/docs-stable-blue.svg)](https://oschulz.github.io/AutoDiffOperators.jl/stable)
[![Documentation for development version](https://img.shields.io/badge/docs-dev-blue.svg)](https://oschulz.github.io/AutoDiffOperators.jl/dev)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://github.com/oschulz/AutoDiffOperators.jl/workflows/CI/badge.svg?branch=main)](https://github.com/oschulz/AutoDiffOperators.jl/actions?query=workflow%3ACI)
[![Codecov](https://codecov.io/gh/oschulz/AutoDiffOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/oschulz/AutoDiffOperators.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


## Documentation

* [Documentation for stable version](https://oschulz.github.io/AutoDiffOperators.jl/stable)
* [Documentation for development version](https://oschulz.github.io/AutoDiffOperators.jl/dev)

This package provides multiplicative operators that act via automatic
differentiation (AD).

AutoDiffOperators.jl uses AD-backend abstractions and supports a subset of the AD-backends
specifiers in both
[AbstractDifferentiation.jl](https://github.com/JuliaDiff/AbstractDifferentiation.jl)
and [ADTypes.jl](https://github.com/SciML/ADTypes.jl). Support for additional
AD-backends is planned.
