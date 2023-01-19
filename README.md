___OOQP.jl___


[![Build Status](https://github.com/PharosAbad/OOQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PharosAbad/OOQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/PharosAbad/OOQP.jl/wiki)

<h1 align="center" margin=0px>
  A pure Julia implementation of OOQP
</h1>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://github.com/PharosAbad/OOQP.jl/wiki">Documentation</a>
</p>

**OOQP.jl** solves the following convex quadratic programming (QP) problems:

$$
\begin{array}
[c]{cl}
\min & \frac{1}{2}\mathbf{z}^{\prime}\mathbf{Vz}+\mathbf{z}^{\prime}%
\mathbf{q}\\
s.t. & \mathbf{Az}=\mathbf{b}\in\mathbb{R}^{M}\\
& \mathbf{Cz}\leq\mathbf{g}\in\mathbb{R}^{L}
\end{array}
$$

with positive semi-definite symmetric matrix $\mathbf{V}\in\mathbb{R}^{N\times N}$.

## Features

* __Light weight__: 100+ lines Julia code. Which follows closely the the implementation of the C/C++ solver [OOQP](https://github.com/emgertz/OOQP)
* __Fast__: beat [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) for efficient portfolio seeking
* __Open Source__: Our code is available on [GitHub](https://github.com/PharosAbad/OOQP.jl) and distributed under the MIT License
* __Arbitrary Precision Arithmetic__: fully support for `BigFloat`


## Installation
__OOQP.jl__ can be added by

- `import Pkg; Pkg.add("OOQP")`
- `pkg> add OOQP`
- `pkg> dev OOQP` for test nightly version. To use the registered version again `pkg> free OOQP`

## License 🔍
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

