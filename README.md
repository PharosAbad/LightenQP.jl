___LightenQP.jl___


[![Build Status](https://github.com/PharosAbad/LightenQP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/PharosAbad/LightenQP.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/PharosAbad/LightenQP.jl/wiki)

<h1 align="center" margin=0px>
  A pure Julia implementation of OOQP
</h1>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://github.com/PharosAbad/LightenQP.jl/wiki">Documentation</a>
</p>

**LightenQP.jl** solves the following convex quadratic programming problems (called `OOQP`):

$$
\begin{array}
[c]{cl}
\min & \frac{1}{2}\mathbf{z}^{\prime}\mathbf{Vz}+\mathbf{z}^{\prime}%
\mathbf{q}\\
s.t. & \mathbf{Az}=\mathbf{b}\in\mathbb{R}^{M}\\
& \mathbf{Cz}\leq\mathbf{g}\in\mathbb{R}^{L}
\end{array}
$$

with positive semi-definite symmetric matrix $\mathbf{V}\in\mathbb{R}^{N\times N}$. The general quadratic programming formulation solved by `LightenQP` is (`OOQP + d≤z≤u + h≤Cz`)

$$
\begin{array}
[c]{cl}
\min & \frac{1}{2}\mathbf{z}^{\prime}\mathbf{Vz}+\mathbf{z}^{\prime}
\mathbf{q}\\
s.t. & \mathbf{Az}=\mathbf{b}\in\mathbb{R}^{M}\\
& \mathbf{h}\leq\mathbf{Cz}\leq\mathbf{g}\in\mathbb{R}^{L}\\
& \boldsymbol{d}\leq\mathbf{z}\leq\boldsymbol{u}\in\mathbb{R}^{N}
\end{array}
$$

## Features

* __Light Weight__: 100+ lines Julia code. Which follows closely the the implementation of the C/C++ solver [OOQP](https://github.com/emgertz/OOQP)
* __Fast__: beat [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) for efficient portfolio seeking (when $N < 100$; for $N > 100$, roughly the same speed or a bit slower, maybe a *Rank-1 Update for LU*, or `Pardiso` will speed up)
* __Versatile__: solving a general quadratic programming problem mentioned above. $\mathbf{V}$ can be positive definite or positive semi-definite
* __Open Source__: Our code is available on [GitHub](https://github.com/PharosAbad/LightenQP.jl) and distributed under the MIT License
* __Arbitrary Precision Arithmetic__: fully support for `BigFloat`


## Installation
__LightenQP.jl__ can be added by

- `import Pkg; Pkg.add("LightenQP")`
- `pkg> add LightenQP`
- `pkg> dev LightenQP` for test nightly version. To use the registered version again `pkg> free LightenQP`

## License 🔍
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

