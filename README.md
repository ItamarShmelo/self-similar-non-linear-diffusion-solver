# Self Similar Non-Linear Diffusion Solver

Solves the self similar problem of a $w$ wave front coming from infinity, governed by the following non-linear diffusion equation

```math
\frac{\partial w}{\partial t} = x^{-a}\frac{\partial}{\partial x}\left(x^{a-b}w^n\frac{\partial w}{\partial x}\right)
```
