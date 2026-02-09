
# MIT 6.7220 / 15.084 — Nonlinear Optimization  
**Lecture 16: Bayesian Optimization**  
**Thu, Apr 25, 2024**

**Instructor:** Prof. Gabriele Farina (gfarina@mit.edu)

---

## Overview

Bayesian optimization is useful when evaluating a function is expensive and gradient information is unavailable. It is especially popular for hyperparameter tuning and architecture search.

---

## 1. A Bayesian Approach to Optimization

We assume access only to function evaluations:

$$
x \mapsto f(x)
$$

Rather than directly minimizing $f$, we model $f$ probabilistically and decide where to query next based on uncertainty and expected improvement.

### Expected Improvement

$$
\text{EI}(x) = \mathbb{E}_{\tilde f}\left[\max\{0, f^\star - \tilde f(x)\}\right]
$$

---

## 2. Gaussian Processes

A Gaussian process (GP) assumes that for any finite set of points  
$\{x_1, \dots, x_t\} \subset \mathbb{R}^n$, the vector

$$
(f(x_1), \dots, f(x_t))
$$

is multivariate Gaussian.

A GP is fully specified by:
- a mean function $m(x)$
- a covariance (kernel) function $K(x,y)$

---

### 2.1 Radial Basis Function Kernel

$$
K(x,y) = k \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)
$$

where:
- $k > 0$ controls amplitude
- $\sigma > 0$ controls smoothness

---

## 2.3 Conditioning a Gaussian Process

Given observed data $X = (x_1, \dots, x_t)$:

$$
\mu(x) = m(x) + K(x,X)^\top K(X,X)^{-1}(f(X)-m(X))
$$

$$
\sigma^2(x) = K(x,x) - K(x,X)^\top K(X,X)^{-1}K(x,X)
$$

For observed points $x_i$:

$$
\mu(x_i) = f(x_i), \qquad \sigma^2(x_i)=0
$$

---

## 3. Acquisition Functions

### Expected Improvement (EI)

$$
\text{EI}(x) =
\frac{1}{\sigma(x)\sqrt{2\pi}}
\int_{-\infty}^{f^\star}
(f^\star - v)
\exp\left(
-\frac12\left(\frac{v-\mu(x)}{\sigma(x)}\right)^2
\right) dv
$$

### Probability of Improvement (PI)

$$
\text{PI}(x) = \mathbb{P}[f(x) \le f^\star]
$$

### Lower Confidence Bound (LCB)

$$
\text{LCB}(x) = \mu(x) - \beta \sigma(x), \quad \beta \ge 0
$$

---

## 4. Simulation Example

The function used is:

$$
f(x) = \frac{1}{3} x \cos(x)
$$

Bayesian optimization iteratively refines its estimate by balancing exploration and exploitation.

---

## Bibliography

- C. A. Micchelli, Y. Xu, and H. Zhang,  
  *Universal Kernels*, Journal of Machine Learning Research, 2006.
