# Differentiating the backward household equation

[backward.py](backward.py) maps derivatives of next period's marginal value
and current inputs $(r,Z,Tr)$ into derivatives of current marginal value and
the endogenous cash-on-hand grid. The sections below follow `step_1st`,
`step_2nd`, and `step_3rd`.

## Nonlinear problem and setup

The arrays $V_a^+$ and $W_a$ have shape $S\times N$: rows index persistent
states $s$ with income $e_s$ and discount factor $\beta_s$, and columns index
assets $a_j$. With transition matrix $\Pi$ of shape $S\times S$, discounted
continuation marginal value is

$$
W_{a,s}(a_j)\equiv\beta_s\sum_{s'}\Pi_{ss'}V_{a,s'}^+(a_j),
\qquad W_a=\operatorname{diag}(\beta)\Pi V_a^+.
$$

The code stores `beta` as an $S\times1$ column, so `beta*Pi` scales each
current-state row of $\Pi$ by $\beta_s$.

Now fix $s$ and suppress its index. The endogenous cash-on-hand grid is
$m_j\equiv a_j+W_a(a_j)^{-\mathrm{eis}}$. At cash on hand $m$, unconstrained saving
$a^*(m)$ and marginal value of cash $v_m(m)$ satisfy

$$
u_c(m-a^*(m))=W_a(a^*(m)),\qquad v_m(m)=W_a(a^*(m)).
$$

With borrowing limit zero, $v_m(m)=u_c(m)$ in the constrained case.
The constrained and unconstrained expressions agree at $m^*\equiv m_0$.

For current assets $a_i$, cash on hand is $m=b+Z\eta$, where
$b\equiv Ra_i+Tr$, $R\equiv1+r$, and $\eta\equiv e_s\epsilon$. The backward
output averages over the iid shock $\log\epsilon\sim N(-\sigma_\xi^2/2,\sigma_\xi^2)$
conditional on state $s$:

$$
\bar V_b(b)\equiv\mathbb E[v_m(b+Z\eta)],\qquad
V_a(a_i)=R\bar V_b(b).
$$

Let $\theta$ index a shock experiment, with $\theta=0$ giving steady-state
continuation values and current inputs. All $d$ derivatives below are with
respect to $\theta$ at zero:
$V_a(\theta)=V_a+\theta\,dV_a+\theta^2d^2V_a/2+\cdots$.
Subscripts denote derivatives with respect to assets or cash. For example,
$dW_a(a)$ (`dWa`) holds assets fixed, and its asset derivative is $dW_{aa}$
(`dWaa`). Similarly, $da^*(m)$ (`da`) holds cash fixed, and its cash derivative
is $da_m^*$ (`da_m`); $d^2v_m$ is written `d2v_m` in code.

At the first endogenous grid point, saving and marginal-value responses use the
unconstrained expressions, evaluated at the borrowing threshold.

The arrays can be multiplied pointwise because **at the steady state**,
$a^*(m_j)=a_j$. Thus `mps*dWa` evaluates
$a_m^*(m)\,dW_a(a^*(m))$ at $m=m_j$.

`a_spline` and `m_spline` interpolate and differentiate using cubic
splines on the asset grid $a_j$ and endogenous grid $m_j$, respectively. These grids
stay fixed at their steady-state values; endogenous grid changes are returned
separately as `dm_endog`, `d2m_endog`, and `d3m_endog`.

## First-order step

With $\beta$ and $\Pi$ fixed, continuation value is linear in next period's
marginal value: $d^kW_a=\operatorname{diag}(\beta)\Pi d^kV_a^+$ at every order.
At first order, `step_1st` then applies
prepared linear maps, with $u\equiv(r,Z,Tr)$:

$$
dV_a=\frac{\partial V_a}{\partial W_a}dW_a
     +\frac{\partial V_a}{\partial u}du,\qquad
dm_j=m_{j,W_a}dW_a.
$$

Within each state, the two matrices have shapes $N\times N$ and $N\times3$.
They are `dVa_dWa` and `dVa_du`; `dm_dWa` multiplies pointwise.
Their construction is given under [First-order maps](#first-order-maps).
The second-order calculation below evaluates the ingredients explicitly.

## Second-order step

### Continuation value and the endogenous grid

The additional endogenous grid derivative is

$$
d^2m_j=m_{j,W_a}\,d^2W_a+m_{j,W_aW_a}(dW_a)^2.
$$

This is `d2m_endog`.

### Marginal value at fixed cash

Differentiating the saving condition with respect to cash gives the
marginal propensity to save, `mps`:
$a_m^*=u_{cc}/(u_{cc}+W_{aa})$. Numerically, the code uses `mps = 1-mpc`,
where `mpc` differentiates the steady-state consumption spline. This
approximates the same derivative without evaluating the ratio above.
Differentiating the saving condition with respect to $\theta$ at fixed cash gives

$$
da^*(m)=-\frac{a_m^*(m)}{u_{cc}(m-a^*(m))}dW_a(a^*(m)),\qquad
dv_m(m)=a_m^*(m)dW_a(a^*(m)).
$$

Differentiating the second identity again at fixed $m$ gives

$$
d^2v_m=a_m^*d^2W_a+da_m^*dW_a+a_m^*da^*dW_{aa}.
$$

This is `d2v_m`. `da_m` is obtained by differentiating `da` on the endogenous grid,
whereas `dWaa` differentiates `dWa` on the asset grid. In the constrained
case, these fixed-cash responses vanish: consumption is already pinned
to cash and does not depend on continuation value.

### Integrate while current cash changes

Suppressing $s$ and $i$, current inputs move cash by
$dm=db+\eta\,dZ$ and $d^2m=d^2b+\eta\,d^2Z$, with
$d^kb=a_i d^kr+d^kTr$.

First hold the split between constrained and unconstrained shock draws at
its steady-state value. The ordinary chain rule then gives

$$
\begin{aligned}
d\bar V_b&=\mathbb E[dv_m+v_{mm}\,dm],\\
(d^2\bar V_b)_{\mathrm{int}}
&=\mathbb E[d^2v_m+2dv_{mm}\,dm+v_{mmm}(dm)^2+v_{mm}\,d^2m].
\end{aligned}
$$

The subscript `int` holds the borrowing cutoff fixed.
The code expands $dm=db+\eta dZ$ and takes the resulting moments.
For example,
$\mathbb E[dv_{mm}\,dm]=db\,\mathbb E[dv_{mm}]+dZ\,\mathbb E[\eta\,dv_{mm}]$.

The code fits `q1` and `q2` to the grid values `dv_m` and `d2v_m`.
For a cash spline $f$ with coefficients `q`, the expectation maps act as follows;
the prefix `z` denotes the income weight $\eta$:

| Matrix acting on `q` | Integral over unconstrained draws |
| --- | --- |
| `maps.value` | $\mathbb E[f]$ |
| `maps.first`, `maps.z_first` | $\mathbb E[f_m]$, $\mathbb E[\eta f_m]$ |
| `maps.second`, `maps.z_second`, `maps.z2_second` | $\mathbb E[f_{mm}]$, $\mathbb E[\eta f_{mm}]$, $\mathbb E[\eta^2 f_{mm}]$ |

These are integrals against the original income density, without dividing by
the probability of being unconstrained. The maps are block diagonal in states;
flattening coefficients and reshaping outputs preserves the $S\times N$ layout.
Thus `maps.first @ q1` approximates $\mathbb E[dv_{mm}]$ by differentiating the
spline fitted to `dv_m`. The steady-state moments include **both** constrained
and unconstrained draws and are prepared once:
`Vbb, VbZ` are $\mathbb E[(1,\eta)v_{mm}]$, and
`Vbbb, VbbZ, VbZZ` are $\mathbb E[(1,\eta,\eta^2)v_{mmm}]$.

### Borrowing-limit correction

The remaining term comes from movement of the income threshold
$\eta^*\equiv(m^*-b)/Z$, where $m^*=m_0$. Households below it are constrained.
Write $\Delta$ for unconstrained minus constrained; continuity gives
$\Delta v_m(m^*)=0$.

To find how the threshold moves, first hold the income draw fixed.
Cash changes by $dm=db+\eta\,dZ$. At the steady-state threshold, the resulting
change in the marginal-value difference (`dDelta`) is

$$
d(\Delta v_m)=dv_m^U+\Delta v_{mm}(db+\eta^*dZ).
$$

The threshold moves to keep marginal values equal, so

$$
d(\Delta v_m)+Z\Delta v_{mm}d\eta^*=0,
\qquad d\eta^*=-\frac{d(\Delta v_m)}{Z\Delta v_{mm}}.
$$

Now differentiate the expectation, split at $\eta^*$. Continuity eliminates
the first-order boundary term. At second order, moving the threshold adds
$-\rho(\eta^*)d(\Delta v_m)d\eta^*$, where $\rho$ is the income density.
Substituting $d\eta^*$ and writing the cash density as
$p\equiv\rho(\eta^*)/Z$ gives

$$
d^2\bar V_b=(d^2\bar V_b)_{\mathrm{int}}
            +p\frac{[d(\Delta v_m)]^2}{\Delta v_{mm}}.
$$

For $\eta^*\leq0$, no income draws are constrained and the correction is zero.

### Marginal value of assets

The product $V_a=R\bar V_b$ gives the returned marginal values:
$dV_a=dr\,\bar V_b+R\,d\bar V_b$ and
$d^2V_a=d^2r\,\bar V_b+2dr\,d\bar V_b+R\,d^2\bar V_b$.

## Additional third-order calculations

### Continuation value and the endogenous grid

The endogenous grid derivative is
$d^3m_j=m_{j,W_a}d^3W_a+3m_{j,W_aW_a}dW_a\,d^2W_a+m_{j,W_aW_aW_a}(dW_a)^3$.

### Marginal value at fixed cash

The code first forms
$dv_{mm}=a_{mm}^*dW_a+(a_m^*)^2dW_{aa}$ (`dv_mm`), with `mps_m`
obtained by differentiating `mps` on the endogenous grid. It also differentiates
`d2v_m` on that grid to obtain `d2v_mm`.

Differentiating $v_m(m)=u_c(m-a^*(m))$ twice gives `d2a` below. Its cash derivative
then gives `d2a_m`, with $c_m=1-a_m^*$ (`mpc`):

$$
\begin{aligned}
d^2a^*&=\frac{u_{ccc}(da^*)^2-d^2v_m}{u_{cc}},\\
d^2a_m^*&=\frac{u_{cccc}c_m(da^*)^2+2u_{ccc}da^*da_m^*
                   -u_{ccc}c_m d^2a^*-d^2v_{mm}}{u_{cc}}.
\end{aligned}
$$

Differentiating the second-order expression for $d^2v_m$ gives

$$
\begin{aligned}
d^3v_m={}&d^2a_m^*dW_a+2da_m^*da^*dW_{aa}+2da_m^*d^2W_a\\
&+a_m^*d^2a^*dW_{aa}+a_m^*(da^*)^2dW_{aaa}
 +2a_m^*da^*d^2W_{aa}+a_m^*d^3W_a.
\end{aligned}
$$

The array `dv_mm` supplies the threshold response below. Interior expectations
instead differentiate the spline fitted to `dv_m`. Both approximate $dv_{mm}$,
but the product-rule and spline-derivative calculations need not agree exactly
on a finite grid.

### Integrate while current cash changes

Before the boundary correction, the third derivative is

$$
\begin{aligned}
(d^3\bar V_b)_{\mathrm{int}}=\mathbb E[{}
&d^3v_m+3d^2v_{mm}\,dm+3dv_{mmm}(dm)^2+3dv_{mm}\,d^2m\\
&+v_{mmmm}(dm)^3+3v_{mmm}\,dm\,d^2m+v_{mm}\,d^3m].
\end{aligned}
$$

Fitting `q3` to `d3v_m` and applying the same maps to `q1`, `q2`, and `q3`
gives the ten perturbation expectations. The steady-state moments also require
`barV_third_fixed`, equal to
$\mathbb E[(1,\eta,\eta^2,\eta^3)v_{mmmm}]$. Substituting
$d^km=d^kb+\eta d^kZ$ produces the expanded `d3barV_b` expression.

### Borrowing-limit correction

At the steady-state threshold, $\Delta v_{mmm}$ is stored as `Delta_v_mmm`.
Differentiating the marginal-value difference and its cash derivative
along $m=b+Z\eta$, with $\eta$ fixed, gives

$$
\begin{aligned}
d^2(\Delta v_m)&=d^2v_m^U+2dv_{mm}^U\,dm+\Delta v_{mmm}(dm)^2+\Delta v_{mm}d^2m,\\
d(\Delta v_{mm})&=dv_{mm}^U+\Delta v_{mmm}dm,
\end{aligned}
$$

where $dm=db+\eta^*dZ$ and the unconstrained derivatives are evaluated at $m^*$.
These are `d2Delta` and `dDelta_m`. Differentiating the split integral three
times gives the boundary contribution below. Terms containing $d^2\eta^*$
cancel by the crossing identity
$d(\Delta v_m)+Z\Delta v_{mm}d\eta^*=0$:

$$
\begin{aligned}
d^3\bar V_b-(d^3\bar V_b)_{\mathrm{int}}
={}&-3\rho\,d^2(\Delta v_m)d\eta^*
 -3\rho\,\partial_\eta d(\Delta v_m)(d\eta^*)^2\\
&-\rho\,\partial_\eta^2(\Delta v_m)(d\eta^*)^3
 -\rho'\,d(\Delta v_m)(d\eta^*)^2.
\end{aligned}
$$

Here $\rho$ and $\rho'$ are evaluated at $\eta^*$. Since $m=b+Z\eta$,

$$
\partial_\eta d(\Delta v_m)=Z\,d(\Delta v_{mm})+dZ\,\Delta v_{mm},
\qquad \partial_\eta^2(\Delta v_m)=Z^2\Delta v_{mmm}.
$$

Substituting these identities, $d\eta^*=-d(\Delta v_m)/(Z\Delta v_{mm})$,
and $\rho=Zp$, $\rho'=Z^2p_m$ gives the correction used in the code:

$$
\begin{aligned}
d^3\bar V_b=(d^3\bar V_b)_{\mathrm{int}}
&+3p\frac{d(\Delta v_m)\,d^2(\Delta v_m)}{\Delta v_{mm}}
\\
&-3p\frac{[d(\Delta v_m)]^2\bigl[d(\Delta v_{mm})+\Delta v_{mm}dZ/Z\bigr]}{(\Delta v_{mm})^2}\\
&+p\frac{[d(\Delta v_m)]^3\Delta v_{mmm}}{(\Delta v_{mm})^3}
 -p_m\frac{[d(\Delta v_m)]^3}{(\Delta v_{mm})^2}.
\end{aligned}
$$

The $dZ/Z$ term enters because the shock slope is $Z\Delta v_{mm}$;
$p_m$ is the derivative of the cash density with respect to cash.

### Marginal value of assets

Differentiating $V_a=R\bar V_b$ gives
$d^3V_a=d^3r\,\bar V_b+3d^2r\,d\bar V_b+3dr\,d^2\bar V_b+R\,d^3\bar V_b$.

## Steady-state preparation

`Backward.__init__` prepares utility derivatives and the expectation matrices
and moments described above. `cash_expectation_matrices` builds the operators
without a supplied function. Separately, `marginal_value_moments` takes consumption
coefficients `q_c` and differentiates $c(m)^{-\gamma}$, with
$\gamma\equiv1/\mathrm{eis}$, to evaluate the steady-state moments.

### Borrowing-threshold derivatives

At $m^*$, $c=m^*$ and constrained consumption has $c_m=1$, $c_{mm}=0$.
The unconstrained derivatives therefore give

$$
\Delta v_{mm}=u_{cc}(c_m-1),\qquad
\Delta v_{mmm}=u_{ccc}(c_m^2-1)+u_{cc}c_{mm}.
$$

`m_spline.derivatives_at_lower_bound(q_c)` evaluates $c_m,c_{mm}$ from
the consumption coefficients. The cash density at $m^*$ is the density of
$Z\eta$ at $m^*-b$; its log mean is `mu + log(Z)`.

### First-order maps

Within each state, let $C$ map grid values to spline coefficients and $M$
integrate those splines over unconstrained income draws. Since $dv_m=mps\,dW_a$,

$$
\frac{\partial V_a}{\partial W_a}=RMC\operatorname{diag}(mps).
$$

The final multiplication by `mps` in code scales the columns. The input
columns follow from $V_a=R\bar V_b$ and $b=Ra_i+Tr$:

$$
\frac{\partial V_a}{\partial r}=\bar V_b+Ra_i\mathbb E[v_{mm}],\qquad
\frac{\partial V_a}{\partial Z}=R\mathbb E[\eta v_{mm}],\qquad
\frac{\partial V_a}{\partial Tr}=R\mathbb E[v_{mm}].
$$

## Approximation and paths

The identities describe differentiation of the household equation.
Spline interpolation and lognormal quadrature approximate their ingredients.
Quadrature is split across spline pieces, where higher derivatives are smooth,
and truncates log income at eight standard deviations from its mean.
The final cash-spline piece is extrapolated beyond the endogenous grid.
The implementation assumes a zero borrowing limit and a nonzero
slope jump $\Delta v_{mm}$ at the crossing.

In [paths.py](../sequence_space/paths.py), backward time sweeps repeatedly apply the
one-period equations using the next date's marginal-value derivatives.
