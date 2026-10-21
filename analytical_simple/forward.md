# Differentiating the forward distribution equation

[forward.py](forward.py) propagates derivatives of the inherited asset CDF
through the saving policy and persistent-state transition. The sections
below follow `step_1st`, `step_2nd`, and `step_3rd`.

## Nonlinear problem and setup

The inherited CDF $F$ and post-saving CDF $\widetilde F$ have shape
$S\times N$, with a conditional CDF in each state row and assets in columns.
The state transition gives $F^+=\Pi_F\widetilde F$, where
$(\Pi_F)_{s's}\equiv\pi_s\Pi_{ss'}/\pi_{s'}$ and $\pi$ is the stationary state
distribution. State population shares remain at $\pi$ throughout the experiment;
perturbations change conditional asset distributions, with $\Pi$ and $\pi$ fixed.

Now fix $s$ and suppress its index. For target assets $a'_j$, the backward
step supplies the required cash on hand $m_j$ (`m_endog`). Given income
$Ze_s\epsilon$, a household saves at most $a'_j$ when its inherited assets
are at most

$$
a(\epsilon)\equiv\frac{m_j-Tr-Ze_s\epsilon}{R},\qquad R\equiv1+r.
$$

Thus $\widetilde F(a'_j)=\mathbb E_\epsilon[F(a(\epsilon))]$, with
$\log\epsilon\sim N(-\sigma_\xi^2/2,\sigma_\xi^2)$.

As in the backward note, $d$ denotes derivatives with respect to the shock
shifter $\theta$ at the steady state. For example,
$dF(a)$ holds assets fixed, $dF_a$ is its asset derivative, and $da(\epsilon)$
holds the income draw fixed. The endogenous grid change $dm_j$ is `dm_endog` in code.

## First-order step

With $u\equiv(r,Z,Tr)$, `step_1st` applies prepared linear maps within each
state, then mixes states:

$$
d\widetilde F=\frac{\partial\widetilde F}{\partial F}dF
 +\frac{\partial\widetilde F}{\partial m_j}dm_j
 +\frac{\partial\widetilde F}{\partial u}du,\qquad
dF^+=\Pi_F d\widetilde F.
$$

The three maps are `dtildeF_dF`, `dtildeF_dm`, and `dtildeF_du`;
the endogenous grid term multiplies pointwise. Their construction, including the
borrowing-limit terms, is given under [First-order maps](#first-order-maps).

## Second-order step

### Asset-cutoff derivatives

Holding the income draw fixed and differentiating $Ra=m_j-Tr-Ze_s\epsilon$
gives

$$
\begin{aligned}
R\,da&=dm_j-dTr-e_s\epsilon\,dZ-a\,dr,\\
R\,d^2a&=d^2m_j-d^2Tr-e_s\epsilon\,d^2Z-a\,d^2r-2dr\,da.
\end{aligned}
$$

At the steady state, $e_s\epsilon=(m_j-Tr-Ra)/Z$. Substituting this expression
makes both derivatives affine functions of $a$:
$da=\alpha_1 a+\beta_1$ and $d^2a=\alpha_2 a+\beta_2$, where

$$
\begin{aligned}
\alpha_1&\equiv\frac{dZ}{Z}-\frac{dr}{R},&
\beta_1&\equiv\frac{dm_j-dTr-(m_j-Tr)dZ/Z}{R},\\
\alpha_2&\equiv\frac{d^2Z}{Z}-\frac{d^2r}{R}-\frac{2dr\,\alpha_1}{R},&
\beta_2&\equiv\frac{d^2m_j-d^2Tr-(m_j-Tr)d^2Z/Z-2dr\,\beta_1}{R}.
\end{aligned}
$$

This form expresses the derivative integrals as expectations weighted by
powers of assets.

### CDF expectation

The interior expectations below integrate against the original shock density
over $a(\epsilon)>a_{\min}$, without renormalizing its probability. Numerically,
the operators retain only $a_{\min}\leq a(\epsilon)\leq a_{\max}$: the upper
CDF tail is treated as flat, with no upper-endpoint movement corrections.
Holding the shock limits fixed, the chain rule gives

$$
\begin{aligned}
(d\widetilde F)_{\mathrm{int}}
 &=\mathbb E[dF+F_a\,da],\\
(d^2\widetilde F)_{\mathrm{int}}
 &=\mathbb E[d^2F+2dF_a\,da+F_{aa}(da)^2+F_a\,d^2a].
\end{aligned}
$$

These are the initial `first` and `second` arrays. For example,

$$
\mathbb E[F_{aa}(da)^2]
=\alpha_1^2\mathbb E[a^2F_{aa}]
 +2\alpha_1\beta_1\mathbb E[aF_{aa}]
 +\beta_1^2\mathbb E[F_{aa}].
$$

The code fits `q1` and `q2` to `dF` and `d2F` on the fixed asset grid.
For an asset spline $f$ with coefficients `q`, the maps act as follows;
the prefix `z` denotes the asset weight $a$:

| Matrix acting on `q` | Integral over the asset-grid region |
| --- | --- |
| `M.value` | $\mathbb E[f]$ |
| `M.first`, `M.z_first` | $\mathbb E[f_a]$, $\mathbb E[a f_a]$ |
| `M.second`, `M.z_second`, `M.z2_second` | $\mathbb E[f_{aa}]$, $\mathbb E[a f_{aa}]$, $\mathbb E[a^2 f_{aa}]$ |

The maps are block diagonal in states; flattening coefficients and reshaping
outputs preserves the $S\times N$ layout. The steady-state moments are prepared
once: `E_Fa, E_a_Fa` are
$\mathbb E[(1,a)F_a]$, and `E_Faa, E_a_Faa, E_a2_Faa` are
$\mathbb E[(1,a,a^2)F_{aa}]$.

### Borrowing-limit correction

The inherited CDF is zero below $a_{\min}$ and may jump at that point.
The shock at which the asset argument reaches this boundary is

$$
\bar\epsilon\equiv\frac{m_j-Tr-Ra_{\min}}{Ze_s}.
$$

Larger shocks imply an asset cutoff below $a_{\min}$, so this is the upper
shock limit of the nonzero CDF integral. Write its derivatives as
$\epsilon_1\equiv d\bar\epsilon$ and $\epsilon_2\equiv d^2\bar\epsilon$ (`eps1`, `eps2`).
Differentiating $a(\bar\epsilon)=a_{\min}$ gives

$$
\epsilon_1=\frac{R}{Ze_s}da_{\min},\qquad
\epsilon_2=\frac{R}{Ze_s}d^2a_{\min}-2\alpha_1\epsilon_1.
$$

Here $da_{\min}\equiv da(\bar\epsilon)=\alpha_1a_{\min}+\beta_1$
(`da_min`) is evaluated at the steady-state threshold. The same convention
applies to higher derivatives; $a_{\min}$ is fixed.

To differentiate the integral with its moving upper limit, define its
integrand $g(\epsilon)\equiv F(a(\epsilon))p(\epsilon)$, where $p$ is the
shock density. In $dg$, the income draw is fixed; $g_\epsilon$ differentiates
with respect to that draw. At the threshold, with arguments suppressed,

$$
\begin{aligned}
g&=Fp,\\
dg&=[dF+F_a da_{\min}]p,\\
g_\epsilon&=F_a(-Ze_s/R)p+Fp_\epsilon.
\end{aligned}
$$

The moving-limit rule gives

$$
\begin{aligned}
d\widetilde F&=(d\widetilde F)_{\mathrm{int}}+g\epsilon_1,\\
d^2\widetilde F&=(d^2\widetilde F)_{\mathrm{int}}
 +2dg\epsilon_1+g_\epsilon\epsilon_1^2+g\epsilon_2.
\end{aligned}
$$

### Persistent-state transition

Multiplication by `Pi_F` gives the returned CDF derivatives.

## Additional third-order calculations

### Asset-cutoff derivatives

The same cutoff identity gives
$R\,d^3a=d^3m_j-d^3Tr-e_s\epsilon\,d^3Z-a\,d^3r-3d^2r\,da-3dr\,d^2a$.
Its affine coefficients are

$$
\alpha_3\equiv\frac{d^3Z}{Z}-\frac{d^3r}{R}
          -\frac{3(d^2r\,\alpha_1+dr\,\alpha_2)}{R},\qquad
\beta_3\equiv\frac{d^3m_j-d^3Tr-(m_j-Tr)d^3Z/Z-3d^2r\,\beta_1-3dr\,\beta_2}{R}.
$$

### CDF expectation

Before expansion into moments, the third-order chain rule is

$$
\begin{aligned}
(d^3\widetilde F)_{\mathrm{int}}=\mathbb E[{}
&d^3F+3d^2F_a\,da+3dF_{aa}(da)^2+3dF_a\,d^2a\\
&+F_{aaa}(da)^3+3F_{aa}\,da\,d^2a+F_a\,d^3a].
\end{aligned}
$$

Fitting `q3` to `d3F` and applying the same maps to `q1`, `q2`, and `q3`
gives the ten perturbation expectations. The steady-state moments supply the
remaining terms. The initial `third` expression
includes everything except $\mathbb E[F_{aaa}(da)^3]$; the following block
supplies that term by integration by parts.

### Integration by parts

Directly integrating the cubic spline's discontinuous third derivative is
avoidable. Since $a_\epsilon=-Ze_s/R$ is constant in $\epsilon$,
$\partial_\epsilon F_{aa}=F_{aaa}a_\epsilon$. Also,
$\partial_\epsilon da=\alpha_1 a_\epsilon$. Treating the far upper CDF tail
as flat, integration by parts gives

$$
\begin{aligned}
\mathbb E[F_{aaa}(da)^3]
={}&-\frac{R}{Ze_s}F_{aa}(a_{\min})(da_{\min})^3p(\bar\epsilon)
 -3\alpha_1\mathbb E[F_{aa}(da)^2]\\
&+\frac{R}{Ze_s}\mathbb E[F_{aa}(da)^3\,p_\epsilon/p].
\end{aligned}
$$

The lower asset endpoint corresponds to the upper shock limit, accounting
for the sign of the first term.

The second expectation is `E_Faa_da2`. Expanding
$(da)^3=(\alpha_1a+\beta_1)^3$ expresses the last expectation using the
four moments in `self.E_Faa_dlogp`:
$\mathbb E[(1,a,a^2,a^3)F_{aa}\,p_\epsilon/p]$.

### Borrowing-limit correction

Differentiating the boundary identity once more gives

$$
\epsilon_3=\frac{R}{Ze_s}d^3a_{\min}
             -3\alpha_1\epsilon_2-3\alpha_2\epsilon_1.
$$

At the steady-state threshold, the additional integrand derivatives are
below, with $a_\epsilon=-Ze_s/R$.

$$
\begin{aligned}
d^2g={}&[d^2F+2dF_a\,da_{\min}+F_{aa}(da_{\min})^2
                 +F_a\,d^2a_{\min}]p,\\
dg_\epsilon={}&[(dF_a+F_{aa}da_{\min})a_\epsilon
                       +F_a\alpha_1a_\epsilon]p
                  +(dF+F_a da_{\min})p_\epsilon,\\
g_{\epsilon\epsilon}={}&F_{aa}a_\epsilon^2p
                         +2F_a a_\epsilon p_\epsilon+F p_{\epsilon\epsilon}.
\end{aligned}
$$

The moving-limit rule gives

$$
\begin{aligned}
d^3\widetilde F=(d^3\widetilde F)_{\mathrm{int}}
&+3d^2g\epsilon_1+3dg_\epsilon\epsilon_1^2
   +g_{\epsilon\epsilon}\epsilon_1^3\\
&+3dg\epsilon_2+3g_\epsilon\epsilon_1\epsilon_2+g\epsilon_3.
\end{aligned}
$$

### Persistent-state transition

Multiplication by `Pi_F` completes the third-order step.

## Steady-state preparation

`Forward.__init__` stores the endogenous cash-on-hand grid $m_j$ as `self.m_endog` and
passes `a_shift` $=(m_j-Tr)/R$ and `lam` $=Ze_s/R$ to
`asset_expectation_matrices` to build the operators. Applying the derivative
matrices to the steady-state CDF coefficients `q_F` gives its ordinary moments.
`asset_density_moments` separately integrates the moments weighted by $p'/p$.
The constructor also evaluates the shock density
and its derivatives at $\bar\epsilon$ for the borrowing-threshold corrections.

### Borrowing-threshold derivatives

`a_spline.derivatives_at_lower_bound(q_F)` evaluates $F_a,F_{aa}$ at $a_{\min}$
from the CDF coefficients. The same operation evaluates $dF_a$ from the
coefficients of $dF$ in `step_3rd`.

### First-order maps

The cash and input derivatives combine the interior expectation with the
moving-limit correction. With $\eta\equiv e_s\epsilon$ and $p$ evaluated at
$\bar\epsilon$,

$$
\begin{aligned}
\frac{\partial\widetilde F}{\partial m_j}
 &=\frac{\mathbb E[F_a]}{R}+\frac{F(a_{\min})p}{Ze_s},\\
\frac{\partial\widetilde F}{\partial r}
 &=-\frac{\mathbb E[aF_a]}{R}-\frac{a_{\min}F(a_{\min})p}{Ze_s},\\
\frac{\partial\widetilde F}{\partial Z}
 &=-\frac{\mathbb E[\eta F_a]}{R}-\frac{F(a_{\min})p\bar\epsilon}{Z},\\
\frac{\partial\widetilde F}{\partial Tr}
 &=-\frac{\partial\widetilde F}{\partial m_j}.
\end{aligned}
$$

The code obtains $\mathbb E[\eta F_a]$ by substituting
$\eta=(m_j-Tr-Ra)/Z$. For changes in the inherited CDF, each state's map is
$\partial\widetilde F/\partial F=MC$, where $C$ converts grid values to spline
coefficients and $M$ takes their expectations.

### Mean-asset weights

The mean-asset derivative is
$d^kA=-\sum_s\pi_s\int_{a_{\min}}^{a_{\max}}d^kF_s(a)\,da$.
Interpolating each unit vector of grid values and integrating its spline
therefore gives weights acting directly on $dF$. `aggregate_assets` applies
these weights; [jacobian.py](../sequence_space/jacobian.py) propagates them through the transpose
of the linear distribution update for first-order fake news.

## Approximation and paths

The code approximates the expectation and its derivative integrals using
spline interpolation and Gauss–Legendre quadrature over the shocks that place
the asset argument inside the spline grid. Quadrature is split across spline
pieces, where higher derivatives are smooth, and truncates log income at eight
standard deviations from its mean. The far upper CDF tail is treated as flat,
with no upper-tail movement corrections.

In [paths.py](../sequence_space/paths.py), the backward sweep supplies date $t$'s endogenous
grid derivatives. The forward step uses them to propagate $d^kF_t$ to
$d^kF_{t+1}$, which is aggregated into $d^kA_t$ (`dA[t]` at first order).
By default, initial CDF and terminal marginal-value derivatives are zero.
