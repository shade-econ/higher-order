# Higher-order sequence space: an introduction

These notebooks accompany *Higher-Order Perturbation in Sequence Space: the
Certainty Correspondence*, by Adrien Auclert, Rodolfo Rigato, Matthew Rognlie,
and Ludwig Straub. They introduce the paper's methods for its smooth
standard incomplete markets (SIM) household model in both PE and GE. Starting from a nonlinear household solver,
they build up to a third-order solution under aggregate uncertainty, then use
it to simulate the economy, calculate generalized impulse responses, and
compute model moments.

The repository is intended as an accessible companion to the paper, and it is
not a full replication archive, although it does replicate a number of figures in the paper.
It focuses on the constructions in sections 3 and 4, with supporting examples from appendices B–D.

## Guide to the notebooks

Notebooks 0 through 3 form a sequence. Notebook 0 introduces the basic
smooth model, and notebook 1 uses numerical differentiation
to implement many of our results with this model. Notebook 2 then introduces analytical differentiation
and uses its far greater speed to calculate more extensive second- and third-order results.
Notebook 3 finally calculates the full third-order solution and uses it for simulation. Notebooks 1 and 2 do most of the sequence-space calculations manually inside the notebook for transparency, while notebook 3 introduces an interface where these calculations are done under the hood.

The supplemental notebook A is separate from the main sequence, focusing on how well perturbation captures the effect of large fiscal shocks, and it can be read after notebook 1.

### 0. The smooth standard incomplete markets (SIM) model

- Introduces the household model used throughout, following appendix C in the paper, and
  compares it with a conventional discrete implementation under the same
  calibration.
- Shows that the two implementations give similar steady states and both first-order
  and nonlinear consumption responses, while only the smooth implementation gives reliable second and
  third numerical derivatives. This explains why a smooth implementation is necessary for the
  paper's methods, even when a discrete model works well at first order.
- **Paper figures:** The first-order and nonlinear responses from Figure 1(a);
  Figure C.7 (sensitivity of derivatives to step size); extends Figure C.6(b)
  with a discrete-model comparison of asset distributions.

### 1. Numerical sequence-space perturbation

- Develops the certainty correspondence in sections 3.1–3.4 (**Propositions
  2–4**) using numerical differentiation: shock-size effects, interactions
  between successive surprises, and precautionary responses to anticipated
  risk. Calculates the risky steady state and the ergodic mean.
- Combines these derivatives into an impact generalized impulse response
  (**Proposition 5**), showing how it differs from a perfect-foresight response.
- Solves the IKC in general equilibrium, one order at a time, using the same
  Jacobian at each order—the **one-shot principle (Proposition 7)**. Compares
  the result with a nonlinear transition and checks the first-order
  invertibility condition underlying **Proposition 1** numerically.
- **Paper figures:** Figures 1(a,b) (PE and GE impulse responses), 2(a,b)
  (shock-size dependence), 3(a,b) (risky steady-state assets and distribution),
  4(a,b) (second- and third-order decompositions), and 5(c) (impact GIRF).

### 2. Analytical sequence-space perturbation

- Introduces the analytical differentiation approach in section 4.5, checks it
  against numerical differentiation, and compares their speed.
- Uses analytical derivatives and the one-shot principle to calculate
  second-order GE responses to shocks up to 99 quarters apart. Adding
  anticipated risk completes the second-order solution (**Propositions 3 and 7**).
- Calculates the third-order terms needed for moments, including the effects
  of the risky steady state and time-varying precautionary behavior on shock
  transmission (**Proposition 4**).
- Applies **Proposition 6** to obtain fifth-order-accurate output variance and
  autocovariance without computing every third-order history. Shows why a
  second-order solution alone misses important contributions to these moments.
- **Paper figures:** Figure B.5(a) (risky steady-state output),
  Figure B.6(a) (second-order GE decomposition into size, history, and risk effects), and Figure B.7(a,b) (output
  variance and first autocovariance).

### 3. Complete solutions and simulation

- Assembles the complete third-order GE solution with a single call, including
  interactions among three distinct shock dates, and uses it to simulate output.
- Checks the analytical moment formulas (**Proposition 6**) by differentiating
  sample moments with respect to the shock standard deviation, using the same
  simulated innovations throughout.
- Calculates a history-dependent GIRF (**Proposition 5**) and shows how local
  fiscal multipliers vary along the simulated path.
- **Connection to the paper:** illustrates the simulation and multiplier
  exercises of section 5 in the simpler IKC model.

### A. How accurate are perturbation approximations for large shocks?

- Follows appendix D, comparing first-, second-, and third-order approximations
  with nonlinear perfect-foresight responses in partial and general equilibrium.
  Finds that third order captures a wide range of economically relevant fiscal
  shocks accurately.
- Shows that larger approximation errors arise mainly in the special situation where 
 most or all of the following are true: income shocks or tax cuts are extremely large, they have zero persistence, they are lump sum rather than proportional, and we are interested only in the first-quarter impact effect (rather than the cumulative effect over two or more quarters).
- Finds that third order performs well even for a stylized COVID-sized shock
  with persistence and a plausible mix of lump-sum and proportional income gains, which we view as providing
  a demanding stress test of perturbation.
- **Paper figures:** Figures 2 and B.4 (PE and GE shock-size dependence),
  D.1 (cumulative responses), D.2–D.5 (one-time and persistent lump-sum
  transfers), and the consumption-response panels of D.6 (COVID-sized shock).

## Requirements and runtime

The code requires Python 3.10 or newer, NumPy, SciPy, Matplotlib, Numba, and a Jupyter notebook
interface. Approximate runtimes on a laptop for notebooks 0, 1, 2, 3, and A are 1, 1, 2, 6, and 7 minutes, respectively, including all setup and initial compilation. (Notebook 3 computes a full third-order solution and uses it for simulation; notebook A is expensive mainly because it solves for fully nonlinear GE paths for many different shocks.)

## Repository structure

Supporting code and tests are organized in seven folders:

- `smooth_sim/` implements the smooth SIM household model, including its
  calibration, steady state, and nonlinear transitions. See `core.py` for the underlying routines and `model.py` for a `Household` class that provides a simple interface.

- `analytical_simple/` implements the household model's analytical derivatives
  through third order, with an emphasis on readability. `backward.py` and
  `forward.py` provide `Backward` and `Forward` objects for differentiating
  the backward policy iteration and forward distribution iteration, respectively, with parallel derivations in the accompanying `.md` files.

- `analytical_fast/` provides faster versions of the `Backward` and `Forward`
  objects, used for the larger calculations in notebooks 2 and 3. It uses Numba compilation, evaluates several expectations together in one loop, and reuses intermediate results to improve speed at some cost to readability.

- `sequence_space/` contains the sequence-space calculations. `jacobian.py`
  provides the asset-income Jacobian used throughout for GE computations.
  `paths.py` and `risk_terms.py` use analytical household derivatives to calculate
  transition responses and anticipated-risk effects, supporting the explicit
  calculations in notebook 2. For the automated workflow in notebook 3,
  `histories.py` calculates responses to successive unanticipated shocks,
  `solution.py` assembles complete PE and GE solutions, and `stochastic.py`
  uses their coefficients for simulation, moments, and GIRFs.

- `spline/` provides cubic spline interpolation, evaluation, and differentiation.
  `core.py` contains the basic routines; `fixed_grid.py` and `cached_solve.py`
  support efficient repeated calculations on the same grid.

- `discrete_sim/` contains the conventional discrete SIM code used for the
  comparison in notebook 0. The steady-state and transition routines use our
  standard code (see, for instance, lecture 1 of
  [our workshop](https://github.com/shade-econ/goethe-workshop-2026));
  `calibration.py` sets up the comparison with the smooth model.

- `tests/` contains test code for the household model, analytical derivatives,
  numerical routines, and sequence-space calculations.

## A quick start with the code

Get the calibrated household and steady state, then calculate consumption after
an income shock of 5% of quarterly GDP:

```python
import numpy as np
from smooth_sim.model import paper_household

hh, ss = paper_household()
income = np.zeros(300)
income[0] = 1.

def C(nu):
    return hh.transition(ss, dZ=nu*income)["C"]

dC = C(0.05) - ss["C"]
```

Differentiate this consumption path numerically with respect to shock size:

```python
h = 2e-4
Cplus, Czero, Cminus = C(h), C(0), C(-h)
C1 = (Cplus - Cminus)/(2*h)
C2 = (Cplus - 2*Czero + Cminus)/h**2
```

Obtain the complete second-order GE solution and ergodic mean output
(about 15–20 seconds on a laptop after household setup):

```python
from sequence_space.solution import solve_ge

solution = solve_ge(hh, ss, order=2)
sigma = 0.05
mean = solution.ergodic_mean(sigma)
```

Or obtain the complete third-order GE solution and use it for simulation and
moments, as in notebook 3 (this solve takes several minutes on a laptop):

```python
solution = solve_ge(hh, ss, order=3, workers=4)

sigma = 0.05
burnin = solution.T + solution.L - 2
shocks = np.random.default_rng(0).standard_normal(1000 + burnin)
y1, y2, y3 = solution.simulate(shocks, burnin=burnin)
Y_sim = solution.X_ss + sigma*y1 + sigma**2*y2 + sigma**3*y3

mean = solution.ergodic_mean(sigma)
variance, autocovariance = solution.covariances(sigma, lags=(0, 1))
```
