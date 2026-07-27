# Charged Galactic Wormhole Shadow Analysis

## 1. Overview

This repository provides a faithful computational implementation of the charged galactic wormhole model presented by Rahaman et al. (2025), arXiv:2503.16111.  It constructs the spacetime metric for a traversable wormhole embedded in an exponential dark matter halo, computes the associated energy conditions, integrates null geodesics, and generates observable shadow profiles.  The codebase extends the static analysis to rotating wormholes of Teo-type and incorporates plasma effects on photon trajectories.  All mathematical expressions are transcribed directly from the source code, which explicitly references the equation numbers of the primary paper.

The target audience consists of General Relativity researchers, astrophysicists, cosmologists, and graduate students interested in the interplay between wormhole geometry, dark matter profiles, and strong-field gravitational lensing.

---

## 2. Scientific Motivation

Observational campaigns such as the Event Horizon Telescope (EHT) have opened a new window into strong-field gravity by imaging black-hole shadows.  A natural theoretical question is whether traversable wormholes—hypothetical shortcuts through spacetime supported by exotic matter—can produce shadow morphologies distinguishable from Kerr black holes.  Rahaman et al. (2025) proposed a static, spherically symmetric charged wormhole sourced by an exponential dark matter density profile.  This repository translates that analytical model into a numerical pipeline: it verifies the metric structure, checks the flare-out and energy conditions, computes photon spheres in vacuum and plasma environments, and produces synthetic shadow images for comparison with EHT observations.

---

## 3. Mathematical Formulation

### 3.1 Spacetime Metric

The line element implemented in `src/metric.py` corresponds to Eq. (3) of Rahaman et al. (2025):

$$
ds^{2} = -\left(1 + \frac{Q^{2}}{r^{2}}\right)dt^{2} + \left(1 - \frac{b(r)}{r} + \frac{Q^{2}}{r^{2}}\right)^{-1}dr^{2} + r^{2}\left(d\theta^{2} + \sin^{2}\theta\,d\phi^{2}\right)
$$

The metric components are:

- **Temporal component:** $g_{tt}(r) = -\left(1 + Q^{2}/r^{2}\right)$
- **Radial component:** $g_{rr}(r) = \left(1 - b(r)/r + Q^{2}/r^{2}\right)^{-1}$
- **Angular components:** $g_{\theta\theta}(r) = r^{2}$, $g_{\phi\phi}(r,\theta) = r^{2}\sin^{2}\theta$

Here $Q$ is the electric charge and $b(r)$ is the shape function determined by the dark matter distribution.  The inverse metric is computed algebraically and returned as a dictionary for use in the geodesic integrator.

### 3.2 Dark Matter Profile and Shape Function

The dark matter density follows an exponential profile (`src/dark_matter.py`, Eq. 9):

$$
\rho_{w}(r) = \rho_{s}\,\exp\!\left(-\frac{r}{r_{s}}\right)
$$

with central density $\rho_{s}$ and scale radius $r_{s}$.  The shape function $b(r)$ is obtained by integrating the matter distribution (`src/shape_function.py`, Eq. 11):

$$
b(r) = -8\pi\rho_{s} r_{s}\left(r^{2} + 2r r_{s} + 2r_{s}^{2}\right)\exp\!\left(-\frac{r}{r_{s}}\right) + C_{1}
$$

where $C_{1}$ is an integration constant.  The effective shape function including the electric charge contribution (Eq. 12) is:

$$
b_{\text{eff}}(r) = b(r) - \frac{Q^{2}}{r}
$$

**Throat condition:** The wormhole throat radius $r_{0}$ is defined by $b_{\text{eff}}(r_{0}) = r_{0}$.  The module `src/shape_function.py` solves this root-finding problem numerically via `scipy.optimize.root_scalar`.

**Flare-out condition:** Traversability requires $b_{\text{eff}}'(r_{0}) < 1$ at the throat.  The derivative $b'(r)$ is implemented analytically:

$$
b'(r) = -8\pi\rho_{s} r_{s}\left[(2r + 2r_{s}) - \frac{r^{2} + 2r r_{s} + 2r_{s}^{2}}{r_{s}}\right]\exp\!\left(-\frac{r}{r_{s}}\right)
$$

and $b_{\text{eff}}'(r) = b'(r) + Q^{2}/r^{2}$.

### 3.3 Energy Conditions

The stress-energy tensor is derived from the Einstein field equations for the metric above.  `src/energy_conditions.py` implements the following components (Eqs. 4–8):

**Energy density:**
$$
\rho(r) = \frac{1}{8\pi}\left(\frac{b'(r)}{r^{2}} + \frac{Q^{2}}{r^{4}}\right)
$$

**Radial pressure:**
$$
P_{r}(r) = -\tau(r)
$$
where $\tau(r)$ is a composite function of $b(r)$, $b'(r)$, $Q$, and $g_{rr}$ as given in Eq. (5) of the paper.

**Tangential pressure:**
$$
P_{t}(r) = P(r)
$$
with $P(r)$ given by Eq. (6), involving the terms
$$
\text{term}_{1} = \frac{Q^{2}(3r^{2}+Q^{2})}{r^{2}(r^{2}+Q^{2})}, \quad
\text{term}_{2} = \frac{b'r - b + 2Q^{2}/r^{2}}{2(r^{2}-br+Q^{2})}, \quad
\text{term}_{3} = \frac{Q^{2}}{r(r^{2}+Q^{2})^{2}}
$$
assembled with the inverse metric factor $g_{rr}^{-1}$.

**Null Energy Condition (NEC):** The code checks
- Radial NEC: $\rho + P_{r} \geq 0$
- Tangential NEC: $\rho + P_{t} \geq 0$

Violations are flagged explicitly, indicating the presence of exotic matter required to keep the wormhole throat open.

### 3.4 Rotating Extension

`src/metric_rotating.py` extends the static metric to a Teo-type rotating charged wormhole.  The rotation is introduced via a frame-dragging term:

$$
\omega(r) = \frac{2a}{r^{3}}
$$

where $a$ is the spin parameter.  The off-diagonal and modified angular components are:

- $g_{t\phi}(r,\theta) = -\omega(r)\,r^{2}\sin^{2}\theta$
- $g_{\phi\phi}^{\text{rot}}(r,\theta) = g_{\phi\phi}(r,\theta)\left[1 + \omega(r)^{2}r^{2}\sin^{2}\theta\right]$

The inverse metric accounts for the $2\times2$ $(t,\phi)$ block determinant:
$$
\det = g_{tt}\,g_{\phi\phi} - g_{t\phi}^{2}
$$

### 3.5 Plasma Environment

`src/plasma.py` models the plasma frequency squared $\omega_{p}^{2}(r,\theta)$ through four phenomenological profiles:

| Profile | $\omega_{p}^{2}(r,\theta)$ |
|---|---|
| Homogeneous | $\rho_{0}$ |
| Longitudinal | $\rho_{0}\left(1 + 2\sin^{2}\theta\right)$ |
| Radial | $\rho_{0}\,r^{-3/2}$ |
| Spherical | $\rho_{0}\,r^{-2}$ |

The plasma modifies the effective potential for photons via a multiplicative factor $(1 - \omega_{p}^{2})$.

### 3.6 Null Geodesics and Shadows

**Static case (`src/geodesics.py`):** Null geodesics are integrated using a Hamiltonian formulation.  The Hamiltonian is
$$
H = \frac{1}{2}\left(g^{tt}p_{t}^{2} + g^{rr}p_{r}^{2} + g^{\theta\theta}p_{\theta}^{2} + g^{\phi\phi}p_{\phi}^{2}\right)
$$
and the equations of motion are solved with `scipy.integrate.solve_ivp` (RK45, relative tolerance $10^{-8}$).

**Shadow computation (`src/shadow.py`):** The effective potential for photons in the equatorial plane ($\theta = \pi/2$) is
$$
V_{\text{eff}}(r) = -\frac{g_{tt}\left(1 - \omega_{p}^{2}\right)}{g_{rr}}
$$

The photon sphere radius $r_{\text{ph}}$ is found by minimizing $-V_{\text{eff}}(r)$.  For a distant observer at inclination $\theta_{o}$, the shadow boundary in celestial coordinates $(\alpha,\beta)$ is constructed geometrically.  In the presence of a homogeneous plasma with density $\rho < 1$, the critical radius scales as
$$
r_{\text{crit}} = \frac{r_{\text{ph}}}{\sqrt{1 - \rho}}
$$

**Rotating case (`src/shadow_rotating.py`):** The effective potential becomes
$$
V_{\text{eff}}(r) = -\frac{\det(g)}{g_{\phi\phi}\left(1 - \omega_{p}^{2}\right)}
$$
and the shadow boundary incorporates the spin-dependent shift in the celestial coordinates.

---

## 4. Computational Architecture

### 4.1 Package Layout
src/
├── init.py              # Package initialization, imports all modules
├── metric.py              # Static charged wormhole metric (Eq. 3)
├── metric_rotating.py     # Teo-type rotating extension
├── shape_function.py      # b(r), b_eff(r), derivatives, throat finder
├── dark_matter.py         # Exponential DM profile ρ_w(r)
├── energy_conditions.py   # ρ, P_r, P_t, NEC checks (Eqs. 4–8)
├── plasma.py              # Plasma frequency profiles
├── geodesics.py           # Null geodesic integration (static)
├── geodesics_rotating.py  # Null geodesic integration (rotating + plasma)
├── shadow.py              # Shadow computation (static)
├── shadow_rotating.py     # Shadow computation (rotating)
├── shadow_plasma.py       # Shadow computation with plasma
├── visualization.py       # Basic plotting utilities
├── extended_visualization.py  # Shadow/plasma/rotation comparison plots
└── plot_generator.py      # Publication-quality plot templates
plain

### 4.2 Data Flow

```mermaid
graph TD
    A[config/params.yaml] --> B[DarkMatterProfile]
    A --> C[ShapeFunction]
    B --> C
    C --> D[ChargedWormholeMetric]
    C --> E[EnergyConditions]
    C --> F[RotatingChargedWormholeMetric]
    D --> G[NullGeodesic]
    D --> H[WormholeShadow]
    D --> I[PlasmaShadow]
    F --> J[RotatingNullGeodesic]
    F --> K[RotatingWormholeShadow]
    L[PlasmaProfile] --> H
    L --> I
    L --> J
    L --> K
    G --> M[solve_ivp integration]
    H --> N[Shadow images & boundaries]
    K --> N
    E --> O[Energy condition plots]
    C --> P[Shape function plots]
    Q[final_plots.py] --> R[Publication-ready multi-panel figures]
5. Source Modules
5.1 src/metric.py — Static Metric
Scientific responsibility: Encodes the line element (Eq. 3).
Computational responsibility: Evaluates g 
tt
​
  , g 
rr
​
  , g 
θθ
​
  , g 
ϕϕ
​
   and their inverses for arbitrary (r,θ) .
Inputs: ShapeFunction instance.
Outputs: Dictionary of metric components.
5.2 src/shape_function.py — Shape Function
Scientific responsibility: Computes b(r)  and b 
eff
​
 (r)  from the dark matter integral (Eqs. 11–12).
Computational responsibility: Provides both numeric and SymPy symbolic evaluations, computes b 
′
 (r)  and b 
eff
′
​
 (r) , finds the throat via root-finding, and checks the flare-out condition.
Inputs: ρ 
s
​
  , r 
s
​
  , C 
1
​
  , Q  (or a YAML config path).
Outputs: Numeric arrays of b(r) , b 
eff
​
 (r) , derivatives, and throat radius.
5.3 src/dark_matter.py — Dark Matter Halo
Scientific responsibility: Exponential density profile (Eq. 9).
Inputs: ρ 
s
​
  , r 
s
​
  .
Outputs: ρ 
w
​
 (r) .
5.4 src/energy_conditions.py — Stress-Energy and NEC
Scientific responsibility: Computes ρ , P 
r
​
  , P 
t
​
   and tests the Null Energy Condition.
Inputs: ShapeFunction instance.
Outputs: Energy densities, pressures, and boolean NEC flags.
5.5 src/metric_rotating.py — Rotating Metric
Scientific responsibility: Teo-type rotating extension of the static metric.
Computational responsibility: Adds frame-dragging ω(r)  and modifies g 
ϕϕ
​
  ; computes the full 5 -component inverse metric including g 
tϕ
  .
Inputs: ShapeFunction instance, spin parameter a .
5.6 src/plasma.py — Plasma Profiles
Scientific responsibility: Phenomenological plasma frequency models.
Outputs: ω 
p
2
​
 (r,θ)  for homogeneous, longitudinal, radial, and spherical profiles.
5.7 src/shadow.py and src/shadow_rotating.py — Shadow Engines
Scientific responsibility: Determine the photon sphere and map it to observer celestial coordinates.
Computational responsibility: Minimize the effective potential to find r 
ph
​
  , then construct (α,β)  boundaries and generate N×N  binary shadow images.
Key algorithm: scipy.optimize.minimize_scalar on −V 
eff
​
 (r) .
5.8 src/geodesics.py and src/geodesics_rotating.py — Geodesic Integrators
Scientific responsibility: Hamiltonian formulation of null geodesics.
Computational responsibility: solve_ivp (RK45) integration of the 8 -dimensional phase-space vector (r,θ,ϕ,t,p 
r
​
 ,p 
θ
​
 ,p 
ϕ
​
 ,p 
t
​
 ) .
5.9 Visualization Modules
src/visualization.py: Shape-function panels, energy-condition grids, metric-component curves.
src/extended_visualization.py: Plasma-effect series, rotation-effect series, Kerr-vs-wormhole side-by-sides, shadow-boundary overlays, and EHT constraint polygons.
src/plot_generator.py: Reusable templates for deflection-angle plots, 2-D/3-D parameter-space contours, and boundary comparisons.
6. Executable Scripts
6.1 run_analysis.py — Basic Pipeline
Loads config/params.yaml, instantiates the dark matter profile, shape function, metric, and energy conditions, then:
Plots b(r)  and b 
eff
​
 (r)  versus r .
Evaluates and plots ρ , ρ+P 
r
​
  , ρ+P 
t
​
  , and ρ+P 
r
​
 +2P 
t
​
  .
Plots g 
tt
​
 (r)  and g 
rr
​
 (r) .
Reports the throat radius and flare-out status.
Expected outputs: PNG files in plots/shape_functions/, plots/energy_conditions/, and plots/.
6.2 run_extended_analysis.py — Shadow, Plasma & Rotation Pipeline
Executes a seven-part workflow:
Static vacuum shadow: 200×200  pixel image.
Plasma series: Shadows for homogeneous plasma densities ρ=0.0,0.3,0.5,0.7 .
Rotation series: Shadows for spin parameters a=0.0,0.3,0.6,0.9 .
Kerr comparison: Synthetic Kerr shadow (a=0.9 ) versus rotating wormhole shadow.
Boundary comparison: Celestial-coordinate curves (α,β)  for static vacuum, static plasma, and rotating cases.
3-D parameter space: Surface plot of mean shadow radius R(a,ρ) .
EHT constraints: Allowed (a,ρ)  polygons for M87* and Sgr A*.
Expected outputs: PNG files in plots/shadows/, plots/comparisons/.
6.3 final_plots.py — Publication Figures
Generates seven advanced multi-panel figures saved to plots/advanced/:
01_shape_function_advanced.png — Six panels: b(r) , b 
eff
​
 (r) , flare-out ratio b 
eff
​
 /r , derivative b 
eff
′
​
  , dark matter density, and a parameter table.
02_energy_conditions_advanced.png — Six panels: ρ(r) , radial NEC, tangential NEC, SEC, pressure comparison, and a summary table.
03_shadow_radius_advanced.png — Four panels: synthetic shadow radius versus spin, versus plasma density, percent deviation from Kerr, and a comparison table.
04_deflection_advanced.png — Four panels: deflection angle versus impact parameter on linear and log-log scales, parameter dependence, and a summary table.  Note: deflection curves are phenomenological (∝1/b  with wormhole corrections), not full numerical integrations.
05_eht_advanced.png — Four panels: allowed parameter-space polygons, observational bar chart, shadow-radius contours, and an EHT summary table.
06_shadow_boundary_advanced.png — Two panels: spin dependence and plasma dependence of the shadow boundary.
07_comprehensive_summary_advanced.png — Six panels combining shape function, energy conditions, pressures, metric components, flare-out condition, and a global summary table.
7. Notebooks
notebooks/01_metric_derivation.ipynb
Purpose: Symbolic verification of the metric tensor.
Workflow: Loads config/params.yaml, instantiates DarkMatterProfile and ShapeFunction, and uses SymPy to print the closed-form expressions for g 
tt
​
   and g 
rr
​
  .  This confirms that the numerical implementation in src/metric.py matches the analytical formula (Eq. 3).
Outputs: LaTeX-formatted metric components and static PNG figures in notebooks/plots/.
8. Testing
tests/test_metric.py contains unit tests for the static metric:
test_g_tt: Asserts g 
tt
​
 (r)=−(1+Q 
2
 /r 
2
 ) .
test_g_rr: Asserts g 
rr
​
 (r)=(1−b(r)/r+Q 
2
 /r 
2
 ) 
−1
  .
test_metric_tensor: Verifies dictionary keys and angular components.
Run with:
bash
pytest tests/test_metric.py
9. Configuration
All parameters are centralized in config/params.yaml:
yaml
dark_matter:
  rho_s: 0.01          # Central density (Eq. 9)
  r_s: 1.0             # Scale radius

wormhole:
  Q: 0.0               # Electric charge
  C1: 1.0              # Integration constant (Eq. 11)

analysis:
  r_min: 0.5
  r_max: 10.0
  n_points: 1000
  r_throat: 1.0
Scripts read this file at runtime; no hard-coded astrophysical parameters appear in the analysis drivers.
10. Installation
Dependencies are listed in requirements.txt:
plain
numpy>=1.24.0
scipy>=1.10.0
sympy>=1.11.0
matplotlib>=3.7.0
pandas>=1.5.0
plotly>=5.13.0
seaborn>=0.12.0
pyyaml>=6.0
jupyter>=1.0.0
tqdm>=4.64.0
pytest>=7.0.0
imageio>=2.25.0
Install in a virtual environment:
bash
pip install -r requirements.txt
11. References
The BibTeX database in docs/references.bib contains:
Rahaman et al. (2025) — Gravitational lensing due to charged galactic wormhole, arXiv:2503.16111.  Primary reference for the metric, shape function, dark matter profile, and energy conditions.
Morris & Thorne (1988) — Wormholes in spacetime and their use for interstellar travel, Am. J. Phys. 56, 395.  Foundational traversable wormhole framework.
Teo (1998) — Rotating traversable wormholes, Phys. Rev. D 58, 024014.  Basis for the rotating extension in src/metric_rotating.py.
Sofue (2013) — Rotation curve of the Milky Way, PASJ 65, S5.  Galactic rotation-curve context.
12. Implementation Notes and Caveats
Shadow method: The code uses an effective-potential / critical-radius approximation rather than full numerical ray tracing.  Shadow boundaries are circles (or shifted circles for rotation) whose radii are set by r 
ph
​
   and the plasma scaling factor.  This is consistent with the analytical approach of the parent paper.
Synthetic comparisons: final_plots.py employs phenomenological curves for Kerr shadow radii, deflection angles, and EHT constraint regions.  These are illustrative models designed for rapid parameter-space exploration, not the output of detailed general-relativistic ray-tracing simulations.
Plasma transparency: The homogeneous plasma model requires ρ<1  for a real critical radius; the code skips shadow generation when ρ≥1 .
Geodesic integration: While the Hamiltonian and equations of motion are fully implemented, the shadow modules do not call the integrator for every pixel; instead they rely on the effective-potential minimum, which is computationally efficient and appropriate for the spherically symmetric (or slowly rotating) case.
SymPy support: ShapeFunction provides both numeric (numpy) and symbolic (sympy) evaluations, enabling exact algebraic verification in the notebook pipeline.
plain
# Charged Galactic Wormhole Shadow Analysis

## 1. Overview

This repository provides a faithful computational implementation of the charged galactic wormhole model presented by Rahaman et al. (2025), arXiv:2503.16111.  It constructs the spacetime metric for a traversable wormhole embedded in an exponential dark matter halo, computes the associated energy conditions, integrates null geodesics, and generates observable shadow profiles.  The codebase extends the static analysis to rotating wormholes of Teo-type and incorporates plasma effects on photon trajectories.  All mathematical expressions are transcribed directly from the source code, which explicitly references the equation numbers of the primary paper.

The target audience consists of General Relativity researchers, astrophysicists, cosmologists, and graduate students interested in the interplay between wormhole geometry, dark matter profiles, and strong-field gravitational lensing.

---

## 2. Scientific Motivation

Observational campaigns such as the Event Horizon Telescope (EHT) have opened a new window into strong-field gravity by imaging black-hole shadows.  A natural theoretical question is whether traversable wormholes—hypothetical shortcuts through spacetime supported by exotic matter—can produce shadow morphologies distinguishable from Kerr black holes.  Rahaman et al. (2025) proposed a static, spherically symmetric charged wormhole sourced by an exponential dark matter density profile.  This repository translates that analytical model into a numerical pipeline: it verifies the metric structure, checks the flare-out and energy conditions, computes photon spheres in vacuum and plasma environments, and produces synthetic shadow images for comparison with EHT observations.

---

## 3. Mathematical Formulation

### 3.1 Spacetime Metric

The line element implemented in `src/metric.py` corresponds to Eq. (3) of Rahaman et al. (2025):

$$
ds^{2} = -\left(1 + \frac{Q^{2}}{r^{2}}\right)dt^{2} + \left(1 - \frac{b(r)}{r} + \frac{Q^{2}}{r^{2}}\right)^{-1}dr^{2} + r^{2}\left(d\theta^{2} + \sin^{2}\theta\,d\phi^{2}\right)
$$

The metric components are:

- **Temporal component:** $g_{tt}(r) = -\left(1 + Q^{2}/r^{2}\right)$
- **Radial component:** $g_{rr}(r) = \left(1 - b(r)/r + Q^{2}/r^{2}\right)^{-1}$
- **Angular components:** $g_{\theta\theta}(r) = r^{2}$, $g_{\phi\phi}(r,\theta) = r^{2}\sin^{2}\theta$

Here $Q$ is the electric charge and $b(r)$ is the shape function determined by the dark matter distribution.  The inverse metric is computed algebraically and returned as a dictionary for use in the geodesic integrator.

### 3.2 Dark Matter Profile and Shape Function

The dark matter density follows an exponential profile (`src/dark_matter.py`, Eq. 9):

$$
\rho_{w}(r) = \rho_{s}\,\exp\!\left(-\frac{r}{r_{s}}\right)
$$

with central density $\rho_{s}$ and scale radius $r_{s}$.  The shape function $b(r)$ is obtained by integrating the matter distribution (`src/shape_function.py`, Eq. 11):

$$
b(r) = -8\pi\rho_{s} r_{s}\left(r^{2} + 2r r_{s} + 2r_{s}^{2}\right)\exp\!\left(-\frac{r}{r_{s}}\right) + C_{1}
$$

where $C_{1}$ is an integration constant.  The effective shape function including the electric charge contribution (Eq. 12) is:

$$
b_{\text{eff}}(r) = b(r) - \frac{Q^{2}}{r}
$$

**Throat condition:** The wormhole throat radius $r_{0}$ is defined by $b_{\text{eff}}(r_{0}) = r_{0}$.  The module `src/shape_function.py` solves this root-finding problem numerically via `scipy.optimize.root_scalar`.

**Flare-out condition:** Traversability requires $b_{\text{eff}}'(r_{0}) < 1$ at the throat.  The derivative $b'(r)$ is implemented analytically:

$$
b'(r) = -8\pi\rho_{s} r_{s}\left[(2r + 2r_{s}) - \frac{r^{2} + 2r r_{s} + 2r_{s}^{2}}{r_{s}}\right]\exp\!\left(-\frac{r}{r_{s}}\right)
$$

and $b_{\text{eff}}'(r) = b'(r) + Q^{2}/r^{2}$.

### 3.3 Energy Conditions

The stress-energy tensor is derived from the Einstein field equations for the metric above.  `src/energy_conditions.py` implements the following components (Eqs. 4–8):

**Energy density:**
$$
\rho(r) = \frac{1}{8\pi}\left(\frac{b'(r)}{r^{2}} + \frac{Q^{2}}{r^{4}}\right)
$$

**Radial pressure:**
$$
P_{r}(r) = -\tau(r)
$$
where $\tau(r)$ is a composite function of $b(r)$, $b'(r)$, $Q$, and $g_{rr}$ as given in Eq. (5) of the paper.

**Tangential pressure:**
$$
P_{t}(r) = P(r)
$$
with $P(r)$ given by Eq. (6), involving the terms
$$
\text{term}_{1} = \frac{Q^{2}(3r^{2}+Q^{2})}{r^{2}(r^{2}+Q^{2})}, \quad
\text{term}_{2} = \frac{b'r - b + 2Q^{2}/r^{2}}{2(r^{2}-br+Q^{2})}, \quad
\text{term}_{3} = \frac{Q^{2}}{r(r^{2}+Q^{2})^{2}}
$$
assembled with the inverse metric factor $g_{rr}^{-1}$.

**Null Energy Condition (NEC):** The code checks
- Radial NEC: $\rho + P_{r} \geq 0$
- Tangential NEC: $\rho + P_{t} \geq 0$

Violations are flagged explicitly, indicating the presence of exotic matter required to keep the wormhole throat open.

### 3.4 Rotating Extension

`src/metric_rotating.py` extends the static metric to a Teo-type rotating charged wormhole.  The rotation is introduced via a frame-dragging term:

$$
\omega(r) = \frac{2a}{r^{3}}
$$

where $a$ is the spin parameter.  The off-diagonal and modified angular components are:

- $g_{t\phi}(r,\theta) = -\omega(r)\,r^{2}\sin^{2}\theta$
- $g_{\phi\phi}^{\text{rot}}(r,\theta) = g_{\phi\phi}(r,\theta)\left[1 + \omega(r)^{2}r^{2}\sin^{2}\theta\right]$

The inverse metric accounts for the $2\times2$ $(t,\phi)$ block determinant:
$$
\det = g_{tt}\,g_{\phi\phi} - g_{t\phi}^{2}
$$

### 3.5 Plasma Environment

`src/plasma.py` models the plasma frequency squared $\omega_{p}^{2}(r,\theta)$ through four phenomenological profiles:

| Profile | $\omega_{p}^{2}(r,\theta)$ |
|---|---|
| Homogeneous | $\rho_{0}$ |
| Longitudinal | $\rho_{0}\left(1 + 2\sin^{2}\theta\right)$ |
| Radial | $\rho_{0}\,r^{-3/2}$ |
| Spherical | $\rho_{0}\,r^{-2}$ |

The plasma modifies the effective potential for photons via a multiplicative factor $(1 - \omega_{p}^{2})$.

### 3.6 Null Geodesics and Shadows

**Static case (`src/geodesics.py`):** Null geodesics are integrated using a Hamiltonian formulation.  The Hamiltonian is
$$
H = \frac{1}{2}\left(g^{tt}p_{t}^{2} + g^{rr}p_{r}^{2} + g^{\theta\theta}p_{\theta}^{2} + g^{\phi\phi}p_{\phi}^{2}\right)
$$
and the equations of motion are solved with `scipy.integrate.solve_ivp` (RK45, relative tolerance $10^{-8}$).

**Shadow computation (`src/shadow.py`):** The effective potential for photons in the equatorial plane ($\theta = \pi/2$) is
$$
V_{\text{eff}}(r) = -\frac{g_{tt}\left(1 - \omega_{p}^{2}\right)}{g_{rr}}
$$

The photon sphere radius $r_{\text{ph}}$ is found by minimizing $-V_{\text{eff}}(r)$.  For a distant observer at inclination $\theta_{o}$, the shadow boundary in celestial coordinates $(\alpha,\beta)$ is constructed geometrically.  In the presence of a homogeneous plasma with density $\rho < 1$, the critical radius scales as
$$
r_{\text{crit}} = \frac{r_{\text{ph}}}{\sqrt{1 - \rho}}
$$

**Rotating case (`src/shadow_rotating.py`):** The effective potential becomes
$$
V_{\text{eff}}(r) = -\frac{\det(g)}{g_{\phi\phi}\left(1 - \omega_{p}^{2}\right)}
$$
and the shadow boundary incorporates the spin-dependent shift in the celestial coordinates.

---

## 4. Computational Architecture

### 4.1 Package Layout
src/
├── init.py              # Package initialization, imports all modules
├── metric.py              # Static charged wormhole metric (Eq. 3)
├── metric_rotating.py     # Teo-type rotating extension
├── shape_function.py      # b(r), b_eff(r), derivatives, throat finder
├── dark_matter.py         # Exponential DM profile ρ_w(r)
├── energy_conditions.py   # ρ, P_r, P_t, NEC checks (Eqs. 4–8)
├── plasma.py              # Plasma frequency profiles
├── geodesics.py           # Null geodesic integration (static)
├── geodesics_rotating.py  # Null geodesic integration (rotating + plasma)
├── shadow.py              # Shadow computation (static)
├── shadow_rotating.py     # Shadow computation (rotating)
├── shadow_plasma.py       # Shadow computation with plasma
├── visualization.py       # Basic plotting utilities
├── extended_visualization.py  # Shadow/plasma/rotation comparison plots
└── plot_generator.py      # Publication-quality plot templates
plain

### 4.2 Data Flow

```mermaid
graph TD
    A[config/params.yaml] --> B[DarkMatterProfile]
    A --> C[ShapeFunction]
    B --> C
    C --> D[ChargedWormholeMetric]
    C --> E[EnergyConditions]
    C --> F[RotatingChargedWormholeMetric]
    D --> G[NullGeodesic]
    D --> H[WormholeShadow]
    D --> I[PlasmaShadow]
    F --> J[RotatingNullGeodesic]
    F --> K[RotatingWormholeShadow]
    L[PlasmaProfile] --> H
    L --> I
    L --> J
    L --> K
    G --> M[solve_ivp integration]
    H --> N[Shadow images & boundaries]
    K --> N
    E --> O[Energy condition plots]
    C --> P[Shape function plots]
    Q[final_plots.py] --> R[Publication-ready multi-panel figures]
5. Source Modules
5.1 src/metric.py — Static Metric
Scientific responsibility: Encodes the line element (Eq. 3).
Computational responsibility: Evaluates g 
tt
​
  , g 
rr
​
  , g 
θθ
​
  , g 
ϕϕ
​
   and their inverses for arbitrary (r,θ) .
Inputs: ShapeFunction instance.
Outputs: Dictionary of metric components.
5.2 src/shape_function.py — Shape Function
Scientific responsibility: Computes b(r)  and b 
eff
​
 (r)  from the dark matter integral (Eqs. 11–12).
Computational responsibility: Provides both numeric and SymPy symbolic evaluations, computes b 
′
 (r)  and b 
eff
′
​
 (r) , finds the throat via root-finding, and checks the flare-out condition.
Inputs: ρ 
s
​
  , r 
s
​
  , C 
1
​
  , Q  (or a YAML config path).
Outputs: Numeric arrays of b(r) , b 
eff
​
 (r) , derivatives, and throat radius.
5.3 src/dark_matter.py — Dark Matter Halo
Scientific responsibility: Exponential density profile (Eq. 9).
Inputs: ρ 
s
​
  , r 
s
​
  .
Outputs: ρ 
w
​
 (r) .
5.4 src/energy_conditions.py — Stress-Energy and NEC
Scientific responsibility: Computes ρ , P 
r
​
  , P 
t
​
   and tests the Null Energy Condition.
Inputs: ShapeFunction instance.
Outputs: Energy densities, pressures, and boolean NEC flags.
5.5 src/metric_rotating.py — Rotating Metric
Scientific responsibility: Teo-type rotating extension of the static metric.
Computational responsibility: Adds frame-dragging ω(r)  and modifies g 
ϕϕ
​
  ; computes the full 5 -component inverse metric including g 
tϕ
  .
Inputs: ShapeFunction instance, spin parameter a .
5.6 src/plasma.py — Plasma Profiles
Scientific responsibility: Phenomenological plasma frequency models.
Outputs: ω 
p
2
​
 (r,θ)  for homogeneous, longitudinal, radial, and spherical profiles.
5.7 src/shadow.py and src/shadow_rotating.py — Shadow Engines
Scientific responsibility: Determine the photon sphere and map it to observer celestial coordinates.
Computational responsibility: Minimize the effective potential to find r 
ph
​
  , then construct (α,β)  boundaries and generate N×N  binary shadow images.
Key algorithm: scipy.optimize.minimize_scalar on −V 
eff
​
 (r) .
5.8 src/geodesics.py and src/geodesics_rotating.py — Geodesic Integrators
Scientific responsibility: Hamiltonian formulation of null geodesics.
Computational responsibility: solve_ivp (RK45) integration of the 8 -dimensional phase-space vector (r,θ,ϕ,t,p 
r
​
 ,p 
θ
​
 ,p 
ϕ
​
 ,p 
t
​
 ) .
5.9 Visualization Modules
src/visualization.py: Shape-function panels, energy-condition grids, metric-component curves.
src/extended_visualization.py: Plasma-effect series, rotation-effect series, Kerr-vs-wormhole side-by-sides, shadow-boundary overlays, and EHT constraint polygons.
src/plot_generator.py: Reusable templates for deflection-angle plots, 2-D/3-D parameter-space contours, and boundary comparisons.
6. Executable Scripts
6.1 run_analysis.py — Basic Pipeline
Loads config/params.yaml, instantiates the dark matter profile, shape function, metric, and energy conditions, then:
Plots b(r)  and b 
eff
​
 (r)  versus r .
Evaluates and plots ρ , ρ+P 
r
​
  , ρ+P 
t
​
  , and ρ+P 
r
​
 +2P 
t
​
  .
Plots g 
tt
​
 (r)  and g 
rr
​
 (r) .
Reports the throat radius and flare-out status.
Expected outputs: PNG files in plots/shape_functions/, plots/energy_conditions/, and plots/.
6.2 run_extended_analysis.py — Shadow, Plasma & Rotation Pipeline
Executes a seven-part workflow:
Static vacuum shadow: 200×200  pixel image.
Plasma series: Shadows for homogeneous plasma densities ρ=0.0,0.3,0.5,0.7 .
Rotation series: Shadows for spin parameters a=0.0,0.3,0.6,0.9 .
Kerr comparison: Synthetic Kerr shadow (a=0.9 ) versus rotating wormhole shadow.
Boundary comparison: Celestial-coordinate curves (α,β)  for static vacuum, static plasma, and rotating cases.
3-D parameter space: Surface plot of mean shadow radius R(a,ρ) .
EHT constraints: Allowed (a,ρ)  polygons for M87* and Sgr A*.
Expected outputs: PNG files in plots/shadows/, plots/comparisons/.
6.3 final_plots.py — Publication Figures
Generates seven advanced multi-panel figures saved to plots/advanced/:
01_shape_function_advanced.png — Six panels: b(r) , b 
eff
​
 (r) , flare-out ratio b 
eff
​
 /r , derivative b 
eff
′
​
  , dark matter density, and a parameter table.
02_energy_conditions_advanced.png — Six panels: ρ(r) , radial NEC, tangential NEC, SEC, pressure comparison, and a summary table.
03_shadow_radius_advanced.png — Four panels: synthetic shadow radius versus spin, versus plasma density, percent deviation from Kerr, and a comparison table.
04_deflection_advanced.png — Four panels: deflection angle versus impact parameter on linear and log-log scales, parameter dependence, and a summary table.  Note: deflection curves are phenomenological (∝1/b  with wormhole corrections), not full numerical integrations.
05_eht_advanced.png — Four panels: allowed parameter-space polygons, observational bar chart, shadow-radius contours, and an EHT summary table.
06_shadow_boundary_advanced.png — Two panels: spin dependence and plasma dependence of the shadow boundary.
07_comprehensive_summary_advanced.png — Six panels combining shape function, energy conditions, pressures, metric components, flare-out condition, and a global summary table.
7. Notebooks
notebooks/01_metric_derivation.ipynb
Purpose: Symbolic verification of the metric tensor.
Workflow: Loads config/params.yaml, instantiates DarkMatterProfile and ShapeFunction, and uses SymPy to print the closed-form expressions for g 
tt
​
   and g 
rr
​
  .  This confirms that the numerical implementation in src/metric.py matches the analytical formula (Eq. 3).
Outputs: LaTeX-formatted metric components and static PNG figures in notebooks/plots/.
8. Testing
tests/test_metric.py contains unit tests for the static metric:
test_g_tt: Asserts g 
tt
​
 (r)=−(1+Q 
2
 /r 
2
 ) .
test_g_rr: Asserts g 
rr
​
 (r)=(1−b(r)/r+Q 
2
 /r 
2
 ) 
−1
  .
test_metric_tensor: Verifies dictionary keys and angular components.
Run with:
bash
pytest tests/test_metric.py
9. Configuration
All parameters are centralized in config/params.yaml:
yaml
dark_matter:
  rho_s: 0.01          # Central density (Eq. 9)
  r_s: 1.0             # Scale radius

wormhole:
  Q: 0.0               # Electric charge
  C1: 1.0              # Integration constant (Eq. 11)

analysis:
  r_min: 0.5
  r_max: 10.0
  n_points: 1000
  r_throat: 1.0
Scripts read this file at runtime; no hard-coded astrophysical parameters appear in the analysis drivers.
10. Installation
Dependencies are listed in requirements.txt:
plain
numpy>=1.24.0
scipy>=1.10.0
sympy>=1.11.0
matplotlib>=3.7.0
pandas>=1.5.0
plotly>=5.13.0
seaborn>=0.12.0
pyyaml>=6.0
jupyter>=1.0.0
tqdm>=4.64.0
pytest>=7.0.0
imageio>=2.25.0
Install in a virtual environment:
bash
pip install -r requirements.txt
11. References
The BibTeX database in docs/references.bib contains:
Rahaman et al. (2025) — Gravitational lensing due to charged galactic wormhole, arXiv:2503.16111.  Primary reference for the metric, shape function, dark matter profile, and energy conditions.
Morris & Thorne (1988) — Wormholes in spacetime and their use for interstellar travel, Am. J. Phys. 56, 395.  Foundational traversable wormhole framework.
Teo (1998) — Rotating traversable wormholes, Phys. Rev. D 58, 024014.  Basis for the rotating extension in src/metric_rotating.py.
Sofue (2013) — Rotation curve of the Milky Way, PASJ 65, S5.  Galactic rotation-curve context.
12. Implementation Notes and Caveats
Shadow method: The code uses an effective-potential / critical-radius approximation rather than full numerical ray tracing.  Shadow boundaries are circles (or shifted circles for rotation) whose radii are set by r 
ph
​
   and the plasma scaling factor.  This is consistent with the analytical approach of the parent paper.
Synthetic comparisons: final_plots.py employs phenomenological curves for Kerr shadow radii, deflection angles, and EHT constraint regions.  These are illustrative models designed for rapid parameter-space exploration, not the output of detailed general-relativistic ray-tracing simulations.
Plasma transparency: The homogeneous plasma model requires ρ<1  for a real critical radius; the code skips shadow generation when ρ≥1 .
Geodesic integration: While the Hamiltonian and equations of motion are fully implemented, the shadow modules do not call the integrator for every pixel; instead they rely on the effective-potential minimum, which is computationally efficient and appropriate for the spherically symmetric (or slowly rotating) case.
SymPy support: ShapeFunction provides both numeric (numpy) and symbolic (sympy) evaluations, enabling exact algebraic verification in the notebook pipeline.
plain
