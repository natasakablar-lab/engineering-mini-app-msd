# Engineering Mini App — Mass–Spring–Damper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23005871.svg)](https://doi.org/10.5281/zenodo.23005871)

## Version 1.0

Interactive engineering application for modeling, analysis, PID control, disturbance rejection, and robustness assessment of a classical translational mass–spring–damper (MSD) system.

Version 1.0 is the archived release of the regression-tested final application used for the associated ACC 2027 study.

## Main features

The application is organized into five tabs:

1. **Model**
   - physical parameters `m`, `c`, and `k`
   - modeling assumptions and physical interpretation
   - governing differential equation
   - derived parameters: natural frequency, damping ratio, critical damping, and damped frequency
   - transfer-function and state-space representations
   - open-loop poles and stability classification

2. **Open-loop Analysis**
   - step response to an external force
   - analytical steady-state displacement
   - rise time, peak time, overshoot, peak value, and 2% settling time
   - special handling for monotonic and undamped responses
   - physical interpretation and unit checks

3. **PID Control**
   - Initial PID, Retuned PID, and Custom PID modes
   - derivative-on-measurement implementation
   - closed-loop characteristic polynomial and poles
   - stability classification
   - tracking response and control effort
   - protection against misleading steady-state metrics for unstable cases
   - side-by-side comparison of Initial and Retuned PID controllers

4. **Disturbance Rejection**
   - independent controller selection for the disturbance experiment
   - equilibrium-initialized step-disturbance test
   - maximum displacement deviation and recovery time
   - final tracking error and control force
   - analytical equilibrium-force check
   - Initial PID vs Retuned PID comparison

5. **Robustness**
   - fixed-controller uncertainty analysis
   - independent variation of `m`, `c`, and `k`
   - analytical Routh–Hurwitz robust-stability check
   - one-at-a-time parameter sensitivity
   - 3- or 5-level full parameter sweep
   - worst-case overshoot and settling-time detection
   - explicit distinction between robust stability and robust performance

## Software environment

The application is implemented in Python using:

- Streamlit
- NumPy
- pandas
- SciPy

The interface uses Streamlit native charts and tables. No external measured data are required.

## Files

```text
engineering_mini_app_v1_0.py   Main Streamlit application
msd_physical_sketch.png        Physical-system sketch used in the Model tab
requirements.txt               Python package requirements
README.md                      Release documentation
```

Keep `msd_physical_sketch.png` in the same directory as the Python application.

## Installation

From the project directory:

```bash
python -m pip install -r requirements.txt
```

On Windows, the following form is also convenient:

```bash
py -m pip install -r requirements.txt
```

## Running the application

```bash
python -m streamlit run engineering_mini_app_v1_0.py
```

or on Windows:

```bash
py -m streamlit run engineering_mini_app_v1_0.py
```

Streamlit opens the application in a web browser.

## Nominal model

Default physical parameters:

- `m = 1 kg`
- `c = 2 N·s/m`
- `k = 20 N/m`
- `F0 = 1 N`
- simulation time `T = 6 s`

Expected open-loop values are approximately:

- natural frequency: `4.4721 rad/s`
- damping ratio: `0.2236`
- damped frequency: `4.3589 rad/s`
- poles: `-1 ± j4.3589`
- steady-state displacement: `0.0500 m`
- overshoot: `48.64%`
- rise time (10–90%): `0.2746 s`
- peak time: `0.7202 s`
- 2% settling time: `3.781 s`

## PID benchmark

Preset controllers:

- **Initial PID:** `Kp = 31.45`, `Ki = 114.46`, `Kd = 12`
- **Retuned PID:** `Kp = 31.45`, `Ki = 90`, `Kd = 9`

For the nominal `r = 0.05 m` tracking test, the Retuned PID gives approximately:

- overshoot: `8.274%`
- rise time: `0.369 s`
- peak time: `0.801 s`
- 2% settling time: `1.292 s`
- steady-state error: approximately zero
- maximum control force: `1.5725 N`
- final control force: `1.0 N`

## Disturbance-rejection benchmark

Default disturbance test:

- reference: `0.05 m`
- disturbance: `0.5 N`
- disturbance time: `1.0 s`

Representative results:

- Initial PID: maximum deviation `6.978 mm`, recovery time `1.111 s`
- Retuned PID: maximum deviation `8.033 mm`, recovery time `1.183 s`
- both controllers remove the steady-state position error
- final control force approaches `0.5 N`, consistent with `u_ss = k r - d0`

## Robustness benchmark

Default robustness study:

- Retuned PID kept fixed
- independent `±20%` uncertainty in `m`, `c`, and `k`
- performance requirements: overshoot `≤ 10%`, settling time `≤ 2 s`, steady-state error `≤ 1e-4 m`
- 5 levels per parameter: `5^3 = 125` tested combinations

For the nominal release test:

- analytical Routh–Hurwitz robust-stability check: **PASS**
- all 125 sampled combinations: stable
- worst overshoot: approximately `17.948%` at `m = 1.2 kg`, `c = 1.6 N·s/m`, `k = 16 N/m`
- worst settling time: approximately `1.551 s` at `m = 0.8 kg`, `c = 2.4 N·s/m`, `k = 16 N/m`
- settling-time requirement: **PASS** over the sampled grid
- steady-state accuracy requirement: **PASS** over the sampled grid
- overshoot requirement: **FAIL** for some uncertain plants

The full grid sweep is an engineering robustness test. It is not a formal proof of continuous robust performance between sampled parameter combinations. The analytical Routh–Hurwitz check provides the robust-stability guarantee for the defined uncertainty box.

## Validation status

Version 1.0 corresponds to the final regression-passed release candidate. Validation included:

- open-loop edge cases, including undamped, near-critical, critically damped, and overdamped behavior
- first-local-maximum peak handling for oscillatory cases
- closed-loop pole and stability classification
- unstable custom-PID handling and suppression of invalid steady-state metrics
- disturbance-rejection consistency checks
- robustness sweep and worst-case reporting

The source file also passes Python syntax compilation.

## Scope

The current release uses a linear, one-degree-of-freedom MSD model with constant parameters and a practical PID controller. The application is intended for engineering analysis, simulation, research support, and education; it is not a real-time controller for physical hardware.

Possible future extensions include nonlinear stiffness or damping, additional degrees of freedom, actuator saturation, measurement noise, filtered derivative action, and additional control methods.
