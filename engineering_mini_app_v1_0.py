from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import TransferFunction, step, find_peaks
from scipy.integrate import solve_ivp

st.set_page_config(
    page_title="Engineering Mini App — Mass–Spring–Damper",
    page_icon="⚙️",
    layout="wide",
)

st.title("Engineering Mini App — Mass–Spring–Damper")
st.caption("Version 1.0 — physical model → open-loop analysis → PID control → disturbance rejection → robustness")

APP_DIR = Path(__file__).resolve().parent
SKETCH_PATH = APP_DIR / "msd_physical_sketch.png"

# Compact metric styling: keep the dashboard readable without oversized numbers.
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        padding-top: 0.15rem;
        padding-bottom: 0.15rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def system_classification(zeta: float) -> str:
    if np.isclose(zeta, 0.0, atol=1e-10, rtol=0.0):
        return "Undamped"
    if abs(zeta - 1.0) <= 1e-4:
        if abs(zeta - 1.0) <= 1e-10:
            return "Critically damped"
        if zeta < 1.0:
            return "Near-critical (slightly underdamped)"
        return "Near-critical (slightly overdamped)"
    if zeta < 1.0:
        return "Underdamped"
    return "Overdamped"


def step_metrics(t, y, y_final):
    result = {
        "rise_time_10_90": np.nan,
        "peak_time": np.nan,
        "overshoot_pct": np.nan,
        "settling_time_2pct": np.nan,
        "peak_value": np.nan,
        "monotonic": False,
    }

    if len(t) == 0 or len(y) == 0 or not np.isfinite(y_final):
        return result

    scale = max(abs(float(y_final)), float(np.ptp(y)), 1.0)
    monotonic_tol = 1e-8 * scale
    dy = np.diff(y)
    monotonic_non_decreasing = bool(np.all(dy >= -monotonic_tol))
    result["monotonic"] = monotonic_non_decreasing

    if monotonic_non_decreasing:
        # A critically damped / overdamped step response has no finite transient peak.
        peak_value = float(y_final)
        result["peak_value"] = peak_value
        result["peak_time"] = np.nan
    else:
        # Standard peak time is the first local maximum, not a later equal maximum.
        # This matters in the undamped case, where every peak has the same height.
        peak_indices, _ = find_peaks(y)
        if len(peak_indices):
            peak_idx = int(peak_indices[0])
        else:
            peak_idx = int(np.argmax(y))
        peak_value = float(y[peak_idx])
        result["peak_value"] = peak_value
        result["peak_time"] = float(t[peak_idx])

    if abs(y_final) > 1e-12:
        result["overshoot_pct"] = max(
            0.0, (peak_value - y_final) / abs(y_final) * 100.0
        )

        if y_final > 0:
            idx10 = np.where(y >= 0.1 * y_final)[0]
            idx90 = np.where(y >= 0.9 * y_final)[0]
            if len(idx10) and len(idx90):
                result["rise_time_10_90"] = float(t[idx90[0]] - t[idx10[0]])

        band = 0.02 * abs(y_final)
        inside = np.abs(y - y_final) <= band
        for i in range(len(t)):
            if np.all(inside[i:]):
                result["settling_time_2pct"] = float(t[i])
                break

    return result


def pid_characteristic_coeffs(m, c, k, Kp, Ki, Kd):
    """Closed-loop characteristic polynomial for the implemented PID law."""
    if abs(Ki) <= 1e-12:
        # With Ki = 0 the integral state is dynamically irrelevant to x(t).
        return [m, c + Kd, k + Kp]
    return [m, c + Kd, k + Kp, Ki]


def pole_stability_classification(poles, tol=1e-8):
    real_parts = np.real(np.asarray(poles))
    if np.any(real_parts > tol):
        return "Unstable"
    if np.any(np.abs(real_parts) <= tol):
        return "Marginal"
    return "Stable"


def fmt(value, digits=4):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}g}"


def matrix_to_latex(name, matrix, digits=4):
    rows = []
    for row in np.atleast_2d(matrix):
        rows.append(" & ".join(f"{float(v):.{digits}g}" for v in row))
    body = r" \\ ".join(rows)
    return rf"{name}=\begin{{bmatrix}}{body}\end{{bmatrix}}"


def simulate_pid_closed_loop(m, c, k, Kp, Ki, Kd, r, sim_time):
    def ode(t, state):
        x, v, z = state
        e = r - x
        u = Kp * e + Ki * z - Kd * v   # derivative on measurement
        xdot = v
        vdot = (u - c * v - k * x) / m
        zdot = e
        return [xdot, vdot, zdot]

    t_eval = np.linspace(0.0, float(sim_time), 4000)
    sol = solve_ivp(
        ode,
        (0.0, float(sim_time)),
        [0.0, 0.0, 0.0],
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )

    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    z = sol.y[2]
    e = r - x
    u = Kp * e + Ki * z - Kd * v

    char_coeffs = pid_characteristic_coeffs(m, c, k, Kp, Ki, Kd)
    poles_cl = np.roots(char_coeffs)
    stability = pole_stability_classification(poles_cl)

    metrics = step_metrics(t, x, r)
    ess = abs(r - x[-1])

    return {
        "t": t,
        "x": x,
        "v": v,
        "z": z,
        "e": e,
        "u": u,
        "metrics": metrics,
        "ess": ess,
        "char_coeffs": char_coeffs,
        "poles": poles_cl,
        "stability": stability,
        "stable": stability == "Stable",
    }




def simulate_pid_tracking_tf(m, c, k, Kp, Ki, Kd, r, sim_time, n_points=1800):
    """
    Fast reference-tracking simulation using the closed-loop transfer function
    for PID with derivative on measurement:

        X(s)/R(s) = (Kp*s + Ki) /
                    (m*s^3 + (c+Kd)*s^2 + (k+Kp)*s + Ki)
    """
    denominator = [m, c + Kd, k + Kp, Ki]
    numerator = [Kp, Ki]

    poles_cl = np.roots(denominator)
    stable = bool(np.all(np.real(poles_cl) < 0.0))

    t = np.linspace(0.0, float(sim_time), int(n_points))
    sys_cl = TransferFunction(numerator, denominator)
    t, y_unit = step(sys_cl, T=t)
    x = r * y_unit

    metrics = step_metrics(t, x, r)
    final_error = float(abs(r - x[-1]))

    return {
        "t": t,
        "x": x,
        "poles": poles_cl,
        "stable": stable,
        "metrics": metrics,
        "final_error": final_error,
    }


def simulate_disturbance_rejection(
    m, c, k, Kp, Ki, Kd, r, disturbance, disturbance_time, sim_time
):
    """
    Disturbance-rejection test initialized at the reference equilibrium.

    Plant:
        m*xdd + c*xd + k*x = u + d

    PID (derivative on measurement):
        u = Kp*(r-x) + Ki*z - Kd*xd

    For Ki > 0, z(0) is chosen so that before the disturbance:
        x(0)=r, xdot(0)=0, u(0)=k*r.
    """
    if Ki <= 0:
        raise ValueError(
            "This equilibrium-initialized disturbance test requires Ki > 0."
        )

    z0 = (k * r) / Ki

    def ode(t, state):
        x, v, z = state
        e = r - x
        u = Kp * e + Ki * z - Kd * v
        d = disturbance if t >= disturbance_time else 0.0

        xdot = v
        vdot = (u + d - c * v - k * x) / m
        zdot = e
        return [xdot, vdot, zdot]

    t_eval = np.linspace(0.0, float(sim_time), 5000)
    sol = solve_ivp(
        ode,
        (0.0, float(sim_time)),
        [r, 0.0, z0],
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )

    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    z = sol.y[2]

    e = r - x
    u = Kp * e + Ki * z - Kd * v
    d = np.where(t >= disturbance_time, disturbance, 0.0)

    post = t >= disturbance_time
    post_indices = np.where(post)[0]

    deviation = np.abs(x - r)
    max_deviation = float(np.max(deviation[post]))

    # First time after disturbance after which response remains in ±2% band.
    band = 0.02 * abs(r)
    recovery_time = np.nan
    if band > 0:
        for idx in post_indices:
            if np.all(deviation[idx:] <= band):
                recovery_time = float(t[idx] - disturbance_time)
                break

    final_error = float(abs(r - x[-1]))
    predicted_open_loop_shift = float(abs(disturbance / k))
    predicted_final_u = float(k * r - disturbance)

    return {
        "t": t,
        "x": x,
        "v": v,
        "z": z,
        "e": e,
        "u": u,
        "d": d,
        "max_deviation": max_deviation,
        "recovery_time": recovery_time,
        "final_error": final_error,
        "predicted_open_loop_shift": predicted_open_loop_shift,
        "predicted_final_u": predicted_final_u,
    }


# -------------------------------------------------
# Inputs
# -------------------------------------------------
st.sidebar.header("Physical parameters")

m = st.sidebar.number_input(
    "Mass, m [kg]",
    min_value=0.001,
    value=1.0,
    step=0.1,
    format="%.3f",
)

c = st.sidebar.number_input(
    "Damping coefficient, c [N·s/m]",
    min_value=0.0,
    value=2.0,
    step=0.1,
    format="%.3f",
)

k = st.sidebar.number_input(
    "Spring stiffness, k [N/m]",
    min_value=0.001,
    value=20.0,
    step=1.0,
    format="%.3f",
)

F0 = st.sidebar.number_input(
    "Step force, F₀ [N]",
    min_value=0.001,
    value=1.0,
    step=0.1,
    format="%.3f",
)

sim_time = st.sidebar.number_input(
    "Simulation time [s]",
    min_value=0.5,
    value=6.0,
    step=0.5,
    format="%.1f",
)

# Derived quantities
omega_n = np.sqrt(k / m)
c_crit = 2.0 * np.sqrt(k * m)
zeta = c / c_crit
omega_d = omega_n * np.sqrt(max(0.0, 1.0 - zeta**2)) if zeta < 1.0 else np.nan
poles = np.roots([m, c, k])

A = np.array([[0.0, 1.0], [-k / m, -c / m]])
B = np.array([[0.0], [1.0 / m]])
C = np.array([[1.0, 0.0]])
D = np.array([[0.0]])

# Tabs
tab_model, tab_analysis, tab_pid, tab_disturbance, tab_robustness = st.tabs(
    [
        "1. Model",
        "2. Open-loop Analysis",
        "3. PID Control",
        "4. Disturbance Rejection",
        "5. Robustness",
    ]
)

# =========================================================
# MODEL TAB
# =========================================================
with tab_model:
    st.subheader("Physical system and assumptions")

    if SKETCH_PATH.exists():
        st.image(
            str(SKETCH_PATH),
            caption="Mass–Spring–Damper physical sketch with labeled elements and signals.",
            use_container_width=True,
        )
    else:
        st.warning("Physical sketch file not found. Keep msd_physical_sketch.png in the same folder as the app.")

    st.markdown(
        r"""
The model represents a rigid mass moving in one translational direction,
connected to a linear spring and a linear viscous damper.

**Input:** external force \(F(t)\)  
**Output:** displacement \(x(t)\)
"""
    )

    with st.expander("Assumptions — Why? / What if not?", expanded=False):
        assumptions = pd.DataFrame(
            [
                ["Rigid mass", "One displacement coordinate x(t) is sufficient.", "Deformable body → additional modes / distributed-parameter model."],
                ["One-dimensional motion", "One relevant degree of freedom.", "2D/3D or rotation → multiple coordinates and coupled equations."],
                ["Linear spring", "Use Hooke law: Fₖ = kx.", "Nonlinear stiffness → Fₖ = f(x), nonlinear model."],
                ["Linear viscous damping", "Use F_c = c ẋ.", "Nonlinear damping / Coulomb friction → nonlinear or nonsmooth model."],
                ["Constant m, c, k", "Gives a time-invariant model.", "Variable parameters → LTV, LPV, or nonlinear model."],
                ["No additional friction", "Only modeled viscous dissipation is retained.", "Extra friction adds another force law."],
                ["x = 0 at equilibrium", "Dynamics are written in deviation coordinates.", "Absolute coordinates retain constant bias/equilibrium terms."],
            ],
            columns=["Assumption", "Why we use it", "If removed"],
        )
        st.dataframe(assumptions, use_container_width=True, hide_index=True)

    st.subheader("Governing equation")
    st.latex(r"m\ddot{x}(t)+c\dot{x}(t)+kx(t)=F(t)")

    st.markdown("**Physical interpretation:**")
    st.latex(
        r"\underbrace{m\ddot{x}}_{\text{inertia}}"
        r"+\underbrace{c\dot{x}}_{\text{energy dissipation}}"
        r"+\underbrace{kx}_{\text{elastic restoring force}}"
        r"=\underbrace{F(t)}_{\text{external excitation}}"
    )

    st.caption("Current numerical model")
    st.latex(
        rf"{m:.4g}\,\ddot{{x}}+{c:.4g}\,\dot{{x}}+{k:.4g}\,x=F(t)"
    )

    st.subheader("Derived dynamic parameters")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Natural frequency ωₙ", f"{omega_n:.4f} rad/s")
    c2.metric("Damping ratio ζ", f"{zeta:.4f}")
    c3.metric("Damped frequency ω_d", f"{omega_d:.4f} rad/s" if np.isfinite(omega_d) else "not oscillatory")
    c4.metric("Critical damping c_cr", f"{c_crit:.4f} N·s/m")
    st.markdown(f"**System type:** `{system_classification(zeta)}`")

    st.subheader("Transfer function")
    st.latex(r"G(s)=\frac{X(s)}{F(s)}=\frac{1}{ms^2+cs+k}")
    st.latex(rf"G(s)=\frac{{1}}{{{m:.4g}s^2+{c:.4g}s+{k:.4g}}}")

    st.subheader("State-space model")
    st.latex(r"\mathbf{x}=\begin{bmatrix}x & \dot{x}\end{bmatrix}^{T}")
    st.latex(r"\dot{\mathbf{x}}=A\mathbf{x}+B F")
    st.latex(r"y=C\mathbf{x}+D F")

    colA, colB = st.columns(2)
    with colA:
        st.latex(matrix_to_latex("A", A))
        st.latex(matrix_to_latex("B", B))
    with colB:
        st.latex(matrix_to_latex("C", C))
        st.latex(matrix_to_latex("D", D))

    st.subheader("Poles")
    pole_table = pd.DataFrame(
        {
            "Pole": [f"{p.real:.4f} {p.imag:+.4f}j" for p in poles],
            "Real part": [p.real for p in poles],
            "Imaginary part": [p.imag for p in poles],
        }
    )
    st.dataframe(pole_table, use_container_width=True, hide_index=True)

    pole_plot = pd.DataFrame(
        {
            "Real part": np.real(poles),
            "Imaginary part": np.imag(poles),
        }
    )
    st.scatter_chart(
        pole_plot,
        x="Real part",
        y="Imaginary part",
        use_container_width=True,
    )
    st.caption("Pole map in the complex plane: left-half-plane poles indicate asymptotic stability.")

    if np.all(np.real(poles) < 0):
        st.success("Open-loop plant is asymptotically stable.")
    elif np.any(np.real(poles) > 0):
        st.error("Open-loop plant is unstable.")
    else:
        st.warning("Plant has pole(s) on the imaginary axis.")

# =========================================================
# OPEN-LOOP ANALYSIS TAB
# =========================================================
with tab_analysis:
    st.subheader("Step response")

    st.markdown("A constant force step is applied:")
    st.latex(r"F(t)=F_0\,u(t)")
    st.latex(rf"F_0={F0:.4g}\,\mathrm{{N}}")

    st.markdown("At steady state:")
    st.latex(r"\dot{x}=0,\qquad \ddot{x}=0")

    st.markdown("so the spring alone balances the applied force:")
    st.latex(r"kx_{ss}=F_0\;\Rightarrow\;x_{ss}=\frac{F_0}{k}")

    x_ss = F0 / k
    t = np.linspace(0.0, float(sim_time), 4000)
    sys = TransferFunction([F0], [m, c, k])
    t, x = step(sys, T=t)
    metrics = step_metrics(t, x, x_ss)

    chart_data = pd.DataFrame(
        {
            "Time [s]": t,
            "Displacement x(t) [m]": x,
            "Steady-state x_ss [m]": np.full_like(t, x_ss),
        }
    ).set_index("Time [s]")

    st.line_chart(chart_data, use_container_width=True)
    st.caption("The simulation confirms the physical model; it does not replace the analytical reasoning.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Steady-state displacement", f"{x_ss:.5f} m")
    c2.metric("Overshoot", f"{fmt(metrics['overshoot_pct'])} %")
    if metrics["monotonic"]:
        c3.metric("Peak value", f"approaches {x_ss:.5f} m")
    else:
        c3.metric("Peak value", f"{fmt(metrics['peak_value'], 5)} m")

    c4, c5, c6 = st.columns(3)
    c4.metric("Rise time 10–90%", f"{fmt(metrics['rise_time_10_90'])} s")
    if metrics["monotonic"]:
        c5.metric("Peak time", "N/A — monotonic")
    else:
        c5.metric("Peak time", f"{fmt(metrics['peak_time'])} s")
    c6.metric("Settling time 2%", f"{fmt(metrics['settling_time_2pct'])} s")

    st.subheader("Physical reading of the response")
    st.markdown(
        f"""
- At early times, **mass inertia** governs how quickly acceleration develops.
- The spring produces the restoring force $kx$.
- The damper removes mechanical energy through $c\\dot{{x}}$.
- The final displacement is **{x_ss:.5f} m**, determined by $F_0/k$.
- Changing $m$ and $c$ changes the transient path, while $k$ also changes the static displacement.
"""
    )

    st.subheader("Model checks")
    checks = pd.DataFrame(
        [
            ["Inertia term", "m ẍ", "N"],
            ["Damping term", "c ẋ", "N"],
            ["Spring term", "k x", "N"],
            ["Input force", "F(t)", "N"],
        ],
        columns=["Term", "Expression", "Unit"],
    )
    st.dataframe(checks, use_container_width=True, hide_index=True)

# =========================================================
# PID CONTROL TAB
# =========================================================
with tab_pid:
    st.subheader("PID closed-loop control")

    st.markdown(
        "We now control the mass displacement using the actuator force **u(t)**, "
        "with reference **r(t)** and tracking error **e(t)**."
    )
    st.latex(r"e(t)=r(t)-x(t)")

    st.markdown("For a practical PID implementation, derivative action is applied to the measured motion:")
    st.latex(r"u(t)=K_P(r-x)+K_I\int (r-x)\,dt-K_D\dot{x}")

    st.markdown(r"Using the states $x$, $v=\dot{x}$ and the integral state $z$, the closed-loop model is:")
    st.latex(r"\dot{x}=v")
    st.latex(r"\dot{v}=\frac{K_P(r-x)+K_I z-(c+K_D)v-kx}{m}")
    st.latex(r"\dot{z}=r-x")
    st.latex(r"z(t)=\int_0^t [r(\tau)-x(\tau)]\,d\tau")


    st.subheader("Design specifications and reference")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Target overshoot", "≤ 10 %")
    d2.metric("Target settling time", "≤ 2 s")
    d3.metric("Target steady-state error", "0")
    d4.metric("Reference type", "step")

    r = st.number_input(
        "Reference displacement, r [m]",
        min_value=0.001,
        value=0.05,
        step=0.005,
        format="%.3f",
        key="reference_displacement",
    )

    st.subheader("Controller selection")
    mode = st.radio(
        "Choose PID gains",
        ["Initial PID", "Retuned PID", "Custom PID"],
        horizontal=True,
    )

    if mode == "Initial PID":
        Kp, Ki, Kd = 31.45, 114.46, 12.0
    elif mode == "Retuned PID":
        Kp, Ki, Kd = 31.45, 90.0, 9.0
    else:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            Kp = st.number_input("Kp", value=31.45, step=1.0, format="%.3f")
        with cc2:
            Ki = st.number_input("Ki", value=90.0, step=1.0, format="%.3f")
        with cc3:
            Kd = st.number_input("Kd", value=9.0, step=1.0, format="%.3f")

    c1, c2, c3 = st.columns(3)
    c1.metric("Kp", f"{Kp:.3f}")
    c2.metric("Ki", f"{Ki:.3f}")
    c3.metric("Kd", f"{Kd:.3f}")

    sim = simulate_pid_closed_loop(m, c, k, Kp, Ki, Kd, r, sim_time)
    t = sim["t"]
    x = sim["x"]
    e = sim["e"]
    u = sim["u"]
    metrics = sim["metrics"]
    ess = sim["ess"]

    st.subheader("Closed-loop characteristic and stability")
    if len(sim["char_coeffs"]) == 4:
        st.latex(
            rf"{m:.4g}s^3+{c+Kd:.4g}s^2+{k+Kp:.4g}s+{Ki:.4g}=0"
        )
    else:
        st.latex(rf"{m:.4g}s^2+{c+Kd:.4g}s+{k+Kp:.4g}=0")

    cl_pole_table = pd.DataFrame(
        {
            "Closed-loop pole": [
                f"{p.real:.4f} {p.imag:+.4f}j" for p in sim["poles"]
            ],
            "Real part": np.real(sim["poles"]),
            "Imaginary part": np.imag(sim["poles"]),
        }
    )
    st.dataframe(cl_pole_table, use_container_width=True, hide_index=True)

    if sim["stability"] == "Stable":
        st.success("Closed-loop system is asymptotically stable.")
    elif sim["stability"] == "Marginal":
        st.warning(
            "Closed-loop system is marginal. Standard steady-state performance metrics "
            "should be interpreted with caution."
        )
    else:
        st.error(
            "Closed-loop system is unstable. Step-response performance metrics such as "
            "settling time, steady-state error and peak time are not valid as steady-state "
            "performance measures."
        )

    st.subheader("Closed-loop tracking response")
    x_chart = pd.DataFrame(
        {
            "Time [s]": t,
            "Displacement x(t) [m]": x,
            "Reference r [m]": np.full_like(t, r),
        }
    ).set_index("Time [s]")
    st.line_chart(x_chart, use_container_width=True)

    st.subheader("Control effort")
    u_chart = pd.DataFrame(
        {
            "Time [s]": t,
            "Control input u(t) [N]": u,
        }
    ).set_index("Time [s]")
    st.line_chart(u_chart, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)
    m7, m8, m9 = st.columns(3)

    if sim["stability"] == "Stable":
        m1.metric("Steady-state error", f"{ess:.6f} m")
        m2.metric("Overshoot", f"{fmt(metrics['overshoot_pct'])} %")
        m3.metric("Peak displacement", f"{fmt(metrics['peak_value'], 5)} m")
        m4.metric("Rise time 10–90%", f"{fmt(metrics['rise_time_10_90'])} s")
        if metrics.get("monotonic", False):
            m5.metric("Peak time", "N/A — monotonic")
        else:
            m5.metric("Peak time", f"{fmt(metrics['peak_time'])} s")
        m6.metric("Settling time 2%", f"{fmt(metrics['settling_time_2pct'])} s")
        m7.metric("Max control force", f"{np.max(u):.4f} N")
        m8.metric("Min control force", f"{np.min(u):.4f} N")
        m9.metric("Final control force", f"{u[-1]:.4f} N")
    else:
        m1.metric("Steady-state error", "N/A")
        m2.metric("Overshoot", "N/A")
        m3.metric("Max displacement in window", f"{np.max(np.abs(x)):.5f} m")
        m4.metric("Rise time 10–90%", "N/A")
        m5.metric("Peak time", "N/A")
        m6.metric("Settling time 2%", "N/A")
        m7.metric("Max |control force| in window", f"{np.max(np.abs(u)):.4f} N")
        m8.metric("Min control force in window", f"{np.min(u):.4f} N")
        m9.metric("Control force at Tsim", f"{u[-1]:.4f} N")
        st.caption(
            f"Values labeled 'in window' are finite-simulation observations over 0–{sim_time:.1f} s, "
            "not steady-state performance measures."
        )

    st.subheader("Physical reading of the PID response")
    if sim["stability"] == "Stable":
        st.markdown(
            f"""
- At **t = 0**, the initial error is **{r:.4f} m**, so proportional action immediately generates force.
- Derivative action opposes motion and therefore behaves like **virtual damping**.
- Integral action builds the steady control effort required to hold the desired position.
- For the current gains, the final control force approaches **{u[-1]:.4f} N**.
"""
        )
        st.latex(r"u_D=-K_D\dot{x}")
        st.latex(rf"u_{{ss}}=kr={k:.4g}\cdot {r:.4g}={k*r:.4f}\,\mathrm{{N}}")
    else:
        st.markdown(
            f"""
- At **t = 0**, the initial error is **{r:.4f} m**, so proportional action generates force.
- The selected gains place at least one closed-loop pole on or to the right of the imaginary axis.
- The response therefore does **not** converge to the commanded equilibrium in the usual asymptotic sense.
- The displayed endpoint values at **Tsim = {sim_time:.1f} s** are simulation-window values, not steady-state values.
"""
        )
        st.latex(r"u_D=-K_D\dot{x}")
        st.caption(
            f"If the closed loop were stable at r = {r:.4f} m, the equilibrium spring force would be "
            f"kr = {k*r:.4f} N. The current controller does not converge to that equilibrium."
        )

    # Comparison of preset controllers
    st.subheader("Preset comparison: Initial PID vs Retuned PID")

    sim_initial = simulate_pid_closed_loop(m, c, k, 31.45, 114.46, 12.0, r, sim_time)
    sim_retuned = simulate_pid_closed_loop(m, c, k, 31.45, 90.0, 9.0, r, sim_time)

    compare_x = pd.DataFrame(
        {
            "Time [s]": sim_initial["t"],
            "Initial PID x(t) [m]": sim_initial["x"],
            "Retuned PID x(t) [m]": sim_retuned["x"],
            "Reference r [m]": np.full_like(sim_initial["t"], r),
        }
    ).set_index("Time [s]")
    st.line_chart(compare_x, use_container_width=True)

    compare_u = pd.DataFrame(
        {
            "Time [s]": sim_initial["t"],
            "Initial PID u(t) [N]": sim_initial["u"],
            "Retuned PID u(t) [N]": sim_retuned["u"],
        }
    ).set_index("Time [s]")
    st.line_chart(compare_u, use_container_width=True)

    summary = pd.DataFrame(
        [
            [
                "Initial PID",
                31.45, 114.46, 12.0,
                sim_initial["metrics"]["overshoot_pct"],
                sim_initial["metrics"]["settling_time_2pct"],
                sim_initial["metrics"]["rise_time_10_90"],
                sim_initial["ess"],
                np.max(sim_initial["u"]),
            ],
            [
                "Retuned PID",
                31.45, 90.0, 9.0,
                sim_retuned["metrics"]["overshoot_pct"],
                sim_retuned["metrics"]["settling_time_2pct"],
                sim_retuned["metrics"]["rise_time_10_90"],
                sim_retuned["ess"],
                np.max(sim_retuned["u"]),
            ],
        ],
        columns=[
            "Controller", "Kp", "Ki", "Kd",
            "Overshoot [%]", "Settling time [s]", "Rise time [s]",
            "Steady-state error [m]", "Max |u| [N]",
        ],
    )

    # Absolute max value for control input
    summary["Max |u| [N]"] = [
        float(np.max(np.abs(sim_initial["u"]))),
        float(np.max(np.abs(sim_retuned["u"]))),
    ]

    st.dataframe(summary, use_container_width=True, hide_index=True)


# =========================================================
# DISTURBANCE REJECTION TAB
# =========================================================
with tab_disturbance:
    st.subheader("Disturbance rejection")

    st.markdown(
        """
This test asks a different question from reference tracking:

**Can the same controller keep the mass at the desired position when an external force suddenly acts on the plant?**
"""
    )

    st.latex(r"m\ddot{x}+c\dot{x}+kx=u+d")

    st.markdown(
        "The system is initialized at the reference equilibrium. "
        "A step disturbance is then applied while the reference remains unchanged."
    )

    st.subheader("Test setup")

    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        r_d = st.number_input(
            "Reference displacement, r [m]",
            min_value=0.001,
            value=0.05,
            step=0.005,
            format="%.3f",
            key="dist_reference",
        )

    with dc2:
        disturbance = st.number_input(
            "Disturbance force, d₀ [N]",
            value=0.5,
            step=0.1,
            format="%.3f",
            key="dist_force",
        )

    max_td = max(0.1, float(sim_time) - 0.1)
    default_td = min(1.0, max_td)

    with dc3:
        disturbance_time = st.number_input(
            "Disturbance starts at [s]",
            min_value=0.0,
            max_value=max_td,
            value=default_td,
            step=0.1,
            format="%.2f",
            key="dist_time",
        )

    st.markdown("**Controller used for the disturbance test**")
    st.caption(
        "This controller selection is independent of the controller selected in the PID Control tab."
    )

    dist_mode = st.radio(
        "Choose controller",
        ["Initial PID", "Retuned PID", "Custom PID"],
        horizontal=True,
        key="dist_controller_mode",
    )

    if dist_mode == "Initial PID":
        dKp, dKi, dKd = 31.45, 114.46, 12.0
    elif dist_mode == "Retuned PID":
        dKp, dKi, dKd = 31.45, 90.0, 9.0
    else:
        dk1, dk2, dk3 = st.columns(3)
        with dk1:
            dKp = st.number_input(
                "Disturbance-test Kp",
                value=31.45,
                step=1.0,
                format="%.3f",
                key="dist_kp",
            )
        with dk2:
            dKi = st.number_input(
                "Disturbance-test Ki",
                value=90.0,
                step=1.0,
                format="%.3f",
                key="dist_ki",
            )
        with dk3:
            dKd = st.number_input(
                "Disturbance-test Kd",
                value=9.0,
                step=1.0,
                format="%.3f",
                key="dist_kd",
            )

    g1, g2, g3 = st.columns(3)
    g1.metric("Kp", f"{dKp:.3f}")
    g2.metric("Ki", f"{dKi:.3f}")
    g3.metric("Kd", f"{dKd:.3f}")

    if dKi <= 0:
        st.error(
            "For this clean equilibrium-initialized test, Ki must be greater than zero. "
            "Choose a preset PID or use Custom PID with Ki > 0."
        )
    else:
        dist_sim = simulate_disturbance_rejection(
            m, c, k,
            dKp, dKi, dKd,
            r_d,
            disturbance,
            disturbance_time,
            sim_time,
        )

        st.subheader("Position response to the external disturbance")

        dist_x_chart = pd.DataFrame(
            {
                "Time [s]": dist_sim["t"],
                "Position x(t) [m]": dist_sim["x"],
                "Reference r [m]": np.full_like(dist_sim["t"], r_d),
            }
        ).set_index("Time [s]")

        st.line_chart(dist_x_chart, use_container_width=True)

        st.subheader("Controller action and disturbance force")

        dist_u_chart = pd.DataFrame(
            {
                "Time [s]": dist_sim["t"],
                "Control input u(t) [N]": dist_sim["u"],
                "External disturbance d(t) [N]": dist_sim["d"],
            }
        ).set_index("Time [s]")

        st.line_chart(dist_u_chart, use_container_width=True)

        dm1, dm2, dm3 = st.columns(3)
        dm1.metric(
            "Maximum position deviation",
            f"{1000.0 * dist_sim['max_deviation']:.3f} mm",
        )
        dm2.metric(
            "Recovery time (2% band)",
            f"{fmt(dist_sim['recovery_time'])} s",
        )
        dm3.metric(
            "Final tracking error",
            f"{dist_sim['final_error']:.8f} m",
        )

        dm4, dm5, dm6 = st.columns(3)
        dm4.metric(
            "Final control force",
            f"{dist_sim['u'][-1]:.4f} N",
        )
        dm5.metric(
            "Predicted final control force",
            f"{dist_sim['predicted_final_u']:.4f} N",
        )
        dm6.metric(
            "Open-loop static shift",
            f"{1000.0 * dist_sim['predicted_open_loop_shift']:.2f} mm",
        )

        st.subheader("Physical interpretation")

        st.markdown(
            f"""
Before the disturbance, the mass is already at the desired position **{r_d:.3f} m**.
The controller therefore supplies the equilibrium force required to hold the spring.

When the external force **{disturbance:.3f} N** appears:

- the mass first moves away from the reference,
- proportional action reacts to the new position error,
- derivative action reacts to the motion and adds damping,
- integral action changes the steady control effort until the error is removed.

Without feedback, the same constant disturbance would produce a static displacement change of approximately **{1000.0 * dist_sim['predicted_open_loop_shift']:.2f} mm**.
"""
        )
        st.markdown("At the new equilibrium:")
        st.latex(r"kr=u_{ss}+d_0")
        st.latex(r"u_{ss}=kr-d_0")
        st.markdown(
            f"For the current values, the predicted final controller force is **{dist_sim['predicted_final_u']:.4f} N**."
        )

        st.subheader("Initial PID vs Retuned PID — disturbance comparison")

        dist_initial = simulate_disturbance_rejection(
            m, c, k,
            31.45, 114.46, 12.0,
            r_d,
            disturbance,
            disturbance_time,
            sim_time,
        )

        dist_retuned = simulate_disturbance_rejection(
            m, c, k,
            31.45, 90.0, 9.0,
            r_d,
            disturbance,
            disturbance_time,
            sim_time,
        )

        compare_dist = pd.DataFrame(
            {
                "Time [s]": dist_initial["t"],
                "Initial PID x(t) [m]": dist_initial["x"],
                "Retuned PID x(t) [m]": dist_retuned["x"],
                "Reference r [m]": np.full_like(dist_initial["t"], r_d),
            }
        ).set_index("Time [s]")

        st.line_chart(compare_dist, use_container_width=True)

        dist_summary = pd.DataFrame(
            [
                [
                    "Initial PID",
                    31.45,
                    114.46,
                    12.0,
                    1000.0 * dist_initial["max_deviation"],
                    dist_initial["recovery_time"],
                    dist_initial["final_error"],
                    dist_initial["u"][-1],
                ],
                [
                    "Retuned PID",
                    31.45,
                    90.0,
                    9.0,
                    1000.0 * dist_retuned["max_deviation"],
                    dist_retuned["recovery_time"],
                    dist_retuned["final_error"],
                    dist_retuned["u"][-1],
                ],
            ],
            columns=[
                "Controller",
                "Kp",
                "Ki",
                "Kd",
                "Max deviation [mm]",
                "Recovery time [s]",
                "Final error [m]",
                "Final u [N]",
            ],
        )

        st.dataframe(
            dist_summary,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "A controller that is best for reference tracking is not necessarily "
            "the best for disturbance rejection. This trade-off is one reason "
            "PID tuning must be evaluated against the full control objective."
        )



# =========================================================
# ROBUSTNESS TAB
# =========================================================
with tab_robustness:
    st.subheader("Robustness to plant-parameter uncertainty")

    st.markdown(
        """
The controller is now **kept fixed** while the real plant parameters are changed.

This test asks:

**Does the same PID remain stable and preserve acceptable performance when the real mass, damping, and spring stiffness differ from the nominal model?**
"""
    )

    st.latex(r"m\ddot{x}+c\dot{x}+kx=u")
    st.latex(r"u=K_P(r-x)+K_I\int(r-x)\,dt-K_D\dot{x}")

    st.subheader("Nominal plant and fixed controller")

    rc1, rc2 = st.columns(2)

    with rc1:
        robustness_controller = st.radio(
            "Controller used for the robustness test",
            ["Retuned PID", "Initial PID", "Custom PID"],
            horizontal=True,
            key="robust_controller_mode",
        )

    if robustness_controller == "Retuned PID":
        rKp, rKi, rKd = 31.45, 90.0, 9.0
    elif robustness_controller == "Initial PID":
        rKp, rKi, rKd = 31.45, 114.46, 12.0
    else:
        rk1, rk2, rk3 = st.columns(3)
        with rk1:
            rKp = st.number_input(
                "Robustness Kp",
                value=31.45,
                step=1.0,
                format="%.3f",
                key="robust_kp",
            )
        with rk2:
            rKi = st.number_input(
                "Robustness Ki",
                value=90.0,
                step=1.0,
                format="%.3f",
                key="robust_ki",
            )
        with rk3:
            rKd = st.number_input(
                "Robustness Kd",
                value=9.0,
                step=1.0,
                format="%.3f",
                key="robust_kd",
            )

    with rc2:
        robust_reference = st.number_input(
            "Reference displacement for robustness test, r [m]",
            min_value=0.001,
            value=0.05,
            step=0.005,
            format="%.3f",
            key="robust_reference",
        )

    rg1, rg2, rg3 = st.columns(3)
    rg1.metric("Fixed Kp", f"{rKp:.3f}")
    rg2.metric("Fixed Ki", f"{rKi:.3f}")
    rg3.metric("Fixed Kd", f"{rKd:.3f}")

    st.subheader("Uncertainty range and performance requirements")

    ru1, ru2, ru3, ru4 = st.columns(4)

    with ru1:
        uncertainty_pct = st.number_input(
            "Parameter uncertainty ± [%]",
            min_value=0.0,
            max_value=80.0,
            value=20.0,
            step=5.0,
            format="%.1f",
            key="uncertainty_pct",
        )

    with ru2:
        target_os = st.number_input(
            "Maximum allowed overshoot [%]",
            min_value=0.0,
            value=10.0,
            step=1.0,
            format="%.1f",
            key="robust_target_os",
        )

    with ru3:
        target_ts = st.number_input(
            "Maximum allowed settling time [s]",
            min_value=0.1,
            value=2.0,
            step=0.1,
            format="%.2f",
            key="robust_target_ts",
        )

    with ru4:
        ess_tol = st.number_input(
            "Steady-state error tolerance [m]",
            min_value=0.0,
            value=0.0001,
            step=0.0001,
            format="%.5f",
            key="robust_ess_tol",
        )

    delta = uncertainty_pct / 100.0

    m_min, m_max = m * (1.0 - delta), m * (1.0 + delta)
    c_min, c_max = c * (1.0 - delta), c * (1.0 + delta)
    k_min, k_max = k * (1.0 - delta), k * (1.0 + delta)

    ranges = pd.DataFrame(
        [
            ["m [kg]", m, m_min, m_max],
            ["c [N·s/m]", c, c_min, c_max],
            ["k [N/m]", k, k_min, k_max],
        ],
        columns=["Parameter", "Nominal", "Minimum", "Maximum"],
    )
    st.dataframe(ranges, use_container_width=True, hide_index=True)

    # -----------------------------
    # Analytical robust-stability check for the cubic
    # -----------------------------
    st.subheader("Analytical robust-stability check")

    # Characteristic polynomial:
    # m s^3 + (c+Kd) s^2 + (k+Kp) s + Ki = 0
    # Routh-Hurwitz worst corner:
    # (c_min+Kd)(k_min+Kp) > m_max Ki
    lhs = (c_min + rKd) * (k_min + rKp)
    rhs = m_max * rKi
    routh_margin = lhs - rhs

    st.latex(
        r"m s^3+(c+K_D)s^2+(k+K_P)s+K_I=0"
    )
    st.latex(
        r"(c_{\min}+K_D)(k_{\min}+K_P)>m_{\max}K_I"
    )

    rs1, rs2, rs3 = st.columns(3)
    rs1.metric("Worst-case LHS", f"{lhs:.3f}")
    rs2.metric("Worst-case RHS", f"{rhs:.3f}")
    rs3.metric("Routh margin", f"{routh_margin:.3f}")

    if (
        m_min > 0
        and c_min + rKd > 0
        and k_min + rKp > 0
        and rKi > 0
        and lhs > rhs
    ):
        st.success(
            "Analytical Routh–Hurwitz check: robust stability PASS for the defined parameter box."
        )
        analytical_stability_pass = True
    else:
        st.error(
            "Analytical Routh–Hurwitz check: robust stability is NOT guaranteed for the defined parameter box."
        )
        analytical_stability_pass = False

    # -----------------------------
    # One-at-a-time sensitivity to plant parameters
    # -----------------------------
    st.subheader("One-at-a-time plant-parameter variations")

    variation_rows = []

    cases = [
        ("m − uncertainty", m_min, c, k),
        ("m nominal", m, c, k),
        ("m + uncertainty", m_max, c, k),
        ("c − uncertainty", m, c_min, k),
        ("c nominal", m, c, k),
        ("c + uncertainty", m, c_max, k),
        ("k − uncertainty", m, c, k_min),
        ("k nominal", m, c, k),
        ("k + uncertainty", m, c, k_max),
    ]

    for name, mv, cv, kv in cases:
        res = simulate_pid_tracking_tf(
            mv, cv, kv,
            rKp, rKi, rKd,
            robust_reference,
            sim_time,
            n_points=1600,
        )

        os_val = res["metrics"]["overshoot_pct"]
        ts_val = res["metrics"]["settling_time_2pct"]
        tr_val = res["metrics"]["rise_time_10_90"]

        performance_pass = (
            res["stable"]
            and np.isfinite(os_val)
            and os_val <= target_os
            and np.isfinite(ts_val)
            and ts_val <= target_ts
            and res["final_error"] <= ess_tol
        )

        variation_rows.append(
            [
                name,
                mv,
                cv,
                kv,
                "PASS" if res["stable"] else "FAIL",
                os_val,
                ts_val,
                tr_val,
                res["final_error"],
                "PASS" if performance_pass else "FAIL",
            ]
        )

    variation_df = pd.DataFrame(
        variation_rows,
        columns=[
            "Case",
            "m",
            "c",
            "k",
            "Stability",
            "Overshoot [%]",
            "Settling time [s]",
            "Rise time [s]",
            "Final error [m]",
            "Performance",
        ],
    )

    st.dataframe(
        variation_df,
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------
    # Full grid sweep
    # -----------------------------
    st.subheader("Full parameter sweep and worst-case detection")

    sweep_levels = st.selectbox(
        "Sweep resolution per parameter",
        [3, 5],
        index=1,
        help="3 levels = 27 combinations; 5 levels = 125 combinations.",
        key="robust_sweep_levels",
    )

    st.caption(
        "The full sweep checks combinations of m, c, and k simultaneously. "
        "The PID gains remain unchanged."
    )

    if st.button("Run robustness sweep", type="primary", key="run_robustness"):
        m_values = np.linspace(m_min, m_max, int(sweep_levels))
        c_values = np.linspace(c_min, c_max, int(sweep_levels))
        k_values = np.linspace(k_min, k_max, int(sweep_levels))

        sweep_rows = []

        total_cases = len(m_values) * len(c_values) * len(k_values)

        progress = st.progress(0.0)
        counter = 0

        for mv in m_values:
            for cv in c_values:
                for kv in k_values:
                    res = simulate_pid_tracking_tf(
                        mv, cv, kv,
                        rKp, rKi, rKd,
                        robust_reference,
                        sim_time,
                        n_points=1200,
                    )

                    os_val = res["metrics"]["overshoot_pct"]
                    ts_val = res["metrics"]["settling_time_2pct"]
                    tr_val = res["metrics"]["rise_time_10_90"]

                    os_pass = (
                        res["stable"]
                        and np.isfinite(os_val)
                        and os_val <= target_os
                    )
                    ts_pass = (
                        res["stable"]
                        and np.isfinite(ts_val)
                        and ts_val <= target_ts
                    )
                    ess_pass = (
                        res["stable"]
                        and res["final_error"] <= ess_tol
                    )

                    sweep_rows.append(
                        {
                            "m": float(mv),
                            "c": float(cv),
                            "k": float(kv),
                            "Stable": res["stable"],
                            "Overshoot [%]": os_val,
                            "Settling time [s]": ts_val,
                            "Rise time [s]": tr_val,
                            "Final error [m]": res["final_error"],
                            "Overshoot PASS": os_pass,
                            "Settling PASS": ts_pass,
                            "Error PASS": ess_pass,
                        }
                    )

                    counter += 1
                    progress.progress(counter / total_cases)

        progress.empty()

        sweep_df = pd.DataFrame(sweep_rows)

        st.info(f"Full grid sweep completed: {total_cases} combinations tested.")

        stable_all = bool(sweep_df["Stable"].all())
        overshoot_all = bool(sweep_df["Overshoot PASS"].all())
        settling_all = bool(sweep_df["Settling PASS"].all())
        error_all = bool(sweep_df["Error PASS"].all())

        finite_os = sweep_df[np.isfinite(sweep_df["Overshoot [%]"])].copy()
        finite_ts = sweep_df[np.isfinite(sweep_df["Settling time [s]"])].copy()

        if len(finite_os):
            worst_os_row = finite_os.loc[finite_os["Overshoot [%]"].idxmax()]
        else:
            worst_os_row = None

        if len(finite_ts):
            worst_ts_row = finite_ts.loc[finite_ts["Settling time [s]"].idxmax()]
        else:
            worst_ts_row = None

        st.subheader("Robustness verdict")

        rv1, rv2, rv3, rv4 = st.columns(4)
        rv1.metric("Robust stability", "PASS" if stable_all else "FAIL")
        rv2.metric("Overshoot requirement", "PASS" if overshoot_all else "FAIL")
        rv3.metric("Settling-time requirement", "PASS" if settling_all else "FAIL")
        rv4.metric("Steady-state accuracy", "PASS" if error_all else "FAIL")

        if stable_all and overshoot_all and settling_all and error_all:
            st.success(
                "The tested parameter grid is robustly stable and performance-compliant."
            )
        elif stable_all:
            st.warning(
                "The tested parameter grid is robustly stable, but at least one performance requirement is violated."
            )
        else:
            st.error(
                "At least one tested parameter combination is unstable."
            )

        # Worst overshoot
        if worst_os_row is not None:
            st.markdown("**Worst overshoot found**")
            wo1, wo2, wo3, wo4 = st.columns(4)
            wo1.metric("m [kg]", f"{worst_os_row['m']:.4f}")
            wo2.metric("c [N·s/m]", f"{worst_os_row['c']:.4f}")
            wo3.metric("k [N/m]", f"{worst_os_row['k']:.4f}")
            wo4.metric(
                "Overshoot",
                f"{worst_os_row['Overshoot [%]']:.3f} %",
            )

        # Worst settling time
        if worst_ts_row is not None:
            st.markdown("**Worst settling time found**")
            wt1, wt2, wt3, wt4 = st.columns(4)
            wt1.metric("m [kg]", f"{worst_ts_row['m']:.4f}")
            wt2.metric("c [N·s/m]", f"{worst_ts_row['c']:.4f}")
            wt3.metric("k [N/m]", f"{worst_ts_row['k']:.4f}")
            wt4.metric(
                "Settling time",
                f"{worst_ts_row['Settling time [s]']:.3f} s",
            )

        # Plot nominal and worst-overshoot responses
        nominal_res = simulate_pid_tracking_tf(
            m, c, k,
            rKp, rKi, rKd,
            robust_reference,
            sim_time,
            n_points=1800,
        )

        comparison_data = {
            "Time [s]": nominal_res["t"],
            "Nominal x(t) [m]": nominal_res["x"],
            "Reference r [m]": np.full_like(
                nominal_res["t"], robust_reference
            ),
        }

        if worst_os_row is not None:
            worst_os_res = simulate_pid_tracking_tf(
                float(worst_os_row["m"]),
                float(worst_os_row["c"]),
                float(worst_os_row["k"]),
                rKp, rKi, rKd,
                robust_reference,
                sim_time,
                n_points=1800,
            )
            comparison_data["Worst-overshoot x(t) [m]"] = worst_os_res["x"]

        comparison_df = pd.DataFrame(comparison_data).set_index("Time [s]")

        st.subheader("Nominal vs worst-case response")
        st.line_chart(
            comparison_df,
            use_container_width=True,
        )

        st.subheader("Sweep results")
        sweep_display = sweep_df.sort_values(
            by=["Overshoot [%]", "Settling time [s]"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
        st.caption(
            "Results are sorted by overshoot (highest first), so the most demanding cases are visible immediately."
        )
        st.dataframe(
            sweep_display,
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        st.markdown(
            """
**Interpretation:** robust stability means that the closed-loop system remains stable
throughout the tested uncertainty set. Robust performance is stronger: the system must
also preserve the specified transient and steady-state requirements.

A grid sweep is an engineering robustness test, not a formal proof that no worse point
exists between the sampled parameter combinations.
"""
        )


st.divider()
st.caption(
    "Version 1.0 release — edge-case peak handling, closed-loop pole/stability checks, "
    "unstable-metric protection, disturbance-rejection analysis, and robustness-sweep reporting."
)
