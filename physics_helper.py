#!/usr/bin/env python3
"""
Physics Helper — algebraic and calculus-based physics solver.

Usage:
  python physics_helper.py              # interactive menu
  python physics_helper.py solve        # equation solver
  python physics_helper.py calc         # calculus tools
  python physics_helper.py formulas     # browse formula sheets
"""

import sys
import re
from sympy import (
    symbols, solve, diff, integrate, sqrt, pi, oo, simplify,
    latex, pretty, sympify, Rational, Symbol, Function,
    cos, sin, tan, exp, ln, Abs, limit
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


# ─────────────────────────────────────────────
# Formula library
# ─────────────────────────────────────────────

FORMULAS = {
    "Kinematics": {
        "vf = vi + a*t":                  "Final velocity from initial velocity, acceleration, time",
        "d = vi*t + (1/2)*a*t**2":        "Displacement with constant acceleration",
        "vf**2 = vi**2 + 2*a*d":          "Velocity-displacement relation (no time)",
        "d = (vi + vf)/2 * t":            "Displacement via average velocity",
        "d = vf*t - (1/2)*a*t**2":        "Displacement (final velocity form)",
        "x = x0 + v0x*t":                 "Horizontal projectile position",
        "y = y0 + v0y*t - (1/2)*g*t**2":  "Vertical projectile position",
    },
    "Newton's Laws & Forces": {
        "F = m*a":                  "Newton's second law",
        "Fg = m*g":                 "Gravitational force (weight)",
        "Ff = mu*N":                "Kinetic/static friction",
        "Fnet = F1 + F2 + ...":     "Net force (vector sum)",
        "p = m*v":                  "Linear momentum",
        "J = F*t":                  "Impulse",
        "J = delta_p":              "Impulse-momentum theorem",
    },
    "Energy & Work": {
        "W = F*d*cos(theta)":       "Work done by a force",
        "KE = (1/2)*m*v**2":        "Kinetic energy",
        "PE_grav = m*g*h":          "Gravitational potential energy",
        "PE_spring = (1/2)*k*x**2": "Elastic/spring potential energy",
        "ME = KE + PE":             "Mechanical energy",
        "W_net = delta_KE":         "Work-energy theorem",
        "P = W/t":                  "Power (average)",
        "P = F*v":                  "Instantaneous power",
    },
    "Circular & Rotational Motion": {
        "ac = v**2/r":              "Centripetal acceleration",
        "Fc = m*v**2/r":            "Centripetal force",
        "v = r*omega":              "Tangential velocity",
        "omega = 2*pi*f":           "Angular velocity from frequency",
        "T = 1/f":                  "Period from frequency",
        "tau = r*F*sin(theta)":     "Torque",
        "L = I*omega":              "Angular momentum",
        "I_disk = (1/2)*m*r**2":    "Moment of inertia — solid disk",
        "I_sphere = (2/5)*m*r**2":  "Moment of inertia — solid sphere",
        "I_ring = m*r**2":          "Moment of inertia — ring/hoop",
    },
    "Gravitation": {
        "Fg = G*m1*m2/r**2":        "Newton's law of gravitation",
        "g = G*M/r**2":             "Gravitational field strength",
        "T**2 = (4*pi**2/G*M)*r**3":"Kepler's third law",
        "Ug = -G*m1*m2/r":          "Gravitational potential energy",
        "v_esc = sqrt(2*G*M/r)":    "Escape velocity",
        "v_orbit = sqrt(G*M/r)":    "Orbital velocity",
    },
    "Waves & Oscillations": {
        "v = f*lambda":             "Wave speed",
        "T = 1/f":                  "Period-frequency relation",
        "v_sound ≈ 343 m/s (20°C)": "Speed of sound in air",
        "fn = n*v/(2*L)":           "Harmonics — string / open pipe",
        "fn = n*v/(4*L)":           "Harmonics — closed pipe (odd n only)",
        "x(t) = A*cos(omega*t+phi)": "SHM position",
        "omega = sqrt(k/m)":         "SHM angular frequency — spring",
        "T_spring = 2*pi*sqrt(m/k)": "Period — mass-spring",
        "T_pendulum = 2*pi*sqrt(L/g)":"Period — simple pendulum",
        "E_SHM = (1/2)*k*A**2":     "Total energy in SHM",
    },
    "Thermodynamics": {
        "PV = nRT":                 "Ideal gas law",
        "P1*V1/T1 = P2*V2/T2":     "Combined gas law",
        "Q = m*c*delta_T":          "Heat transfer",
        "Q = m*L":                  "Latent heat",
        "delta_U = Q - W":          "First law of thermodynamics",
        "e = 1 - Tc/Th":            "Carnot efficiency",
        "W = P*delta_V":            "Work done by gas",
        "KE_avg = (3/2)*k_B*T":     "Average kinetic energy of gas molecule",
    },
    "Electrostatics": {
        "F = k*q1*q2/r**2":         "Coulomb's law",
        "E = k*q/r**2":             "Electric field (point charge)",
        "V = k*q/r":                "Electric potential (point charge)",
        "W = q*delta_V":            "Work to move charge",
        "C = Q/V":                  "Capacitance",
        "U_cap = (1/2)*C*V**2":     "Energy stored in capacitor",
        "E_field = V/d":            "Uniform field between plates",
    },
    "Circuits": {
        "V = I*R":                  "Ohm's law",
        "P = I*V = I**2*R = V**2/R": "Electrical power",
        "R_series = R1+R2+...":     "Series resistance",
        "1/R_parallel = 1/R1+1/R2": "Parallel resistance",
        "Q = I*t":                  "Charge from current and time",
        "tau = R*C":                "RC time constant",
        "tau_L = L/R":              "RL time constant",
        "X_C = 1/(omega*C)":        "Capacitive reactance",
        "X_L = omega*L":            "Inductive reactance",
    },
    "Optics": {
        "1/f = 1/do + 1/di":        "Thin lens / mirror equation",
        "M = -di/do":               "Magnification",
        "n1*sin(t1) = n2*sin(t2)":  "Snell's law",
        "c = lambda*f":             "Light wave speed",
        "E = h*f":                  "Photon energy",
        "p = h/lambda":             "De Broglie wavelength",
    },
    "Special Relativity": {
        "gamma = 1/sqrt(1-(v/c)**2)":"Lorentz factor",
        "t' = gamma*t0":             "Time dilation",
        "L = L0/gamma":              "Length contraction",
        "E = gamma*m*c**2":          "Relativistic energy",
        "E**2 = (pc)**2+(mc**2)**2": "Energy-momentum relation",
        "p = gamma*m*v":             "Relativistic momentum",
    },
}

CONSTANTS = {
    "g":   ("9.80665",  "m/s²",  "Standard gravity"),
    "G":   ("6.674e-11","N m²/kg²","Gravitational constant"),
    "c":   ("2.998e8",  "m/s",   "Speed of light"),
    "h":   ("6.626e-34","J·s",   "Planck's constant"),
    "k_B": ("1.381e-23","J/K",   "Boltzmann constant"),
    "N_A": ("6.022e23", "mol⁻¹", "Avogadro's number"),
    "R":   ("8.314",    "J/(mol·K)","Ideal gas constant"),
    "e":   ("1.602e-19","C",     "Elementary charge"),
    "k":   ("8.988e9",  "N m²/C²","Coulomb's constant"),
    "eps0":("8.854e-12","C²/(N m²)","Permittivity of free space"),
    "mu0": ("1.257e-6", "T·m/A", "Permeability of free space"),
    "m_e": ("9.109e-31","kg",    "Electron mass"),
    "m_p": ("1.673e-27","kg",    "Proton mass"),
    "m_n": ("1.675e-27","kg",    "Neutron mass"),
}


# ─────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────

def parse(expr_str: str):
    """
    Parse a math expression string into a SymPy expression.

    Multi-character identifiers (vi, vf, m1, k_B …) are preserved as single
    symbols by pre-populating the local_dict before handing off to SymPy's
    parser.  Implicit multiplication is intentionally NOT used because it
    would split 'vi' into v*i (imaginary unit).
    """
    # Build a local dict so every bare identifier becomes a Symbol,
    # overriding SymPy's built-ins only for unknown names.
    known_builtins = {
        "sqrt", "sin", "cos", "tan", "exp", "ln", "log",
        "pi", "oo", "abs", "Abs", "diff", "integrate", "limit",
        "E", "I",            # SymPy constants — keep as-is
    }
    tokens = re.findall(r"[A-Za-z_]\w*", expr_str)
    local_dict = {t: Symbol(t) for t in tokens if t not in known_builtins}
    return parse_expr(expr_str,
                      transformations=standard_transformations,
                      local_dict=local_dict)


def _collect_symbols(expr_str: str) -> list[str]:
    """Return sorted list of unique symbol names found in an expression string."""
    tokens = re.findall(r"[A-Za-z_]\w*", expr_str)
    ignore = {"sqrt", "sin", "cos", "tan", "exp", "ln", "log", "pi",
              "oo", "abs", "Abs", "diff", "integrate", "limit", "E", "I"}
    return sorted(set(t for t in tokens if t not in ignore))


# ─────────────────────────────────────────────
# Core solvers
# ─────────────────────────────────────────────

def solve_equation(equation_str: str, solve_for: str, known: dict[str, str]) -> str:
    """
    Solve an equation for one unknown, substituting known values.

    equation_str  — e.g. "vf = vi + a*t"   or   "vf - vi - a*t"
    solve_for     — symbol name to isolate
    known         — {symbol_name: value_string}
    """
    if "=" in equation_str:
        lhs, rhs = equation_str.split("=", 1)
        expr = parse(lhs.strip()) - parse(rhs.strip())
    else:
        expr = parse(equation_str.strip())

    target = Symbol(solve_for)

    # substitute known values
    subs = {Symbol(k): parse(v) for k, v in known.items()}
    expr_sub = expr.subs(subs)

    solutions = solve(expr_sub, target)
    if not solutions:
        return f"No solution found for {solve_for}."

    lines = [f"\nSolving for {solve_for}:"]
    for i, sol in enumerate(solutions):
        simplified = simplify(sol)
        lines.append(f"  {solve_for} = {simplified}")
        if len(solutions) > 1:
            lines[-1] = f"  Solution {i+1}: {solve_for} = {simplified}"
    return "\n".join(lines)


def differentiate(expr_str: str, var: str, order: int = 1) -> str:
    """Differentiate an expression with respect to a variable."""
    expr = parse(expr_str)
    x = Symbol(var)
    result = diff(expr, x, order)
    simplified = simplify(result)
    order_str = {1: "", 2: "²", 3: "³"}.get(order, f"^{order}")
    return (
        f"\nd{order_str}/d{var}{order_str} [{expr_str}]\n"
        f"  = {simplified}"
    )


def integrate_expr(expr_str: str, var: str,
                   lower: str = None, upper: str = None) -> str:
    """Integrate an expression, definite or indefinite."""
    expr = parse(expr_str)
    x = Symbol(var)

    if lower is not None and upper is not None:
        lo = parse(lower) if lower not in ("-oo", "-inf") else -oo
        hi = parse(upper) if upper not in ("oo", "inf") else oo
        result = integrate(expr, (x, lo, hi))
        simplified = simplify(result)
        return (
            f"\n∫ from {lower} to {upper}  [{expr_str}] d{var}\n"
            f"  = {simplified}"
        )
    else:
        result = integrate(expr, x)
        simplified = simplify(result)
        return (
            f"\n∫ [{expr_str}] d{var}\n"
            f"  = {simplified} + C"
        )


def compute_limit(expr_str: str, var: str, point: str) -> str:
    """Compute the limit of an expression."""
    expr = parse(expr_str)
    x = Symbol(var)
    pt = parse(point) if point not in ("oo", "inf") else oo
    if point in ("-oo", "-inf"):
        pt = -oo
    result = limit(expr, x, pt)
    simplified = simplify(result)
    return (
        f"\nlim ({var} → {point})  [{expr_str}]\n"
        f"  = {simplified}"
    )


# ─────────────────────────────────────────────
# Interactive UI helpers
# ─────────────────────────────────────────────

def hr(char="─", width=60):
    print(char * width)


def header(title: str):
    hr()
    print(f"  {title}")
    hr()


def menu(title: str, options: list[str]) -> str:
    header(title)
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print("  [0] Back / Exit")
    hr()
    return input("  Choice: ").strip()


# ─────────────────────────────────────────────
# Interactive modes
# ─────────────────────────────────────────────

def mode_formulas():
    categories = list(FORMULAS.keys())
    while True:
        choice = menu("Formula Sheets", categories)
        if choice == "0":
            return
        try:
            idx = int(choice) - 1
            cat = categories[idx]
        except (ValueError, IndexError):
            print("  Invalid choice.\n")
            continue

        header(cat)
        for formula, desc in FORMULAS[cat].items():
            print(f"  {formula}")
            print(f"      {desc}")
        print()
        input("  Press Enter to continue...")


def mode_constants():
    header("Physical Constants")
    col = max(len(k) for k in CONSTANTS)
    for sym, (val, unit, name) in CONSTANTS.items():
        print(f"  {sym:<{col}}  =  {val:>12}  {unit:<14}  ({name})")
    print()
    input("  Press Enter to continue...")


def mode_solve():
    header("Equation Solver")
    print("  Enter an equation to solve for one unknown.")
    print("  Examples:")
    print("    Equation : vf = vi + a*t")
    print("    Solve for: vf")
    print("    Known    : vi=0, a=9.8, t=3")
    print()

    eq = input("  Equation : ").strip()
    if not eq:
        return

    sv = input("  Solve for: ").strip()
    if not sv:
        return

    known_raw = input("  Known (e.g. vi=0, a=9.8, t=3): ").strip()
    known = {}
    if known_raw:
        for part in known_raw.split(","):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                known[k.strip()] = v.strip()

    try:
        result = solve_equation(eq, sv, known)
        print(result)
    except Exception as exc:
        print(f"\n  Error: {exc}")
    print()
    input("  Press Enter to continue...")


def mode_calc():
    while True:
        choice = menu("Calculus Tools", [
            "Differentiate",
            "Integrate (indefinite)",
            "Integrate (definite)",
            "Compute limit",
        ])

        if choice == "0":
            return
        elif choice == "1":
            header("Differentiate")
            expr = input("  Expression: ").strip()
            var  = input("  Variable  : ").strip() or "x"
            ord_ = input("  Order (default 1): ").strip() or "1"
            try:
                print(differentiate(expr, var, int(ord_)))
            except Exception as exc:
                print(f"\n  Error: {exc}")
        elif choice == "2":
            header("Indefinite Integral")
            expr = input("  Expression: ").strip()
            var  = input("  Variable  : ").strip() or "x"
            try:
                print(integrate_expr(expr, var))
            except Exception as exc:
                print(f"\n  Error: {exc}")
        elif choice == "3":
            header("Definite Integral")
            expr  = input("  Expression   : ").strip()
            var   = input("  Variable     : ").strip() or "x"
            lower = input("  Lower bound  : ").strip()
            upper = input("  Upper bound  : ").strip()
            try:
                print(integrate_expr(expr, var, lower, upper))
            except Exception as exc:
                print(f"\n  Error: {exc}")
        elif choice == "4":
            header("Limit")
            expr  = input("  Expression  : ").strip()
            var   = input("  Variable    : ").strip() or "x"
            point = input("  Approaching : ").strip()
            try:
                print(compute_limit(expr, var, point))
            except Exception as exc:
                print(f"\n  Error: {exc}")
        else:
            print("  Invalid choice.\n")
            continue

        print()
        input("  Press Enter to continue...")


def mode_quick():
    """Quick solve: auto-detect unknowns from an equation."""
    header("Quick Solve (auto-detect unknowns)")
    print("  Enter an equation with one unknown and known values substituted.")
    print("  Example: vf = 0 + 9.8*3")
    print("  Or with one symbol: KE = (1/2)*5*v**2, solve for v, KE=100")
    print()

    eq = input("  Equation : ").strip()
    if not eq:
        return

    if "=" in eq:
        lhs, rhs = eq.split("=", 1)
        expr_str = f"({lhs}) - ({rhs})"
    else:
        expr_str = eq

    syms = _collect_symbols(expr_str)
    if not syms:
        print("  No symbols detected — evaluating numerically.")
        try:
            val = parse(expr_str)
            print(f"  Result: {val}")
        except Exception as exc:
            print(f"  Error: {exc}")
    elif len(syms) == 1:
        sv = syms[0]
        print(f"  Detected unknown: {sv}")
        try:
            result = solve_equation(eq, sv, {})
            print(result)
        except Exception as exc:
            print(f"\n  Error: {exc}")
    else:
        print(f"  Multiple symbols found: {', '.join(syms)}")
        sv = input(f"  Which to solve for? ").strip()
        known_raw = input("  Known values (e.g. a=9.8, t=3): ").strip()
        known = {}
        if known_raw:
            for part in known_raw.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    known[k.strip()] = v.strip()
        try:
            result = solve_equation(eq, sv, known)
            print(result)
        except Exception as exc:
            print(f"\n  Error: {exc}")

    print()
    input("  Press Enter to continue...")


def mode_unit_analysis():
    """Simple conceptual unit reminder for common quantities."""
    UNITS = {
        "displacement / distance": "meters (m)",
        "velocity / speed":        "m/s",
        "acceleration":            "m/s²",
        "force":                   "Newtons (N = kg·m/s²)",
        "energy / work":           "Joules (J = kg·m²/s²)",
        "power":                   "Watts (W = J/s)",
        "momentum":                "kg·m/s",
        "impulse":                 "N·s  =  kg·m/s",
        "pressure":                "Pascals (Pa = N/m²)",
        "frequency":               "Hertz (Hz = 1/s)",
        "electric charge":         "Coulombs (C)",
        "voltage / potential":     "Volts (V = J/C)",
        "current":                 "Amperes (A = C/s)",
        "resistance":              "Ohms (Ω = V/A)",
        "capacitance":             "Farads (F = C/V)",
        "inductance":              "Henrys (H = V·s/A)",
        "magnetic field":          "Tesla (T = kg/(A·s²))",
        "temperature":             "Kelvin (K)  or  Celsius (°C)",
        "angle":                   "radians (rad)  or  degrees (°)",
        "angular velocity":        "rad/s",
        "torque":                  "N·m",
        "moment of inertia":       "kg·m²",
    }
    header("SI Units Quick Reference")
    col = max(len(k) for k in UNITS)
    for qty, unit in UNITS.items():
        print(f"  {qty:<{col}}  →  {unit}")
    print()
    input("  Press Enter to continue...")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

MAIN_OPTIONS = [
    "Browse formula sheets",
    "Physical constants",
    "Equation solver  (specify knowns)",
    "Quick solve      (auto-detect unknown)",
    "Calculus tools   (diff / integrate / limit)",
    "SI units reference",
]

MODE_MAP = {
    "1": mode_formulas,
    "2": mode_constants,
    "3": mode_solve,
    "4": mode_quick,
    "5": mode_calc,
    "6": mode_unit_analysis,
}


def main():
    # Allow direct mode flags from CLI
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("solve",):
            mode_solve(); return
        if arg in ("calc", "calculus"):
            mode_calc(); return
        if arg in ("formulas", "formula"):
            mode_formulas(); return
        if arg in ("constants",):
            mode_constants(); return
        if arg in ("units",):
            mode_unit_analysis(); return
        if arg in ("quick",):
            mode_quick(); return

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║          ⚛  Physics Helper  ⚛               ║")
    print("  ║  Algebra · Calculus · Formula Reference      ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    while True:
        choice = menu("Main Menu", MAIN_OPTIONS)
        if choice == "0":
            print("\n  Goodbye!\n")
            break
        fn = MODE_MAP.get(choice)
        if fn:
            fn()
        else:
            print("  Invalid choice.\n")


if __name__ == "__main__":
    main()
