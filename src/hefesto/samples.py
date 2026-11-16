from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy.stats import qmc

INPUTS: dict[str, tuple[float, float, str]] = {
    "Die Thickness": (0.10, 0.40, "mm"),
    "Die Size": (5.0, 15.0, "mm"),
    "Substrate Thickness": (0.20, 0.60, "mm"),
    "Ball Pitch": (0.40, 1.00, "mm"),
    "Ball Diameter to Pitch Ratio": (0.45, 0.65, "-"),
    "Standoff Height": (0.15, 0.45, "mm"),
    "PCB Thickness": (0.80, 2.00, "mm"),
    "Mold Compound CTE": (7.0, 15.0, "ppm/C"),
    "Temperature Range": (100.0, 165.0, "C"),
    "Dwell Time": (5.0, 30.0, "min"),
}
OUTPUTS: dict[str, str] = {
    "Plastic Work Density": "MPa",
    "Characteristic Life": "cycles",
    "Max Package Warpage": "um",
}


def make_bga_fatigue_doe(n_samples: int = 200, seed: int = 7, noise: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    unit = qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n_samples)
    lows = np.array([v[0] for v in INPUTS.values()])
    highs = np.array([v[1] for v in INPUTS.values()])
    X = pd.DataFrame(qmc.scale(unit, lows, highs), columns=list(INPUTS))

    t_die, size, t_sub = X["Die Thickness"], X["Die Size"], X["Substrate Thickness"]
    pitch, ratio, h = X["Ball Pitch"], X["Ball Diameter to Pitch Ratio"], X["Standoff Height"]
    t_pcb, cte_emc, d_t, dwell = (
        X["PCB Thickness"],
        X["Mold Compound CTE"],
        X["Temperature Range"],
        X["Dwell Time"],
    )

    t_emc = t_die + 0.25
    cte_pkg = (2.6 * t_die + cte_emc * t_emc + 15.0 * t_sub) / (t_die + t_emc + t_sub)
    dnp = (size / 2.0) * np.sqrt(2.0)
    shear_strain = np.abs(17.0 - cte_pkg) * 1e-6 * d_t * dnp / h
    shape = (0.55 / ratio) ** 0.8
    stiffness = (t_pcb / 1.6) ** 0.35
    creep = 1.0 + 0.12 * np.log(dwell / 5.0)
    dW = 22.0 * (shear_strain * shape * stiffness * creep) ** 1.35

    pad = 0.8 * ratio * pitch
    n0 = 3200.0 * dW**-1.30
    da_dn = 4.5e-4 * dW**1.05
    life = n0 + pad / da_dn

    warpage = 10.0 * (0.035 * (cte_emc - 6.0) * size**2 * d_t / 165.0) / (
        1.0 + 6.0 * t_sub + 3.0 * t_die
    ) + 4.0 * np.sin(np.pi * t_die / 0.4)

    y = pd.DataFrame(
        {
            "Plastic Work Density": dW * rng.lognormal(0, noise, n_samples),
            "Characteristic Life": life * rng.lognormal(0, 2 * noise, n_samples),
            "Max Package Warpage": warpage * (1 + rng.normal(0, noise, n_samples)),
        }
    )
    return pd.concat([X, y], axis=1)


def to_workbench_csv(df: pd.DataFrame, failed_rows: tuple[int, ...] = (17, 88, 141)) -> str:
    units = {**{k: v[2] for k, v in INPUTS.items()}, **OUTPUTS}
    headers = [f"P{i + 1} - {c} [{units[c]}]" for i, c in enumerate(df.columns)]
    out = df.copy()
    out.columns = headers
    out = out.round(6)
    out.insert(0, "Name", [f"DP {i}" for i in range(len(out))])
    output_cols = headers[len(INPUTS) :]
    for r in failed_rows:
        if r < len(out):
            out.loc[r, output_cols] = np.nan
    buf = io.StringIO()
    buf.write("# Hefesto demo - synthetic BGA solder joint thermal cycling DOE\n")
    buf.write("# Layout mimics an ANSYS Workbench Parameter Set export (Design Points table)\n")
    buf.write("# P1-P10: input parameters | P11-P13: output parameters\n")
    out.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()
