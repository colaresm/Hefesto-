import numpy as np
import pandas as pd
import pytest

from hefesto.samples import make_bga_fatigue_doe, to_workbench_csv


@pytest.fixture(scope="session")
def bga_df() -> pd.DataFrame:
    return make_bga_fatigue_doe(n_samples=80, seed=3)


@pytest.fixture(scope="session")
def workbench_csv(bga_df) -> str:
    return to_workbench_csv(bga_df, failed_rows=(5, 40))


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    x1, x2 = rng.uniform(0, 1, n), rng.uniform(-1, 1, n)
    material = rng.choice(["SAC305", "SnPb"], n)
    y = 3 * x1**2 + np.sin(2 * x2) + np.where(material == "SAC305", 0.5, 0.0) + 5
    return pd.DataFrame({"x1": x1, "x2": x2, "material": material, "y": y})
