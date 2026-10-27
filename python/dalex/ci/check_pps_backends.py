"""Optional PPS integration in its compatible pandas-1.x environment."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import ppscore
import dalex as dx


class Model:
    def __init__(self, backend):
        self.backend = backend

    def predict(self, data):
        assert isinstance(
            data, pd.DataFrame if self.backend == "pandas" else pl.DataFrame
        )
        return 2 * data["x"].to_numpy() + 0.1 * data["z"].to_numpy()


def main():
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(48, 3)), columns=["x", "z", "u"])
    matrices = []
    for backend in ("pandas", "polars"):
        data = frame if backend == "pandas" else pl.from_pandas(frame)
        exp = dx.Explainer(Model(backend), data, model_type="regression", verbose=False)
        aspect = dx.Aspect(exp, depend_method="pps")
        matrices.append(aspect.depend_matrix)
        assert type(exp.data) is type(data)
    pd.testing.assert_frame_equal(*matrices)
    report = {
        "pandas": pd.__version__,
        "ppscore": ppscore.__version__,
        "passed": True,
        "max_absolute_difference": float(
            np.max(np.abs(matrices[0].to_numpy() - matrices[1].to_numpy()))
        ),
    }
    Path("pps-backend-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
