"""Check the actual runtime after importing the libraries exercised by our tests."""
import importlib
import os
import platform
import sys
import sysconfig

for name in (
    "numpy", "scipy", "pandas", "polars", "narwhals", "pyarrow", "sklearn",
    "shap", "lime", "statsmodels.api", "matplotlib", "flask", "ipywidgets", "dalex",
):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "(no version attribute)"))

print(platform.platform(), sys.version)
free_threaded = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
print("Py_GIL_DISABLED:", free_threaded)
if hasattr(sys, "_is_gil_enabled"):
    enabled = sys._is_gil_enabled()
    print("GIL enabled after scientific/optional dependency imports:", enabled)
    expected = os.environ.get("EXPECT_GIL", "1") == "1"
    assert enabled == expected, "Unexpected GIL state after dependency imports"
    assert free_threaded == (not expected), "Unexpected interpreter build"
else:
    assert os.environ.get("EXPECT_GIL", "1") == "1"
