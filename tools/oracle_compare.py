from pathlib import Path
import math

import numpy as np

DTYPES = ("float16", "float32", "float64", "int16", "int32")

class OutputError(ValueError):
    pass

def read_tensor(path, dtype="float32", expected_elements=None):
    path = Path(path)
    dt = np.dtype(dtype).newbyteorder("<")
    if not path.is_file():
        raise OutputError(f"missing output: {path}")
    size = path.stat().st_size
    if size == 0 or size % dt.itemsize:
        raise OutputError(f"empty or misaligned output: {path} ({size} bytes, {dtype})")
    if expected_elements is not None and size != expected_elements * dt.itemsize:
        raise OutputError(f"unexpected output size: {path}: {size} bytes; "
                          f"expected {expected_elements * dt.itemsize}")
    values = np.fromfile(path, dtype=dt)
    if not np.isfinite(values).all():
        raise OutputError(f"non-finite output (NaN/Inf): {path}; numerical oracle undefined")
    return values

def compare_bins(a, b, threshold=1e-4, dtype="float32", expected_elements=None):
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    x = read_tensor(a, dtype, expected_elements)
    y = read_tensor(b, dtype, expected_elements)
    if x.shape != y.shape:
        raise OutputError(f"different output shapes: {x.shape} vs {y.shape}")

    with np.errstate(over="ignore"):
        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    index = int(np.argmax(delta))
    error = float(delta[index])
    return {
        "status": "FAIL" if error > threshold else "PASS",
        "max_abs_diff": error if math.isfinite(error) else "Infinity",
        "max_abs_diff_index": index,
        "nelem": int(x.size), "threshold": threshold,
        "elements_over_threshold": int(np.count_nonzero(delta > threshold)),
        "bin1": str(a), "bin2": str(b), "dtype": dtype,
    }
