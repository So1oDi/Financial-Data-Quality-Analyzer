import numpy as np
import pandas as pd


def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def detect(series, method="IQR"):
    values = numeric(series)
    if values.count() < 4:
        return values.notna() & False
    if method == "IQR":
        q1, q3 = values.quantile([.25, .75])
        spread = q3 - q1
        return (values.lt(q1 - 1.5 * spread) | values.gt(q3 + 1.5 * spread)) if spread else values.notna() & False
    if method == "Z-score":
        deviation = values.std(ddof=0)
        return (values - values.mean()).abs().gt(3 * deviation) if deviation else values.notna() & False
    raise ValueError("Невідомий метод пошуку аномалій.")

