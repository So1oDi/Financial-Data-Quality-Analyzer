from statistics import median
from time import perf_counter_ns

import pandas as pd

from core.anomalies import numeric

REQUIRED = {"quantity", "price", "customer_id", "order_id", "category"}


def sales(frame):
    line_total = numeric(frame["quantity"]) * numeric(frame["price"])
    order_total = line_total.groupby(frame["order_id"], sort=False).sum(min_count=1)
    return {
        "Загальна сума продажів": float(line_total.sum()),
        "Середня вартість замовлення": float(order_total.mean()),
        "Унікальні клієнти": int(frame["customer_id"].nunique()),
    }, line_total.groupby(frame["category"], sort=False, dropna=False).sum(min_count=1)


def benchmark(frame, repeats=7):
    sales(frame)
    times = []
    for _ in range(repeats):
        start = perf_counter_ns()
        figures, categories = sales(frame)
        times.append((perf_counter_ns() - start) / 1_000_000)
    return figures, categories, median(times)


def relative_accuracy(value, expected):
    if pd.isna(value) or pd.isna(expected):
        return None
    if expected == 0:
        return 100.0 if value == 0 else None
    return 100.0 * max(0.0, 1 - abs(value - expected) / abs(expected))


def category_comparison(before, after, expected):
    totals = pd.concat({"До": before, "Після": after, "Еталон": expected}, axis=1).fillna(0)
    totals.index.name = "Категорія"
    return totals

