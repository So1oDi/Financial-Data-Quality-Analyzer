import numpy as np
import pandas as pd

from core.validation import Rule

CATEGORIES = ("Електроніка", "Дім", "Книги", "Спорт")
PRODUCTS = ("Навушники", "Лампа", "Атлас", "М’яч")
SIZES = (1_000, 10_000, 100_000)
KEYS = ("order_id",)
NUMERIC = ("quantity", "price")
RULES = {
    "order_id": Rule("integer", True),
    "customer_id": Rule("integer", True),
    "email": Rule("email"),
    "quantity": Rule("integer", True),
    "price": Rule("number", True),
    "order_date": Rule("date"),
    "category": Rule(categories=CATEGORIES),
}


def generate(size=1_000):
    if size not in SIZES:
        raise ValueError("Оберіть 1 000, 10 000 або 100 000 записів.")
    rng = np.random.default_rng(42)
    customers = rng.integers(1, size // 4 + 1, size)
    categories = rng.integers(0, len(CATEGORIES), size)
    names = np.array(["Олена", "Андрій", "Марія", "Іван", "Ірина", "Олег"])
    cities = np.array(["Київ", "Львів", "Одеса", "Дніпро", "Харків"])
    date = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 90, size), unit="D")
    clean = pd.DataFrame({
        "order_id": np.arange(1, size + 1),
        "customer_id": customers,
        "customer_name": names[customers % len(names)],
        "email": [f"client{i}@example.com" for i in customers],
        "product": np.array(PRODUCTS)[categories],
        "category": np.array(CATEGORIES)[categories],
        "quantity": rng.integers(1, 9, size),
        "price": rng.integers(10_000, 100_001, size) / 100,
        "order_date": date.strftime("%Y-%m-%d"),
        "city": cities[customers % len(cities)],
    })
    damaged = clean.astype({"quantity": object, "price": object}).copy()
    count = size // 100
    positions = rng.permutation(size).reshape(100, count)
    changes = [
        ("price", None), ("quantity", "п’ять"), ("quantity", -3),
        ("email", "wrong-address"), ("order_date", "31.02.2025"),
        ("category", " електроніка "), ("price", 30_000),
        ("customer_name", " Олена  "), ("city", None),
    ]
    for index, (column, value) in enumerate(changes):
        damaged.loc[positions[index], column] = value
    full = damaged.loc[positions[9]]
    key = damaged.loc[positions[10]].assign(city="Киiв")
    return clean, pd.concat([damaged, full, key], ignore_index=True)

