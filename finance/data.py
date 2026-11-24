import numpy as np
import pandas as pd

from core.validation import Rule, dates, normalized
from utils.data_generator import SIZES

CATEGORIES = {"Дохід": ("Продажі", "Послуги"), "Витрата": ("Оренда", "Зарплата", "Матеріали")}
TYPES = tuple(CATEGORIES)
METHODS = ("Переказ", "Картка", "Готівка")
STATUSES = ("Проведено", "Очікує", "Скасовано")
DIMS = ("date", "category", "department", "counterparty", "payment_method", "transaction_type")
FIELDS = ("transaction_id", "transaction_date", "transaction_type", "category", "amount", "currency",
          "department", "counterparty", "payment_method", "status", "description")
RULES = {"amount": Rule("number", True), "transaction_date": Rule("date"),
         "transaction_type": Rule(categories=TYPES), "status": Rule(categories=STATUSES),
         "category": Rule(categories=sum(CATEGORIES.values(), ())), "currency": Rule(categories=("UAH",))}


def generate(size=1000, seed=42):
    if size not in SIZES:
        raise ValueError("Оберіть один із трьох підтримуваних обсягів.")
    rng = np.random.default_rng(seed)
    day = rng.integers(0, 731, size)
    day[:24] = (pd.date_range("2024-01-01", periods=24, freq="MS") - pd.Timestamp("2024-01-01")).days
    day[-1] = 730
    category = rng.choice(sum(CATEGORIES.values(), ()), size)
    category[:24] = "Оренда"
    income = np.isin(category, CATEGORIES["Дохід"])
    amount = rng.lognormal(8.8, .55, size) * (1 + day / 2200 + .25 * np.sin(day * 2 * np.pi / 365))
    amount[:24] = 35000
    suspicious = rng.choice(np.arange(24, size), size // 50, replace=False)
    amount[suspicious] *= 6
    method = rng.choice(METHODS, size)
    method[suspicious] = "Готівка"
    party = rng.integers(1, 80, size)
    party[:24] = 1
    status = rng.choice(STATUSES, size, p=[.94,.04,.02])
    status[:24] = STATUSES[0]
    ref = pd.DataFrame(dict(zip(FIELDS, [
        [f"T{i:07d}" for i in range(size)], (pd.Timestamp("2024-01-01") + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
        np.where(income, TYPES[0], TYPES[1]), category, amount.round(2), "UAH",
        rng.choice(["Центр","Захід","Схід"], size), [f"Партнер {i}" for i in party],
        method, status, [f"Розрахунок за договором {i}" for i in party],
    ])))
    labels = pd.Series(False, index=ref.transaction_id, name="підозріла")
    labels.iloc[suspicious] = True
    bad = ref.astype({"amount":object}).copy()
    groups = rng.permutation(np.flatnonzero(~labels.to_numpy()))[:9 * (size//100)].reshape(9,-1)
    for group, (col, value) in zip(groups, [
        ("department", None), ("amount",-1), ("transaction_date","31.02.2025"), ("category","???"),
        ("amount","помилка"), ("status","???"), ("department","  Центр "), ("amount",950000),
    ]):
        bad.loc[group,col] = value
    return ref, pd.concat([bad,bad.iloc[groups[-1]]],ignore_index=True), labels


def transform(raw):
    absent = set(FIELDS) - set(raw)
    if absent:
        raise ValueError("Відсутні стовпці: " + ", ".join(sorted(absent)))
    frame = raw.loc[:, FIELDS].copy().reset_index(drop=True)
    for col in frame.columns.drop("amount"):
        frame[col] = normalized(frame[col]).replace("", pd.NA)
    frame["currency"] = frame.currency.str.upper()
    frame["transaction_date"] = dates(frame.transaction_date).dt.strftime("%Y-%m-%d")
    frame["amount"] = pd.to_numeric(frame.amount, errors="coerce")
    reason = pd.Series("", index=frame.index)
    category_ok = pd.Series(False, index=frame.index)
    for kind, choices in CATEGORIES.items():
        category_ok |= frame.transaction_type.eq(kind).fillna(False) & frame.category.isin(choices)
    for invalid, label in [
        (frame.drop(columns="description").isna().any(axis=1),"Пропущене або некоректне поле"),
        (~frame.amount.between(.01, 999999999999.99) | (frame.amount.sub(frame.amount.round(2)).abs() > 1e-8),"Некоректна сума"),
        (~category_ok,"Невідповідна категорія або тип"), (~frame.status.isin(STATUSES),"Некоректний статус"),
        (~frame.currency.eq("UAH"),"Непідтримувана валюта"), (~frame.payment_method.isin(METHODS),"Некоректний спосіб оплати"),
    ]:
        reason.loc[invalid.fillna(True) & reason.eq("")] = label
    # Один ідентифікатор має описувати єдину суму. Конфлікт потребує звірки обох записів.
    inconsistent = frame.groupby("transaction_id").amount.transform("nunique").gt(1)
    reason.loc[inconsistent & reason.eq("")] = "Суперечливі суми одного ідентифікатора"
    if "document_amount" in raw:
        documented = pd.to_numeric(raw["document_amount"].reset_index(drop=True),errors="coerce")
        mismatch = documented.notna() & frame.amount.sub(documented).abs().gt(.005)
        reason.loc[mismatch & reason.eq("")] = "Сума відрізняється від суми первинного документа (UAH)"
    eligible = frame.loc[reason.eq("")]
    reason.loc[eligible.index[eligible.transaction_id.duplicated()]] = "Повторний ідентифікатор"
    frame["description"] = frame.description.fillna("")
    rejected = raw.reset_index(drop=True).loc[reason.ne("")].copy()
    rejected["Причина відхилення"] = reason[reason.ne("")]
    return frame.loc[reason.eq("")].reset_index(drop=True), rejected


def amount_review(accepted):
    """Повторювана велика сума у незалежних контрагентів може бути помилкою імпорту."""
    amount = accepted.amount
    typical = amount.groupby(accepted.category).transform("median")
    parties = accepted.groupby("amount").counterparty.transform("nunique")
    flagged = accepted.loc[amount.gt(10*typical) & parties.ge(3)].copy()
    flagged["Підстава перевірки"] = "Точна сума у ≥3 контрагентів і понад 10 медіан категорії"
    return flagged
