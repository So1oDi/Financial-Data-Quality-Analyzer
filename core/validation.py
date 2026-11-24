from dataclasses import dataclass

import pandas as pd

from core.anomalies import detect, numeric

EMAIL = r"[^\s@]+@[^\s@.]+(?:\.[^\s@.]+)+"
ISSUE_COLUMNS = ["Позиція рядка", "Стовпець", "Проблема", "Значення"]


@dataclass(frozen=True)
class Rule:
    kind: str = ""
    nonnegative: bool = False
    categories: tuple = ()


def normalized(series):
    return series.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def missing(series):
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)


def dates(series):
    text = series.astype("string").str.strip()
    # Числові ідентифікатори не трактуємо як дати.
    text = text.where(text.str.contains(r"[-./]|[A-Za-z]", regex=True, na=False))
    iso = text.str.match(r"^\d{4}[-/]", na=False)
    first = pd.to_datetime(text.where(iso), format="mixed", errors="coerce", utc=True)
    second = pd.to_datetime(text.where(~iso), format="mixed", dayfirst=True, errors="coerce", utc=True)
    return first.fillna(second)


def inspect(frame, rules=None, keys=(), anomaly_columns=(), method="IQR"):
    """Рахує перевірки й збирає всі спрацювання з позиціями рядків."""
    rules = rules or {}
    issues = []
    totals = dict(valid=[0, 0], consistent=[0, 0], missing=0)
    duplicate = frame.duplicated() if len(frame.columns) else pd.Series(False, index=frame.index)
    key_duplicate = pd.Series(False, index=frame.index)

    def record(mask, label, column="—", series=None):
        positions = mask.fillna(False).to_numpy(dtype=bool).nonzero()[0]
        if len(positions):
            issues.append(pd.DataFrame({
                ISSUE_COLUMNS[0]: positions + 1,
                ISSUE_COLUMNS[1]: column,
                ISSUE_COLUMNS[2]: label,
                ISSUE_COLUMNS[3]: "" if series is None else series.iloc[positions].to_numpy(),
            }))

    def check(group, eligible, bad, label, column, series):
        flagged = eligible & bad.fillna(False)
        totals[group][0] += int(eligible.sum())
        totals[group][1] += int(flagged.sum())
        record(flagged, label, column, series)

    record(duplicate, "Повний дублікат")
    if keys:
        complete = ~pd.concat([missing(frame[key]) for key in keys], axis=1).any(axis=1)
        key_duplicate.loc[complete] = frame.loc[complete, list(keys)].duplicated().to_numpy()
        record(key_duplicate, "Дублікат за ключем", ", ".join(keys))
    for column, series in frame.items():
        absent = missing(series)
        eligible = ~absent
        totals["missing"] += int(absent.sum())
        record(absent, "Пропущене значення", column, series)
        rule = rules.get(column, Rule())
        text = series.astype("string")
        if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
            check("consistent", eligible, text.ne(normalized(series)), "Зайві пробіли", column, series)
        values = numeric(series) if rule.kind in ("number", "integer") or rule.nonnegative else None
        if rule.kind in ("number", "integer"):
            bad = values.isna()
            if rule.kind == "integer":
                bad |= values.mod(1).ne(0)
            check("valid", eligible, bad, "Некоректний тип", column, series)
        elif rule.kind == "date":
            check("valid", eligible, dates(series).isna(), "Некоректна дата", column, series)
        elif rule.kind == "email":
            check("valid", eligible, ~text.str.fullmatch(EMAIL, na=False), "Некоректна електронна адреса", column, series)
        if rule.nonnegative:
            check("valid", values.notna(), values.lt(0), "Від’ємне значення", column, series)
        if rule.categories:
            check("consistent", eligible, ~text.isin(rule.categories), "Неузгоджена категорія", column, series)
        if column in anomaly_columns:
            record(detect(series, method), f"Аномалія ({method})", column, series)
    table = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=ISSUE_COLUMNS)
    return table, totals, int((duplicate | key_duplicate).sum())

