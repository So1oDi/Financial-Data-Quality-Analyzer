from dataclasses import dataclass

import pandas as pd

from core.anomalies import detect, numeric
from core.validation import dates, missing, normalized


@dataclass(frozen=True)
class Operation:
    action: str
    columns: tuple = ()
    value: str = ""


ACTIONS = (
    "Видалити повні дублікати", "Видалити дублікати за ключем",
    "Видалити рядки з пропусками", "Заповнити середнім", "Заповнити медіаною",
    "Заповнити модою", "Заповнити своїм значенням", "Нормалізувати текст",
    "Привести до числа", "Привести до цілого", "Привести до тексту",
    "Стандартизувати дати", "Замінити від’ємні на пропуски",
    "Замінити аномалії медіаною", "Видалити рядки з аномаліями",
    "Уніфікувати категорії за словником",
)


def clean(frame, operations, anomaly_method="IQR"):
    result = frame.copy(deep=True)
    journal = []
    for op in operations:
        if op.action not in ACTIONS or not set(op.columns).issubset(result.columns):
            raise ValueError("Невідома операція або стовпець.")
        before = result
        if op.action == ACTIONS[0]:
            result = before.drop_duplicates()
        elif not op.columns:
            raise ValueError("Оберіть стовпці для операції.")
        elif op.action == ACTIONS[1]:
            eligible = ~before[list(op.columns)].apply(missing).any(axis=1)
            result = before.drop(index=before.loc[eligible].loc[lambda data: data.duplicated(list(op.columns))].index)
        elif op.action == ACTIONS[2]:
            result = before.loc[~before[list(op.columns)].apply(missing).any(axis=1)]
        elif op.action == ACTIONS[14]:
            flagged = pd.concat([detect(before[col], anomaly_method) for col in op.columns], axis=1)
            result = before.loc[~flagged.any(axis=1)]
        else:
            result = before.copy()
            for column in op.columns:
                original = before[column]
                absent = missing(original)
                values = original.mask(absent)
                if op.action in ACTIONS[3:7]:
                    if op.action == ACTIONS[6]:
                        fill = op.value
                    elif op.action == ACTIONS[5]:
                        modes = values.mode()
                        fill = modes.iloc[0] if len(modes) else None
                    else:
                        number = numeric(values)
                        fill = number.mean() if op.action == ACTIONS[3] else number.median()
                    if pd.notna(fill):
                        if pd.api.types.is_numeric_dtype(values) and isinstance(fill, str):
                            try:
                                fill = float(fill)
                            except ValueError:
                                values = values.astype(object)
                        if pd.api.types.is_integer_dtype(values) and isinstance(fill, float):
                            values = values.astype("Float64")
                        values = values.mask(absent, fill)
                elif op.action == ACTIONS[7]:
                    values = normalized(values)
                elif op.action in ACTIONS[8:10]:
                    values = numeric(values)
                    if op.action == ACTIONS[9]:
                        values = values.where(values.mod(1).eq(0) & values.ge(-2**63) & values.lt(2**63)).astype("Int64")
                elif op.action == ACTIONS[10]:
                    values = values.astype("string")
                elif op.action == ACTIONS[11]:
                    values = dates(values).dt.strftime("%Y-%m-%d").astype("string")
                elif op.action == ACTIONS[12]:
                    values = values.mask(numeric(values).lt(0))
                elif op.action == ACTIONS[13]:
                    flagged = detect(values, anomaly_method)
                    replacement = numeric(values).loc[~flagged].median()
                    if flagged.any():
                        if pd.api.types.is_integer_dtype(values):
                            values = values.astype("Float64")
                        values = values.mask(flagged, replacement)
                elif op.action == ACTIONS[15]:
                    labels = [line.strip() for line in op.value.splitlines() if line.strip()]
                    dictionary = {label.casefold(): label for label in labels}
                    if not dictionary or len(dictionary) != len(labels):
                        raise ValueError("Словник має містити однозначні назви категорій.")
                    text = normalized(values)
                    values = text.str.casefold().map(dictionary).fillna(text)
                result[column] = values
        retained = before.loc[result.index, result.columns]
        equal = retained.astype(object).fillna("").eq(result.astype(object).fillna(""))
        journal.append({
            "Операція": op.action,
            "Стовпці": ", ".join(op.columns) or "Усі",
            "Параметр": op.value,
            "Видалено рядків": len(before) - len(result),
            "Змінено комірок": int((~equal).to_numpy().sum()),
        })
    return result.reset_index(drop=True), journal


def demo_operations():
    from utils.data_generator import CATEGORIES, KEYS, NUMERIC
    return [
        Operation(ACTIONS[0]), Operation(ACTIONS[1], KEYS),
        Operation(ACTIONS[7], ("customer_name", "city", "category", "email")),
        Operation(ACTIONS[15], ("category",), "\n".join(CATEGORIES)),
        Operation(ACTIONS[8], NUMERIC), Operation(ACTIONS[12], NUMERIC),
        Operation(ACTIONS[13], ("price",)), Operation(ACTIONS[4], NUMERIC),
        Operation(ACTIONS[11], ("order_date",)),
    ]

