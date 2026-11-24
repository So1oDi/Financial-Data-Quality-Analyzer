from html import escape
from io import BytesIO

import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.exceptions import IllegalCharacterError

from core.metrics import METRICS


def _safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def safe_table(frame):
    result = frame.copy()
    for column in result.select_dtypes(exclude="number"):
        result[column] = result[column].map(_safe_cell)
    result.columns = [_safe_cell(value) for value in result.columns]
    return result


def csv_bytes(frame):
    return safe_table(frame).to_csv(index=False).encode("utf-8-sig")


def xlsx_bytes(frame):
    output = BytesIO()
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("Дані")
    try:
        sheet.append([_safe_cell(name) for name in frame.columns])
        for row in frame.itertuples(index=False, name=None):
            sheet.append([None if pd.isna(value) else _safe_cell(value) for value in row])
        workbook.save(output)
    except IllegalCharacterError as error:
        raise ValueError("Дані містять символи, які XLSX не підтримує.") from error
    finally:
        workbook.close()
    return output.getvalue()


def quality_table(scores):
    return pd.DataFrame([
        {"Метрика": name, "Оцінка": "Недостатньо даних" if score is None else f"{score:.2f}%",
         "Вага": METRICS[name][0], "Опис": METRICS[name][1]}
        for name, score in scores.items()
    ])


def html_report(name, frame, scores, overall, issues, journal):
    score = f"{overall:.2f}%" if overall is not None else "Недостатньо даних"
    counts = issues["Проблема"].value_counts().rename_axis("Проблема").reset_index(name="Кількість")
    tables = [
        ("Метрики", quality_table(scores)), ("Типи проблем", counts),
        ("Перші 200 спрацювань", issues.head(200)), ("Журнал очищення", pd.DataFrame(journal)),
    ]
    parts = "".join(f"<h2>{label}</h2>{table.to_html(index=False, escape=True)}" for label, table in tables)
    return f"""<!doctype html><html lang="uk"><meta charset="utf-8">
<title>Звіт про якість даних</title>
<style>
body{{font:16px system-ui;color:#162235;max-width:1200px;margin:3rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%;margin-bottom:2rem}}
td,th{{padding:.5rem;border:1px solid #c9d5e2;text-align:left;overflow-wrap:anywhere}}
th{{background:#e6f4f3}}
</style>
<h1>Звіт про якість даних</h1><p>{escape(name)} · {len(frame):,} рядків · {len(frame.columns)} стовпців</p>
<p>Інтегральна оцінка <strong>{score}</strong> · Спрацювань {len(issues):,}</p>
<p>Інтегральна оцінка — середнє зважене доступних метрик.</p>{parts}</html>"""

