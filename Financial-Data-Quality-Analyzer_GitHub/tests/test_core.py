from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from core.analytics import benchmark, relative_accuracy, sales
from core.anomalies import detect
from core.cleaning import ACTIONS, Operation, clean, demo_operations
from core.data_loader import load_data
from core.metrics import accuracy, evaluate
from core.report import csv_bytes, html_report, xlsx_bytes
from core.validation import Rule, inspect
from utils.data_generator import KEYS, NUMERIC, RULES, SIZES, generate


@pytest.mark.parametrize("encoding,separator", [("utf-8-sig", ";"), ("cp1251", "\t"), ("utf-16", "|")])
def test_csv_detection(encoding, separator):
    raw = f"місто{separator}число\nКиїв{separator}4\n".encode(encoding)
    frame, metadata = load_data(raw, "test.csv")
    assert frame.iloc[0].tolist() == ["Київ", 4]
    assert metadata["Роздільник"] == repr(separator)


@pytest.mark.parametrize("raw,name", [(b"", "a.csv"), (b"a,b\n1,2,3\n", "a.csv"),
                                       (b"bad zip", "a.xlsx"), (b"x", "a.xls")])
def test_bad_file_is_rejected(raw, name):
    with pytest.raises(ValueError):
        load_data(raw, name)


def test_empty_table_has_no_scores():
    frame, _ = load_data(b"a,b\n", "header.csv")
    scores, overall, issues = evaluate(frame)
    assert frame.empty and issues.empty
    assert overall is None and all(value is None for value in scores.values())


def test_validation_issue_types_and_union_of_duplicates():
    frame = pd.DataFrame({
        "id": [1, 1, 1, 2], "n": [-1, -1, "x", None],
        "mail": ["wrong", "wrong", "x@y.ua", "x@y.ua"],
        "date": ["30.02.2025", "30.02.2025", "01.02.2025", "2025-01-01"],
        "cat": [" A ", " A ", "B", "A"],
    })
    rules = {"n": Rule("integer", True), "mail": Rule("email"),
             "date": Rule("date"), "cat": Rule(categories=("A", "B"))}
    issues, totals, duplicates = inspect(frame, rules, ("id",))
    assert totals["missing"] == 1 and duplicates == 2
    assert set(issues["Проблема"]) == {
        "Повний дублікат", "Дублікат за ключем", "Пропущене значення",
        "Некоректний тип", "Від’ємне значення", "Некоректна електронна адреса",
        "Некоректна дата", "Зайві пробіли", "Неузгоджена категорія",
    }


def test_metrics_and_weighting():
    frame = pd.DataFrame({"id": [1, 2], "n": [4, np.nan], "d": ["2025-03-31", "bad"]})
    reference = pd.DataFrame({"id": [1, 2], "n": [4, 5]})
    scores, overall, _ = evaluate(frame, {"n": Rule("number")}, ("id",), reference=reference,
                                  freshness=("d", date(2025, 3, 31), 90))
    assert scores == {
        "Повнота": pytest.approx(100 * 5 / 6), "Унікальність": 100, "Валідність": 100,
        "Узгодженість": 100, "Точність": 50, "Своєчасність": 50,
    }
    assert overall == pytest.approx((20 * 100 * 5 / 6 + 15 * 100 + 20 * 100 + 15 * 100 + 20 * 50 + 10 * 50) / 100)
    _, available_only, _ = evaluate(pd.DataFrame({"n": [1, np.nan]}))
    assert available_only == pytest.approx((20 * 50 + 15 * 100) / 35)


def test_accuracy_ambiguous_key_and_no_reference():
    reference = pd.DataFrame({"k": [1, 2], "x": ["a", "b"]})
    duplicated = pd.DataFrame({"k": [1, 1, 2], "x": ["a", "a", "b"]})
    assert accuracy(duplicated, reference, ("k",)) == 50
    assert accuracy(reference, duplicated, ("k",)) is None
    assert accuracy(reference, None, ("k",)) is None


@pytest.mark.parametrize("method", ["IQR", "Z-score"])
def test_anomaly_detection(method):
    values = pd.Series([*range(1, 21), 1000, None, "bad"])
    flags = detect(values, method)
    assert flags.sum() == 1 and flags.iloc[20]


@pytest.mark.parametrize("action,expected", [(3, 14/3), (4, 2), (5, 2), (6, 7)])
def test_fill_choices(action, expected):
    source = pd.DataFrame({"a": [2, 2, 10, np.nan]})
    result, log = clean(source, [Operation(ACTIONS[action], ("a",), "7")])
    assert result["a"].iloc[-1] == pytest.approx(expected)
    assert pd.isna(source["a"].iloc[-1])
    assert log[0]["Змінено комірок"] == 1


def test_clean_pipeline_handles_strings_and_dates():
    source = pd.DataFrame({"k": [1, 1, 2], "n": ["4", "4", "bad"],
                           "cat": [" дім ", " дім ", "Книги"], "d": ["02.03.2025", "02.03.2025", "bad"]})
    result, log = clean(source, [
        Operation(ACTIONS[0]), Operation(ACTIONS[15], ("cat",), "Дім\nКниги"),
        Operation(ACTIONS[9], ("n",)), Operation(ACTIONS[11], ("d",)), Operation(ACTIONS[2], ("d",)),
    ])
    assert result.iloc[0].tolist() == [1, 4, "Дім", "2025-03-02"]
    assert sum(item["Видалено рядків"] for item in log) == 2
    assert len(source) == 3


def test_exports_neutralize_formulas_and_escape_html():
    frame = pd.DataFrame({"=header": ["=1+2", "<script>x</script>"], "n": [-1, 2]})
    assert "'=1+2" in csv_bytes(frame).decode("utf-8-sig")
    raw = xlsx_bytes(frame)
    with BytesIO(raw) as buffer:
        book = load_workbook(buffer)
        assert book.active["A2"].data_type == "s"
        assert book.active["B2"].value == -1
        book.close()
    imported, meta = load_data(raw, "data.xlsx")
    assert imported.shape == frame.shape and meta["Аркуш"]
    scores, overall, issues = evaluate(frame)
    html = html_report("<script>", frame, scores, overall, issues, [])
    assert "<script>" not in html and "&lt;script&gt;" in html


@pytest.mark.parametrize("size", SIZES)
def test_demo_sizes_and_defects(size):
    reference, damaged = generate(size)
    assert len(reference) == size and len(damaged) == size + size // 50
    assert reference["order_id"].is_unique
    assert damaged.duplicated().sum() == size // 100


def test_experiment_improves_sales_accuracy():
    reference, damaged = generate()
    pd.testing.assert_frame_equal(generate()[1], damaged)
    repaired, _ = clean(damaged, demo_operations())
    options = dict(rules=RULES, keys=KEYS, anomaly_columns=NUMERIC, reference=reference)
    assert evaluate(repaired, **options)[1] > evaluate(damaged, **options)[1]
    target, _ = sales(reference)
    before, _, elapsed_before = benchmark(damaged)
    after, _, elapsed_after = benchmark(repaired)
    name = "Загальна сума продажів"
    assert elapsed_before > 0 and elapsed_after > 0
    assert relative_accuracy(after[name], target[name]) > relative_accuracy(before[name], target[name])

