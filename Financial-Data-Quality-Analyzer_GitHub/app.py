from datetime import date
from hashlib import sha256
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from core.analytics import REQUIRED, benchmark, category_comparison, relative_accuracy
from core.cleaning import ACTIONS, Operation, clean, demo_operations
from core.data_loader import load_data
from core.metrics import METRICS, evaluate
from core.report import csv_bytes, html_report, quality_table, xlsx_bytes
from core.validation import Rule
from utils.data_generator import KEYS, NUMERIC, RULES, SIZES, generate

APP_NAME = "Financial Data Quality Analyzer"
PAGES = (
    "Головна", "Завантаження даних", "Аналіз якості даних", "Виявлені проблеми",
    "Очищення даних", "Порівняння результатів", "ETL-пайплайн", "Сховище даних",
    "Фінансова аналітика", "Підозрілі транзакції", "Прогноз грошових потоків",
    "ML-експерименти", "Ефективність аналітичних систем",
    "Експорт результатів",
)
KINDS = {"Без перевірки типу": "", "Число": "number", "Ціле число": "integer",
         "Дата": "date", "Електронна адреса": "email"}
COLORS = ["#59dec4", "#829cff", "#f7b171"]


def format_score(score):
    return "Недостатньо даних" if score is None else f"{score:.2f}%"


def set_dataset(frame, metadata, reference=None):
    demo = reference is not None
    st.session_state.dataset = {
        "original": frame, "metadata": metadata, "reference": reference, "demo_reference": reference, "demo": demo,
        "rules": RULES.copy() if demo else {}, "keys": KEYS if demo else (),
        "anomaly_columns": NUMERIC if demo else (), "method": "IQR",
        "freshness": ("order_date", date(2025, 3, 31), 90) if demo else None,
        "history": [], "queue": [], "preview": None,
        "revision": st.session_state.get("revision", 0) + 1,
    }
    st.session_state.revision = st.session_state.dataset["revision"]
    st.session_state.pop("quality", None)
    st.session_state.pop("experiment", None)
    st.session_state.pop("export_file", None)


def current(data):
    return data["history"][-1][0] if data["history"] else data["original"]


def journal(data):
    return [entry for _, batch in data["history"] for entry in batch]


def assessment(data, original=False):
    frame = data["original"] if original else current(data)
    cache = st.session_state.setdefault("quality", {})
    signature = (id(frame), tuple(data["rules"].items()), data["keys"], data["anomaly_columns"],
                 data["method"], id(data["reference"]), data["freshness"])
    slot = "original" if original else "current"
    if slot not in cache or cache[slot][0] != signature:
        with st.spinner("Обчислюємо показники якості…"):
            result = evaluate(frame, data["rules"], data["keys"], data["anomaly_columns"],
                              data["method"], data["reference"], data["freshness"])
        cache[slot] = (signature, result)
    return cache[slot][1]


def table(frame):
    display = frame.copy()
    for column in display.select_dtypes(include="object", exclude="str"):
        display[column] = display[column].astype("string")
    st.dataframe(display, width="stretch", hide_index=True)


def chart(figure):
    figure.update_layout(
        template="plotly_dark", paper_bgcolor="#182634", plot_bgcolor="#182634",
        margin=dict(l=25, r=25, t=30, b=25), font_color="#e3edf7",
        legend_title_text="", hoverlabel_align="left",
        legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"),
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def upload():
    st.title(PAGES[1])
    left, right = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader("Оберіть локальний файл CSV або XLSX", type=["csv", "xlsx"])
        if uploaded is not None and st.button("Завантажити файл", type="primary"):
            try:
                frame, metadata = load_data(uploaded.getvalue(), uploaded.name)
            except Exception as error:
                st.error(f"Не вдалося завантажити таблицю ({type(error).__name__}). Перевірте файл.")
            else:
                set_dataset(frame, metadata)
                from finance.data import FIELDS, RULES as financial_rules
                if set(FIELDS).issubset(frame):
                    st.session_state.dataset.update(rules=financial_rules.copy(),keys=("transaction_id",),
                                                    anomaly_columns=("amount",))
    with right:
        size = st.selectbox("Обсяг демонстраційного набору", SIZES, format_func=lambda n: f"{n:,}".replace(",", " "))
        if st.button("Завантажити демодані", type="primary"):
            with st.spinner("Генеруємо продажі…"):
                reference, frame = generate(size)
            set_dataset(frame, {"Файл": "Демонстраційні продажі", "Рядків в еталоні": size}, reference)
        if st.button("Згенерувати фінансові транзакції"):
            from finance.data import RULES as financial_rules, generate as generate_financial
            reference, frame, labels = generate_financial(size)
            set_dataset(frame, {"Файл": "Фінансові транзакції", "Рядків в еталоні": size})
            st.session_state.dataset.update(reference=reference, demo_reference=reference,
                financial_labels=labels, rules=financial_rules.copy(), keys=("transaction_id",),
                anomaly_columns=("amount",), freshness=("transaction_date", date(2025,12,31),730))
    data = st.session_state.get("dataset")
    if data:
        describe(data, data["original"])
        st.subheader("Попередній перегляд")
        table(data["original"].head(30))
        st.subheader("Типи стовпців")
        frame = data["original"]
        table(pd.DataFrame({"Стовпець": frame.columns, "Тип pandas": frame.dtypes.astype(str).values}))


def describe(data, frame):
    st.caption(f"Набір даних — {data['metadata']['Файл']}")
    left, right = st.columns(2)
    left.metric("Рядки", f"{len(frame):,}".replace(",", " "))
    right.metric("Стовпці", len(frame.columns))
    details = " · ".join(f"{key} — {value}" for key, value in data["metadata"].items() if key != "Файл")
    if details:
        st.caption(details)


def home(data):
    st.markdown(
        f'<section class="hero"><h1>{APP_NAME}</h1>'
        '<p>Локальний аналіз якості табличних даних. Знаходьте помилки, '
        'перевіряйте очищення та вимірюйте вплив на продажі, фінансові показники й прогнози.</p></section>',
        unsafe_allow_html=True,
    )
    if data is None:
        st.info("Відкрийте «Завантаження даних», щоб імпортувати таблицю або створити демонабір.")
        return
    _, overall, issues = assessment(data)
    left, right = st.columns(2)
    left.metric("Інтегральна оцінка", format_score(overall))
    right.metric("Виявлені спрацювання", f"{len(issues):,}")
    if overall is not None:
        st.progress(overall / 100, text="Рівень якості")
    describe(data, current(data))


def rules_panel(data):
    columns = list(data["original"].columns)
    revision = data["revision"]
    with st.expander("Налаштування аналізу"):
        keys = st.multiselect("Ключові стовпці", columns, data["keys"], key=f"keys{revision}",
                              placeholder="Оберіть стовпці")
        anomalies = st.multiselect("Числові стовпці для аномалій", columns, data["anomaly_columns"],
                                   key=f"anomalies{revision}", placeholder="Оберіть стовпці")
        method = st.selectbox("Метод пошуку аномалій", ["IQR", "Z-score"], index=int(data["method"] == "Z-score"),
                              key=f"method{revision}")
        freshness = data["freshness"]
        date_column = st.selectbox("Стовпець дати для своєчасності", [None, *columns],
                                   index=columns.index(freshness[0]) + 1 if freshness else 0,
                                   format_func=lambda item: item or "Не вибрано", key=f"fresh{revision}")
        left, right = st.columns(2)
        as_of = left.date_input("Дата оцінки", freshness[1] if freshness else date.today(),
                                min_value=date(1700, 1, 1), max_value=date(2200, 12, 31), key=f"asof{revision}")
        days = right.number_input("Допустимий вік, днів", 0, 36500, freshness[2] if freshness else 90,
                                  key=f"days{revision}")
        data.update(keys=tuple(keys), anomaly_columns=tuple(anomalies), method=method,
                    freshness=(date_column, as_of, days) if date_column else None)
        ref_revision = data.get("ref_revision", 0)
        reference = st.file_uploader("Еталон для точності, CSV або XLSX", type=["csv", "xlsx"],
                                     key=f"ref{revision}-{ref_revision}")
        if reference is not None:
            digest = sha256(reference.getbuffer()).digest()
            if digest != data.get("reference_digest"):
                try:
                    data["reference"], _ = load_data(reference.getvalue(), reference.name)
                except Exception:
                    data["reference_digest"] = digest
                    data["reference_error"] = True
                else:
                    data["reference_digest"] = digest
                    data["reference_error"] = False
            if data.get("reference_error"):
                st.error("Не вдалося прочитати еталонний файл.")
        if "reference_digest" in data and st.button("Скинути завантажений еталон", key=f"reset_ref{revision}"):
            data.pop("reference_digest")
            data.pop("reference_error", None)
            data["reference"] = data["demo_reference"]
            data["ref_revision"] = ref_revision + 1
            st.rerun()
        if data["reference"] is not None:
            st.caption(f"Еталон містить {len(data['reference']):,} рядків. Для зіставлення виберіть ключ.")
        selected = st.selectbox("Стовпець для правила", columns, key=f"rule_col{revision}")
        existing = data["rules"].get(selected, Rule())
        with st.form(f"rule-{revision}-{selected}"):
            choices = list(KINDS)
            kind = st.selectbox("Перевірка типу", choices, index=list(KINDS.values()).index(existing.kind))
            nonnegative = st.checkbox("Заборонити від’ємні значення", existing.nonnegative)
            category = st.text_area("Дозволені категорії, по одній на рядок", "\n".join(existing.categories))
            if st.form_submit_button("Зберегти правило"):
                data["rules"][selected] = Rule(KINDS[kind], nonnegative,
                                               tuple(line.strip() for line in category.splitlines() if line.strip()))
                st.success("Правило збережено.")
        if data["rules"]:
            labels = {value: label for label, value in KINDS.items()}
            table(pd.DataFrame([
                {"Стовпець": name, "Тип": labels[rule.kind], "Без від’ємних": rule.nonnegative,
                 "Категорії": ", ".join(rule.categories)}
                for name, rule in data["rules"].items()
            ]))


def quality(data):
    st.title(PAGES[2])
    rules_panel(data)
    scores, overall, _ = assessment(data)
    st.metric("Інтегральна оцінка", format_score(overall))
    for row in (list(scores.items())[:3], list(scores.items())[3:]):
        for column, (name, score) in zip(st.columns(3), row):
            column.metric(name, format_score(score))
    st.caption("Інтегральна оцінка = Σ(оцінка × вага) / Σ(вага) для доступних метрик.")
    table(quality_table(scores))


def problems(data):
    st.title(PAGES[3])
    rules_panel(data)
    _, _, issues = assessment(data)
    if issues.empty:
        st.success("За поточними правилами проблем не виявлено.")
        return
    counts = issues["Проблема"].value_counts().rename_axis("Проблема").reset_index(name="Кількість")
    chart(px.bar(counts, x="Кількість", y="Проблема", orientation="h", color_discrete_sequence=COLORS))
    selected = st.multiselect("Фільтр за типом проблеми", counts["Проблема"].tolist(),
                              placeholder="Усі типи проблем")
    shown = issues[issues["Проблема"].isin(selected)] if selected else issues
    st.caption(f"Перші 1 000 із {len(shown):,} спрацювань. Повний список доступний в експорті.")
    table(shown.head(1000))


def cleaning(data):
    st.title(PAGES[4])
    frame = current(data)
    revision = data["revision"]
    action = st.selectbox("Метод очищення", ACTIONS, key=f"action{revision}")
    columns = st.multiselect("Стовпці для операції", frame.columns, key=f"clean_cols{revision}",
                             disabled=action == ACTIONS[0], placeholder="Оберіть стовпці")
    value = ""
    if action == ACTIONS[6]:
        value = st.text_input("Значення для заповнення")
    elif action == ACTIONS[15]:
        value = st.text_area("Канонічні категорії, по одній на рядок")
    left, middle, right = st.columns(3)
    if left.button("Додати до черги"):
        if action != ACTIONS[0] and not columns:
            st.error("Оберіть принаймні один стовпець.")
        elif action in (ACTIONS[6], ACTIONS[15]) and not value.strip():
            st.error("Заповніть параметр операції.")
        else:
            data["queue"].append(Operation(action, tuple(columns) if action != ACTIONS[0] else (),
                                           value))
            data["preview"] = None
    if middle.button("Очистити чергу"):
        data["queue"].clear()
        data["preview"] = None
    if right.button("Скасувати очищення", disabled=not data["history"]):
        data["history"].pop()
        data["preview"] = None
        st.session_state.pop("quality", None)
        st.session_state.pop("experiment", None)
        st.session_state.pop("export_file", None)
        st.rerun()
    if data["demo"] and st.button("Заповнити чергу методами експерименту"):
        data["queue"] = demo_operations()
        data["preview"] = None
    if data["queue"]:
        table(pd.DataFrame([{"Операція": op.action, "Стовпці": ", ".join(op.columns) or "Усі",
                             "Параметр": op.value} for op in data["queue"]]))
    if st.button("Побудувати попередній перегляд", disabled=not data["queue"], type="primary"):
        try:
            with st.spinner("Застосовуємо операції до копії…"):
                data["preview"] = (*clean(frame, data["queue"], data["method"]), data["method"])
        except (ValueError, TypeError, OverflowError):
            data["preview"] = None
            st.error("Не вдалося виконати операцію. Перевірте типи стовпців і параметри.")
    if data["preview"] and data["preview"][2] == data["method"]:
        result, entries, _ = data["preview"]
        st.subheader("Попередній результат")
        st.caption(f"Рядків після очищення — {len(result):,}. Метод пошуку аномалій — {data['method']}.")
        table(pd.DataFrame(entries))
        table(result.head(30))
        if st.button("Застосувати очищення", type="primary"):
            data["history"].append((result, entries))
            data["queue"].clear()
            data["preview"] = None
            st.session_state.pop("quality", None)
            st.session_state.pop("experiment", None)
            st.session_state.pop("export_file", None)
            st.rerun()
    if data["history"]:
        st.success(f"Застосовано пакетів очищення — {len(data['history'])}.")


def comparison(data):
    st.title(PAGES[5])
    before, initial, first_issues = assessment(data, True)
    after, final, last_issues = assessment(data)
    initial = initial if initial is not None else float("nan")
    final = final if final is not None else float("nan")
    change = 100 * (final - initial) / initial if initial > 0 else float("nan")
    cols = st.columns(4)
    cols[0].metric("Якість після", format_score(final) if pd.notna(final) else "Недостатньо даних")
    cols[1].metric("Проблем усунуто", f"{len(first_issues) - len(last_issues):,}", help="Різниця кількості спрацювань до і після.")
    cols[2].metric("Видалено рядків", f"{len(data['original']) - len(current(data)):,}")
    cols[3].metric("Зміна якості", f"{change:+.2f}%" if pd.notna(change) else "Недостатньо даних",
                   help="Відносна зміна від початкової оцінки.")
    scores = pd.DataFrame({"Метрика": ["Інтегральна оцінка", *METRICS],
                           "До очищення": [initial, *before.values()], "Після очищення": [final, *after.values()]})
    figure = scores.melt("Метрика", var_name="Стан", value_name="Оцінка, %").dropna()
    if not figure.empty:
        chart(px.bar(figure, x="Метрика", y="Оцінка, %", color="Стан", barmode="group",
                     range_y=[0, 103], color_discrete_sequence=COLORS))
    table(scores.map(lambda item: format_score(item) if isinstance(item, (float, int)) and pd.notna(item)
                     else "Недостатньо даних" if pd.isna(item) else item))
    issue_counts = pd.concat({
        "До очищення": first_issues["Проблема"].value_counts(),
        "Після очищення": last_issues["Проблема"].value_counts(),
    }, axis=1).fillna(0).rename_axis("Проблема").reset_index()
    if not issue_counts.empty:
        counts = issue_counts.melt("Проблема", var_name="Стан", value_name="Кількість")
        chart(px.bar(counts, y="Проблема", x="Кількість", color="Стан", barmode="group",
                     orientation="h", color_discrete_sequence=COLORS))
    st.subheader("Журнал очищення")
    if journal(data):
        table(pd.DataFrame(journal(data)))
    else:
        st.info("Операцій очищення ще немає.")


def experiment(data):
    st.title(PAGES[-2])
    if "financial_labels" in data or "transaction_id" in data["original"]:
        from finance.ui import effectiveness
        effectiveness(data)
        return
    st.write("Порівняння суми продажів, середнього чека, клієнтів і категорій за однаковою аналітичною процедурою.")
    if not data["demo"] or not REQUIRED.issubset(current(data).columns):
        st.info("Завантажте демонстраційні продажі, щоб провести експеримент.")
        return
    if st.button("Виміряти показники", type="primary"):
        with st.spinner("Виконуємо прогрів і сім вимірювань кожного сценарію…"):
            reference = data["demo_reference"]
            st.session_state.experiment = [benchmark(item) for item in (data["original"], current(data), reference)]
    results = st.session_state.get("experiment")
    if results is None:
        st.caption("Застосуйте обрані операції очищення перед вимірюванням, щоб порівняти два стани.")
        return
    (before, groups_before, time_before), (after, groups_after, time_after), (truth, groups_truth, time_truth) = results
    rows = []
    for name, expected in truth.items():
        rows.append({
            "Показник": name, "До": before[name], "Після": after[name], "Еталон": expected,
            "Точність до": format_score(relative_accuracy(before[name], expected)),
            "Точність після": format_score(relative_accuracy(after[name], expected)),
        })
    rows.append({"Показник": "Медіана часу, мс", "До": time_before, "Після": time_after, "Еталон": time_truth})
    table(pd.DataFrame(rows))
    groups = category_comparison(groups_before, groups_after, groups_truth).reset_index()
    groups["Категорія"] = groups["Категорія"].fillna("Без категорії")
    chart(px.bar(groups.melt("Категорія", var_name="Стан", value_name="Продажі"), x="Категорія",
                 y="Продажі", color="Стан", barmode="group", color_discrete_sequence=COLORS))
    for state in ("До", "Після"):
        groups[f"Точність ({state.lower()})"] = [
            format_score(relative_accuracy(value, expected))
            for value, expected in zip(groups[state], groups["Еталон"])
        ]
    table(groups)
    st.caption("Точність показника = 100 × max(0, 1 − |значення − еталон| / |еталон|). "
               "Вимірюється агрегація, без завантаження файлів і без очищення.")


def export(data):
    st.title(PAGES[-1])
    frame = current(data)
    scores, overall, issues = assessment(data)
    reports = [
        ("Очищені дані · CSV", "cleaned.csv", "text/csv", lambda: csv_bytes(frame)),
        ("Очищені дані · XLSX", "cleaned.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
         lambda: xlsx_bytes(frame)),
        ("Звіт про якість · HTML", "quality.html", "text/html",
         lambda: html_report(data["metadata"]["Файл"], frame, scores, overall, issues, journal(data))),
        ("Таблиця проблем · CSV", "issues.csv", "text/csv", lambda: csv_bytes(issues)),
        ("Журнал очищення · CSV", "cleaning_log.csv", "text/csv", lambda: csv_bytes(pd.DataFrame(
            journal(data), columns=["Операція", "Стовпці", "Параметр", "Видалено рядків", "Змінено комірок"]))),
    ]
    selected = st.radio("Файл для експорту", [item[0] for item in reports])
    _, filename, mime, prepare = next(item for item in reports if item[0] == selected)
    signature = (id(frame), id(issues), filename)
    if st.button("Підготувати файл", type="primary"):
        try:
            with st.spinner("Готуємо файл…"):
                st.session_state.export_file = (signature, prepare())
        except (ValueError, TypeError, OSError):
            st.error("Не вдалося створити файл для цих даних.")
    prepared = st.session_state.get("export_file")
    if prepared and prepared[0] == signature:
        st.download_button("Завантажити підготовлений файл", prepared[1], filename, mime, on_click="ignore")


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
    css = Path(__file__).with_name("assets").joinpath("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.title(APP_NAME)
        st.caption("Локальна система аналізу")
        selected = st.radio("Навігація", PAGES, label_visibility="collapsed")
    data = st.session_state.get("dataset")
    if selected == PAGES[0]:
        home(data)
    elif selected == PAGES[1]:
        upload()
    elif selected in PAGES[6:12]:
        from finance.ui import render
        render(selected,data)
    elif selected == PAGES[-2] and data is None:
        from finance.ui import effectiveness
        st.title(selected)
        effectiveness()
    elif data is None:
        st.title(selected)
        st.info("Спочатку імпортуйте таблицю або завантажте демонстраційний набір.")
    else:
        handlers = dict(zip((*PAGES[2:6],*PAGES[-2:]), (quality,problems,cleaning,comparison,experiment,export)))
        handlers[selected](data)


if __name__ == "__main__":
    main()

