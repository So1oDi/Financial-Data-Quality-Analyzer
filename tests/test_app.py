from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def push(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def navigate(app, section):
    app.sidebar.radio[0].set_value(section).run()
    assert not app.exception


def test_streamlit_demo_workflow():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    navigate(app, "Завантаження даних")
    push(app, "Завантажити демодані")
    navigate(app, "Головна")
    assert app.metric[0].label == "Інтегральна оцінка"
    navigate(app, "Аналіз якості даних")
    push(app, "Зберегти правило")
    assert len(app.metric) == 7
    navigate(app, "Виявлені проблеми")
    assert app.dataframe
    navigate(app, "Очищення даних")
    push(app, "Заповнити чергу методами експерименту")
    push(app, "Побудувати попередній перегляд")
    assert not app.session_state.dataset["history"]
    push(app, "Застосувати очищення")
    assert len(app.session_state.dataset["history"][-1][0]) == 1000
    navigate(app, "Порівняння результатів")
    assert app.metric[2].value == "20"
    navigate(app, "Ефективність аналітичних систем")
    push(app, "Виміряти показники")
    assert app.dataframe
    navigate(app, "Експорт результатів")
    for option in app.main.radio[0].options:
        app.main.radio[0].set_value(option).run()
        push(app, "Підготувати файл")
        assert app.get("download_button")
    navigate(app, "Очищення даних")
    push(app, "Скасувати очищення")
    assert not app.session_state.dataset["history"]



def test_financial_defense_workflow(tmp_path,monkeypatch):
    import finance.ui as ui
    from finance import mart, tracking
    from io import BytesIO
    from openpyxl import load_workbook
    import pandas as pd

    db = tmp_path/"financial.duckdb"
    store = tmp_path/"mlflow"
    monkeypatch.setattr(ui,"DB",db)
    monkeypatch.setattr(ui,"load",lambda raw,labels=None: mart.load(raw,db,labels))
    monkeypatch.setattr(ui,"read",lambda sql: mart.read(sql,db))
    register,load = tracking.register,tracking.load_version
    configure = tracking.configure
    monkeypatch.setattr(tracking,"STORE",store)
    monkeypatch.setattr(tracking,"configure",lambda location=store: configure(location))
    monkeypatch.setattr(tracking,"register",lambda *args:register(*args,store=store))
    monkeypatch.setattr(tracking,"load_version",lambda *args:load(*args,store=store))
    app = AppTest.from_file(str(APP),default_timeout=90).run()
    pages = app.sidebar.radio[0].options
    assert len(pages)==14
    navigate(app,pages[1])
    push(app,"Згенерувати фінансові транзакції")
    for page in pages:
        navigate(app,page)
        assert not app.error, [item.value for item in app.error]
        if page=="ETL-пайплайн":
            for _ in range(2):
                push(app,"Виконати повний ETL")
                assert not app.error, [item.value for item in app.error]
            assert len(ui.read("select * from mart.operations"))==940
        elif page=="Підозрілі транзакції":
            push(app,"Навчити детектор і зберегти версію")
            assert not app.error and app.dataframe
        elif page=="Прогноз грошових потоків":
            push(app,"Навчити модель і спрогнозувати")
            assert not app.error and app.get("download_button")
        elif page=="ML-експерименти":
            push(app,"Застосувати версію до поточної вітрини")
            assert not app.error, [item.value for item in app.error]
        elif page=="Ефективність аналітичних систем":
            assert len(app.metric)==4 and len(app.get("plotly_chart"))==3
            app.main.radio[0].set_value("Останній ETL").run()
            assert not app.exception and not app.error
            assert len(app.metric)==4 and app.metric[0].value=="940"
            assert len(app.get("plotly_chart"))==3
        elif page=="Експорт результатів":
            for option in app.main.radio[0].options:
                app.main.radio[0].set_value(option).run()
                push(app,"Підготувати файл")
                signature,content = app.session_state.export_file
                filename = signature[-1]
                if filename.endswith(".xlsx"):
                    book = load_workbook(BytesIO(content),read_only=True)
                    assert next(book.active.values)[0]=="transaction_id"
                    book.close()
                elif filename.endswith(".csv"):
                    assert len(pd.read_csv(BytesIO(content)).columns)>0
                else:
                    assert "<html" in content and "Інтегральна" in content
