import pandas as pd
import plotly.express as px
import streamlit as st

from core.report import csv_bytes
from finance.data import FIELDS
from finance.mart import DB, load, read
from finance.models import MODEL_INPUT, backtest, cash, daily, fit_flow, indicators, isolation, predict_flow, select_flow
from finance.research import RESULTS

PAGES = ("ETL-пайплайн","Сховище даних","Фінансова аналітика","Підозрілі транзакції",
         "Прогноз грошових потоків","ML-експерименти")


def plot(fig):
    fig.update_layout(template="plotly_dark", margin=dict(t=45,l=10,b=10,r=10))
    st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})


def etl(data):
    st.write("Повне оновлення локальної вітрини з оригіналу вибраного набору. Попередній знімок замінюється після успішного завантаження.")
    if data and set(FIELDS).issubset(data["original"]):
        st.caption(data["metadata"]["Файл"])
        if st.button("Виконати повний ETL",type="primary"):
            load(data["original"], labels=data.get("financial_labels"))
            st.success("Вітрину оновлено.")
    else:
        st.info("Завантажте фінансовий CSV/XLSX або згенеруйте фінансові транзакції.")
    if DB.exists():
        history = read("select started, stage, seconds, processed, rejected, status from etl_log order by started desc")
        history.columns=["Час","Етап","Секунди","Оброблено","Відхилено","Стан"]
        st.dataframe(history,hide_index=True)
        if exists_mart():
            st.dataframe(read("select * from staging.quality"),hide_index=True)


def exists_mart():
    return DB.exists() and bool(read("select count(*) n from information_schema.tables where table_schema='mart' and table_name='operations'").iloc[0,0])


def warehouse():
    catalog = read("""select table_schema || '.' || table_name as name from information_schema.tables
                     where table_schema in ('raw','staging','mart') order by 1""")
    name = st.selectbox("Таблиця або SQL-представлення",catalog.name.tolist(),index=catalog.name.tolist().index("mart.daily"))
    st.code(f"SELECT * FROM {name} LIMIT 200",language="sql")
    st.metric("Записів",int(read(f"select count(*) from {name}").iloc[0,0]))
    st.dataframe(read(f"select * from {name} limit 200"),hide_index=True)
    with st.expander("Фізична структура"):
        st.dataframe(read(f"describe {name}"),hide_index=True)


def dashboard(frame):
    work = cash(frame)
    if work.empty:
        st.info("Проведених операцій немає.")
        return
    selected = st.date_input("Період",value=(work.transaction_date.min().date(),work.transaction_date.max().date()))
    if len(selected)==2:
        work = work[work.transaction_date.dt.date.between(*selected)]
    for col,label,slot in zip(["category","department","transaction_type"],["Категорії","Підрозділи","Типи"],st.columns(3)):
        values = slot.multiselect(label,sorted(work[col].unique()),placeholder="Усі")
        if values:
            work = work[work[col].isin(values)]
    st.caption("Касова модель у гривнях. Враховано лише проведені операції. Фінансовий результат дорівнює чистому потоку за обраний період.")
    metrics = list(indicators(work).items())
    for row in [metrics[:4],metrics[4:]]:
        for slot,(name,value) in zip(st.columns(len(row)),row):
            slot.metric(name,f"{value:,.2f}" if pd.notna(value) else "Недостатньо даних")
    if work.empty:
        return
    monthly = work.groupby(work.transaction_date.dt.to_period("M").astype(str))[["income","expense","net"]].sum()
    monthly.index.name = "Місяць"
    monthly.columns = ["Дохід","Витрати","Результат"]
    plot(px.line(monthly,y=["Дохід","Витрати"],title="Доходи й витрати за місяцями",labels={"value":"грн","transaction_date":"Місяць","variable":"Потік"}))
    plot(px.bar(monthly,y="Результат",title="Місячний фінансовий результат",labels={"transaction_date":"Місяць"}))
    expenses = work[work.expense.gt(0)].groupby("category").expense.sum()
    plot(px.pie(expenses.rename("Витрати, грн").rename_axis("Категорія").reset_index(),values="Витрати, грн",names="Категорія",title="Структура витрат"))
    for col,amount,title in [("department","income","Доходи підрозділів"),("category","net","Результат за категоріями")]:
        grouped = work.groupby(col)[amount].sum()
        plot(px.bar(x=grouped.index,y=grouped.values,title=title,labels={"x":"Група","y":"грн"}))
    plot(px.line(work.groupby("transaction_date").net.sum().cumsum().rename("Накопичений потік"),title="Накопичений чистий потік",labels={"value":"грн","transaction_date":"Дата","variable":"Показник"}))


def anomaly(frame,run):
    from finance.tracking import load_version, register
    ordered = frame.sort_values("transaction_date")
    if ordered.empty:
        st.info("Вітрина порожня.")
        return
    cutoff = ordered.transaction_date.iloc[int(len(ordered)*.8)]
    train,test = ordered[ordered.transaction_date.lt(cutoff)],ordered[ordered.transaction_date.ge(cutoff)]
    st.caption("Isolation Forest навчається на ранніх операціях. Позначка сигналізує про потребу перевірки.")
    if st.button("Навчити детектор і зберегти версію"):
        labels = read("select * from raw.evaluation_labels").set_index("transaction_id").suspicious
        model,output,metrics,matrix = isolation(train,test,labels)
        version = register(model,"FinancialIsolation",{"seed":42,"contamination":.02,"trees":80,"cutoff":str(cutoff),"etl":run},metrics)
        restored = load_version("FinancialIsolation",version)
        output["Підозріла"] = restored.predict(test[MODEL_INPUT]) == -1
        st.session_state.detector = run,version,output,metrics,matrix
    saved = st.session_state.get("detector")
    if saved and saved[0]==run:
        _,version,out,metrics,matrix = saved
        st.caption(f"Зареєстрована версія {version}. Тестових операцій — {len(out)}.")
        if metrics:
            st.dataframe(pd.DataFrame([metrics]),hide_index=True)
            st.dataframe(pd.DataFrame(matrix,index=["Факт: звичайна","Факт: підозріла"],columns=["Прогноз: звичайна","Прогноз: підозріла"]))
        else:
            st.info("Немає незалежних міток для оцінювання.")
        shown = out.assign(Статус=out["Підозріла"].map({True:"Потребує перевірки",False:"Звичайна"}))
        plot(px.scatter(shown,x="transaction_date",y="amount",color="Статус",hover_name="transaction_id",
                        labels={"transaction_date":"Дата","amount":"Сума, грн"}))
        flagged = out[out["Підозріла"]].sort_values("Оцінка аномальності",ascending=False)
        st.dataframe(flagged,hide_index=True)
        st.download_button("Завантажити позначені операції",csv_bytes(flagged),"review.csv")


def forecast(frame,run):
    from finance.tracking import load_version, register
    series = daily(frame)
    horizon = st.radio("Горизонт, днів",[7,30],horizontal=True)
    if len(series)<60+4*horizon:
        st.info("Період закороткий для навчання та тестового горизонту.")
        return
    if st.button("Навчити модель і спрогнозувати"):
        train = series.iloc[:-horizon]
        selected,cv = select_flow(train,horizon)
        test,metrics,baseline = backtest(train,series.iloc[-horizon:],selected)
        model = fit_flow(series,selected.name)
        version = register(model,"FinancialCashFlow",{"seed":42,"model":selected.name,"cv_folds":3,"horizon":horizon,
                           "etl":run,"last_date":str(series.index[-1])},metrics)
        future = predict_flow(load_version("FinancialCashFlow",version),series,horizon)
        st.session_state.flow = run,horizon,version,test,metrics,baseline,future,cv,selected.name
    saved = st.session_state.get("flow")
    if saved and len(saved)==9 and saved[:2]==(run,horizon):
        _,_,version,test,metrics,baseline,future,cv,name = saved
        st.caption(f"Версія {version}. Майбутній прогноз використовує модель, перенавчену на всій історії.")
        st.dataframe(pd.DataFrame([metrics,baseline],index=[name,"Базовий: попередній тиждень"]))
        st.dataframe(cv,hide_index=True)
        plot(px.line(series.rename("Потік"),title="Історичний щоденний потік",labels={"value":"грн","index":"Дата","variable":"Ряд"}))
        plot(px.line(test,title="Відкладений період",labels={"value":"грн","index":"Дата","variable":"Ряд"}))
        plot(px.line(future.rename("Прогноз"),title="Майбутній потік",labels={"value":"грн","index":"Дата","variable":"Ряд"}))
        st.download_button("Завантажити прогноз",csv_bytes(future.rename("Прогноз").rename_axis("Дата").reset_index()),"forecast.csv")


def effectiveness(data=None):
    source = st.radio("Джерело показників",["Контрольований дослід","Останній ETL"],horizontal=True)
    if source=="Останній ETL":
        if not exists_mart():
            st.info("Спочатку виконайте фінансовий ETL.")
            return
        raw = read("select * from raw.transactions")
        accepted = read("select * from mart.operations")
        quality = read("select * from staging.quality")
        log = read("""select * from etl_log where run=(select run from etl_log where stage='Load'
                      and status='Успішно' order by started desc limit 1)""")
        n = len(raw)
        for col,(label,val) in zip(st.columns(4),[
            ("Прийнято",len(accepted)),("Карантин",len(read("select * from staging.rejected"))),
            ("Збережено",f"{100*len(accepted)/n:.2f}%" if n else "Недостатньо даних"),
            ("Тривалість ETL",f"{log.seconds.sum():.3f} с")]):
            col.metric(label,val)
        plot(px.bar(quality.dropna(),x="Метрика",y="Оцінка",color="Стан",barmode="group",title="Якість поточної вітрини"))
        records = {"До":indicators(raw),"Після":indicators(accepted)}
        ref = data.get("reference") if data else None
        if ref is not None and set(FIELDS).issubset(ref) and raw.astype("string").equals(data["original"].astype("string")):
            records["Еталон"] = indicators(ref)
        values = pd.DataFrame(records).loc[["Дохід","Витрати","Фінансовий результат","Чистий потік"]]
        st.dataframe(values)
        if "Еталон" in values:
            deviation = values[["До","Після"]].sub(values["Еталон"],axis=0).abs()
            relative = deviation.div(values["Еталон"].abs().replace(0,float("nan")),axis=0)*100
            for title,df in [("Абсолютне відхилення, грн",deviation),("Відносне відхилення, %",relative)]:
                tidy = df.rename_axis("Показник").reset_index().melt("Показник",var_name="Стан",value_name=title)
                plot(px.bar(tidy,x="Показник",y=title,color="Стан",barmode="group",title=title))
        else:
            st.caption("Для відхилень потрібен еталон саме цього вхідного набору.")
        st.write(f"Касовий результат змінився на {values.loc['Фінансовий результат','Після']-values.loc['Фінансовий результат','До']:,.2f} грн.")
        st.caption("Парні ML-вимірювання для трьох фіксованих наборів доступні в режимі «Контрольований дослід».")
        return
    st.caption("Фінансовий дослід · seed 42 · однаковий незалежний тест для обох станів.")
    audit = pd.read_csv(RESULTS/"etl_effectiveness.csv")
    result = pd.read_csv(RESULTS/"financial_experiment.csv")
    size = st.selectbox("Обсяг дослідження",audit["Обсяг"].unique())
    audit = audit[audit["Обсяг"].eq(size)].set_index("Стан")
    result = result[result["Обсяг"].eq(size)]
    stats = audit.iloc[0]
    retained = 100*stats["Прийнято"]/stats["Вхідних"]
    for col,(name,value) in zip(st.columns(4),[
        ("Прийнято",int(stats["Прийнято"])),("Карантин",int(stats["Карантин"])),
        ("Збережено",f"{retained:.2f}%"),("Тривалість ETL",f"{stats['Час ETL, с']:.3f} с")]):
        col.metric(name,value)
    quality = audit.loc[:,"Повнота":].T.rename_axis("Метрика").reset_index()
    plot(px.bar(quality.melt("Метрика",var_name="Стан",value_name="Оцінка, %").dropna(),
                x="Метрика",y="Оцінка, %",color="Стан",barmode="group",title="Якість до та після ETL"))
    amounts = result[result["Метод"].eq("Аналітика")].copy()
    st.dataframe(amounts[["Дані","Показник","Значення","Еталон"]],hide_index=True)
    amounts["Абсолютне відхилення, грн"] = amounts["Відхилення"].abs()
    amounts["Відносне відхилення, %"] = 100*amounts["Відхилення"].abs()/amounts["Еталон"].abs().replace(0,float("nan"))
    for metric in ("Абсолютне відхилення, грн","Відносне відхилення, %"):
        plot(px.bar(amounts,x="Показник",y=metric,color="Дані",barmode="group",title=metric))
    ml = result[result["Метод"].isin(["Обраний прогноз","Isolation Forest"])]
    comparison = ml.pivot(index=["Метод","Показник"],columns="Дані",values="Значення")
    st.subheader("Вплив очищення на моделі")
    st.dataframe(comparison)
    st.write(f"Із {int(stats['Вхідних'])} рядків збережено {retained:.2f}%. "
             f"На звірку сум передано {int(stats['На звірку'])} прийнятих операцій.")
    for method,metric in [("Обраний прогноз","MAE"),("Isolation Forest","F1")]:
        row = comparison.loc[(method,metric)]
        delta = row["Очищені"]-row["Пошкоджені"]
        movement = "зросла" if delta>0 else "знизилася" if delta<0 else "не змінилася"
        st.write(f"Метрика {metric} {movement}: {row['Пошкоджені']:,.3f} → {row['Очищені']:,.3f}.")


def experiments():
    st.write("Парне дослідження пошкоджених та очищених даних на однакових тестових вибірках.")
    result = RESULTS/"financial_experiment.csv"
    if result.exists():
        table = pd.read_csv(result)
        size = st.selectbox("Обсяг транзакцій",table["Обсяг"].unique())
        detail = table[table["Обсяг"].eq(size) & table["Метод"].ne("Аналітика")]
        st.dataframe(detail.drop(columns=["Еталон","Відхилення"]),hide_index=True)
        cv = pd.read_csv(RESULTS/"forecast_validation.csv")
        st.dataframe(cv.loc[cv["Обсяг"].eq(size)].assign(Обрана=lambda x:x["Обрана"].map({True:"Так",False:"Ні"})),hide_index=True)
        st.download_button("Завантажити результати дослідження",result.read_bytes(),result.name)
    st.caption("Реєстр моделей і журнал навчання доступні після запуску start_mlflow.bat.")
    st.link_button("Відкрити локальний MLflow","http://127.0.0.1:5000")
    from finance.tracking import STORE, configure, load_version
    from mlflow import MlflowClient
    if not (STORE/"registry.db").exists() or not exists_mart():
        return
    configure()
    versions = {f"{v.name} · v{v.version}": v for v in MlflowClient().search_model_versions()
                if v.name in ("FinancialIsolation","FinancialCashFlow")}
    if not versions:
        return
    choice = st.selectbox("Збережена модель",list(versions))
    if st.button("Застосувати версію до поточної вітрини"):
        entry = versions[choice]
        model = load_version(entry.name,entry.version)
        frame = read("select * from mart.operations order by transaction_date,transaction_id")
        if entry.name == "FinancialIsolation":
            if frame.empty:
                st.info("Вітрина порожня.")
                return
            frame["Оцінка аномальності"] = -model.decision_function(frame[MODEL_INPUT])
            st.dataframe(frame.nlargest(100,"Оцінка аномальності"),hide_index=True)
        else:
            series = daily(frame)
            if len(series)<28:
                st.info("Потрібно 28 днів історії.")
                return
            st.dataframe(predict_flow(model,series,30).rename("Прогноз, грн"))


def render(page,data):
    st.title(page)
    try:
        if page==PAGES[0]:
            return etl(data)
        if page==PAGES[-1]:
            return experiments()
        if not exists_mart():
            st.info("Спочатку виконайте ETL для фінансового набору.")
            return
        if page==PAGES[1]:
            return warehouse()
        frame = read("select * from mart.operations")
        run = read("select run from etl_log where stage='Load' and status='Успішно' order by started desc limit 1").iloc[0,0]
        if page==PAGES[2]:
            dashboard(frame)
        elif page==PAGES[3]:
            anomaly(frame,run)
        else:
            forecast(frame,run)
    except Exception as error:
        st.error(f"Операцію перервано. {error}")

