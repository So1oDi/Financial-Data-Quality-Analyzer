from pathlib import Path
from tempfile import TemporaryDirectory
from finance.mart import load, read

import pandas as pd

from finance.data import generate, transform
from finance.models import backtest, daily, select_flow, indicators, isolation
from utils.data_generator import SIZES

RESULTS = Path(__file__).resolve().parents[1]/"docs"/"results"


def compare(size, track=False, folds=None, audit=None):
    ref, raw, labels = generate(size)
    clean,_ = transform(raw)
    if audit is not None:
        with TemporaryDirectory() as folder:
            db = Path(folder)/"experiment.duckdb"
            steps = load(raw,db,labels)
            quality = read("select * from staging.quality",db)
            for state in ("До","Після"):
                values = quality.loc[quality["Стан"].eq(state)].set_index("Метрика")["Оцінка"].to_dict()
                audit.append({"Обсяг":size,"Стан":state,"Вхідних":len(raw),"Прийнято":len(clean),
                              "Карантин":len(raw)-len(clean),"На звірку":len(read("select * from staging.amount_review",db)),
                              "Час ETL, с":sum(step[1] for step in steps),**values})
    cutoff = ref.transaction_date.sort_values().iloc[int(.8*len(ref))]
    test = ref[ref.transaction_date.ge(cutoff)]
    ids = ref.loc[ref.transaction_date.lt(cutoff),"transaction_id"]
    truth = daily(ref)
    holdout = truth.iloc[-30:]
    train_days = truth.index[:-30]
    train_ids = ref.loc[ref.transaction_date.lt(holdout.index[0].strftime("%Y-%m-%d")),"transaction_id"]
    reference = indicators(ref)
    rows = []
    for state,frame in [("Пошкоджені",raw),("Очищені",clean)]:
        metrics = indicators(frame)
        detector,_,classification,_ = isolation(frame[frame.transaction_id.isin(ids)],test,labels)
        series = daily(frame[frame.transaction_id.isin(train_ids)],train_days)
        forest,validation = select_flow(series,len(holdout))
        if folds is not None:
            folds.append(validation.assign(Обсяг=size,Дані=state,Обрана=validation["Модель"].eq(forest.name)))
        _,regression,baseline = backtest(series,holdout,forest)
        if track:
            from finance.tracking import register
            params = {"size":size,"state":state,"seed":42}
            register(detector,"FinancialIsolation",{**params,"cutoff":cutoff,"trees":80,"contamination":.02},classification)
            register(forest,"FinancialCashFlow",{**params,"horizon":30,"last_date":str(series.index[-1]),
                     "model":forest.name,"selection":"3-fold temporal MAE"},regression)
        for key in ["Дохід","Витрати","Фінансовий результат","Чистий потік"]:
            rows.append((size,state,"Аналітика",key,metrics[key],reference[key],metrics[key]-reference[key]))
        for group,values in [("Isolation Forest",classification),("Обраний прогноз",regression),("Базовий прогноз",baseline)]:
            rows.extend((size,state,group,key,value,None,None) for key,value in values.items())
    return pd.DataFrame(rows,columns=["Обсяг","Дані","Метод","Показник","Значення","Еталон","Відхилення"])


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    folds,audit = [],[]
    table = pd.concat([compare(size,track=True,folds=folds,audit=audit) for size in SIZES],ignore_index=True)
    pd.concat(folds).to_csv(RESULTS/"forecast_validation.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(RESULTS/"etl_effectiveness.csv",index=False,encoding="utf-8-sig")
    table.to_csv(RESULTS/"financial_experiment.csv",index=False,encoding="utf-8-sig")
    pivot = table.pivot(index=["Обсяг","Метод","Показник"],columns="Дані",values="Значення").reset_index()
    lines = ["# Вплив якості фінансових даних","",
        "Для кожного обсягу використано seed 42. Обидві моделі детекції оцінено на тих самих еталонних операціях останніх 20% хронології. "
        "Навчальні ідентифікатори визначено до поділу. Для регресії обидві версії прогнозують ті самі останні 30 днів еталона з фіксованої точки відсікання. "
        "У дотестовій частині чотири кандидати проходять три послідовні блоки по 30 днів. Мінімальна середня MAE визначає метод до відкриття тесту. Базовий тижневий прогноз також наведено окремо. Пропущені дні мають нульовий потік.","",
        "| Обсяг | Метод | Показник | Пошкоджені | Очищені |","|---:|---|---|---:|---:|"]
    for size,method,metric,clean,raw in pivot.itertuples(index=False,name=None):
        lines.append(f"| {size} | {method} | {metric} | {raw:.4f} | {clean:.4f} |")
    lines.extend(["","Великі додатні спотворені суми проходять структурні перевірки. Відхилення записів із помилками також зменшує покриття. "
                  "Тому ETL може покращити окремі оцінки й погіршити інші. Парні значення вище показують напрям зміни; повні відхилення від еталона наведені у CSV."])
    (RESULTS/"FINANCIAL_EXPERIMENT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()

