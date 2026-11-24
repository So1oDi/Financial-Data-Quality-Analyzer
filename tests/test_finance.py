import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from finance.data import generate, transform
from finance.mart import load, read
from finance.models import MODEL_INPUT, OperationsFeatures, backtest, cash, daily, fit_flow, indicators, isolation, predict_flow
from finance.research import compare
from finance.tracking import load_version, register
from core.report import csv_bytes


def test_financial_validation():
    ref,raw,labels = generate()
    a,b = transform(raw)
    assert len(a)+len(b)==len(raw) and a.transaction_id.is_unique
    assert len(b)==70
    assert set(ref.loc[labels.values,"transaction_id"]).issubset(a.transaction_id)
    assert (a.amount==950000).any()
    assert a.department.str.strip().eq(a.department).all()
    assert not any(c in raw.columns for c in [labels.name,"suspicious"])
    with pytest.raises(ValueError):
        transform(raw.drop(columns="status"))


def test_atomic_idempotent_mart(tmp_path):
    _,raw,labels = generate()
    path = tmp_path/"test.duckdb"
    for _ in range(2):
        load(raw,path,labels)
    assert len(read("select * from mart.operations",path))==940
    assert read("select count(*) from raw.transactions",path).iat[0,0]==1010
    assert len(read("select * from etl_log",path))==6
    assert read("select count(*) from duckdb_constraints() where constraint_type='FOREIGN KEY' and table_name='fact_transactions'",path).iat[0,0]==6
    with pytest.raises(ValueError):
        load(raw.drop(columns="amount"),path)
    assert len(read("select * from mart.fact_transactions",path))==940
    assert len(read("select * from raw.evaluation_labels",path))==1000
    assert abs(read("select sum(net_flow) from mart.monthly",path).iat[0,0]-cash(transform(raw)[0]).net.sum())<.01
    with pytest.raises(Exception):
        load(raw,path,pd.concat([labels,labels]))
    assert len(read("select * from mart.fact_transactions",path))==940


def test_financial_indicators():
    ref = generate()[0].iloc[:3].copy()
    ref["transaction_type"]=["Дохід","Витрата","Дохід"]
    ref["amount"]=[120,50,80]
    ref["status"]=["Проведено","Проведено","Очікує"]
    result = indicators(ref)
    assert result["Чистий потік"]==70 and result["Рентабельність, %"]==pytest.approx(100*70/120)
    assert result["Середня сума"]==85


def test_features_and_isolation():
    ref,_,labels = generate()
    train,test = ref.iloc[:800],ref.iloc[800:]
    feat = OperationsFeatures().fit(train)
    test = test.copy()
    test["counterparty"]="Новий"
    out = feat.transform(test)
    assert out["частота"].eq(0).all()
    model,scores,metrics,cm = isolation(train,test,labels)
    assert len(model.predict(test[MODEL_INPUT]))==len(scores)
    assert cm.sum()==len(test) and 0<=metrics["F1"]<=1


def test_forecast_and_baseline():
    series = daily(generate()[0])
    for horizon in [7,30]:
        train,truth=series.iloc[:-horizon],series.iloc[-horizon:]
        test,metrics,base=backtest(train,truth)
        assert test.index.equals(truth.index)
        assert np.array_equal(test["Базовий"],np.resize(train.iloc[-7:],horizon))
        assert metrics["MAE"]>=0 and base["RMSE"]>=0
        assert predict_flow(fit_flow(train),train,horizon).index.equals(truth.index)


def test_registry_inference(tmp_path):
    X=np.arange(10).reshape(-1,1)
    model=RandomForestRegressor(n_estimators=2,random_state=42).fit(X,X.ravel())
    for expected in [1,2]:
        v=register(model,"TestFlow",{"trees":2},{"MAE":1.},tmp_path)
        assert v==expected
    assert np.array_equal(load_version("TestFlow",1,tmp_path).predict(X),model.predict(X))


def test_paired_experiment_export():
    results=compare(1000)
    assert len(results)==26
    assert results.groupby(["Метод","Показник"]).size().eq(2).all()
    assert "Відхилення" in csv_bytes(results).decode("utf-8-sig")



def test_quarantine_and_review_separate():
    from finance.data import amount_review
    ref,raw,_ = generate()
    clean,_ = transform(raw)
    flagged = amount_review(clean)
    assert set(flagged.transaction_id).issubset(clean.transaction_id)
    assert len(flagged)==10
    example = ref.iloc[:3].copy()
    example["document_amount"] = example.amount
    example.iloc[0,example.columns.get_loc("amount")] += 5
    example.iloc[1,example.columns.get_loc("amount")] = 950001
    example.iloc[1,example.columns.get_loc("document_amount")] = 950001
    valid,rejected = transform(example)
    assert len(valid)==2 and len(rejected)==1 and "документа" in rejected.iloc[0,-1]
    conflict = pd.concat([ref.iloc[:1],ref.iloc[:1].assign(amount=1)])
    assert transform(conflict)[1]["Причина відхилення"].str.contains("Суперечливі").all()


def test_temporal_selection_and_legacy_inference():
    from finance.models import CANDIDATES, calendar_features, select_flow, time_features
    series = daily(generate()[0])
    train = series.iloc[:-30]
    winner,folds = select_flow(train,30)
    assert len(folds)==3*len(CANDIDATES)
    assert (folds["Кінець навчання"] < folds["Початок перевірки"]).all()
    assert folds["Кінець перевірки"].max()==train.index[-1]
    assert winner.name==folds.groupby("Модель").MAE.mean().idxmin()
    dates = pd.date_range("2024-01-01",periods=120)
    past = np.arange(28)
    known = {d.date():-100 for d in dates if d.day==1}
    truncated = {d:v for d,v in known.items() if d<=dates[60].date()}
    assert calendar_features(dates[91],past,known,dates[60])==calendar_features(dates[91],past,truncated,dates[60])
    old = RandomForestRegressor(n_estimators=2,random_state=42).fit(
        [time_features(day,past) for day in dates[:2]],[0,1])
    assert len(predict_flow(old,train,7))==7
