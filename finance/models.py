import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support, r2_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from sklearn.preprocessing import OneHotEncoder

from core.validation import dates

CATEGORICAL = ["category","payment_method","transaction_type"]
NUMERIC = ["сума","відхилення","день","частота"]
MODEL_INPUT = ["amount","transaction_date","counterparty",*CATEGORICAL]


def amounts(frame):
    return pd.to_numeric(frame.amount, errors="coerce").replace([np.inf,-np.inf],np.nan)


def cash(frame):
    booked = frame.loc[frame.status.eq("Проведено") & frame.currency.eq("UAH")].copy()
    booked["transaction_date"] = dates(booked.transaction_date).dt.tz_localize(None).dt.normalize()
    booked["amount"] = amounts(booked)
    booked = booked.dropna(subset=["transaction_date","amount"])
    booked["income"] = booked.amount.where(booked.transaction_type.eq("Дохід"),0)
    booked["expense"] = booked.amount.where(booked.transaction_type.eq("Витрата"),0)
    booked["net"] = booked.income - booked.expense
    return booked


def indicators(frame):
    booked = cash(frame)
    income, expense, net = booked[["income","expense","net"]].sum()
    return {"Дохід":income,"Витрати":expense,"Фінансовий результат":net,
            "Чистий потік":net,"Рентабельність, %":100*net/income if income else np.nan,
            "Проведені операції":len(booked),"Середня сума":booked.amount.mean()}


def daily(frame, index=None):
    booked = cash(frame)
    series = booked.groupby("transaction_date").net.sum()
    if index is None and not series.empty:
        index = pd.date_range(series.index.min(),series.index.max(),freq="D")
    result = series.reindex(index,fill_value=0) if index is not None else series
    rent = booked.loc[booked.category.eq("Оренда")].groupby("transaction_date").net.sum()
    result.attrs["rent"] = {day.date():float(value) for day,value in rent.items()}
    return result


class OperationsFeatures(TransformerMixin,BaseEstimator):
    def fit(self, X, y=None):
        values = amounts(X)
        self.typical_ = values.groupby(X.category).median().to_dict()
        self.median_ = float(values.median()) if values.notna().any() else 0.
        self.frequency_ = X.counterparty.value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        values = amounts(X).fillna(self.median_)
        typical = X.category.map(self.typical_).fillna(self.median_)
        out = X[CATEGORICAL].astype("string").fillna("Невідомо").copy()
        out["сума"] = values
        out["відхилення"] = (values-typical) / (typical.abs()+1)
        out["день"] = dates(X.transaction_date).dt.day_of_week.fillna(-1)
        out["частота"] = X.counterparty.map(self.frequency_).fillna(0)
        return out


def isolation(train, test, labels=None):
    if len(train)<10 or test.empty:
        raise ValueError("Замало операцій для навчання або перевірки.")
    prep = ColumnTransformer([("num","passthrough",NUMERIC),
                              ("cat",OneHotEncoder(handle_unknown="ignore"),CATEGORICAL)])
    model = Pipeline([("features",OperationsFeatures()),("encode",prep),
                      ("forest",IsolationForest(n_estimators=80,contamination=.02,random_state=42))])
    model.fit(train[MODEL_INPUT])
    result = test.copy()
    result["Оцінка аномальності"] = -model.decision_function(test[MODEL_INPUT])
    result["Підозріла"] = result["Оцінка аномальності"].gt(0)
    metrics, matrix = {}, None
    if labels is not None:
        truth = result.transaction_id.map(labels)
        known = truth.notna()
        if known.any():
            y, pred = truth[known].astype(bool), result.loc[known,"Підозріла"]
            metrics = dict(zip(["Precision","Recall","F1"],precision_recall_fscore_support(
                y,pred,average="binary",zero_division=0)[:3]))
            matrix = confusion_matrix(y,pred,labels=[False,True])
    return model,result,metrics,matrix


def time_features(day, history):
    angle = 2*np.pi*day.day_of_year/365.25
    return [day.day_of_week,day.month,np.sin(angle),np.cos(angle),history[-1],history[-7],
            np.mean(history[-7:]),np.mean(history[-28:])]


CANDIDATES = ("Медіана за 28 днів","Попередній тиждень","Random Forest","HistGradientBoosting")


class FlowModel:
    def __init__(self, name, estimator=None):
        self.name = name
        self.estimator = estimator


def calendar_features(day, history, rent, cutoff):
    # Використовуємо лише платежі, відомі на момент побудови прогнозу.
    prior = [day - pd.DateOffset(months=i) for i in (1,2,3,4)]
    rent_history = [rent.get(d.date(),0.) for d in prior if d <= cutoff and d.day == day.day][:3]
    weekly = 2*np.pi*day.day_of_week/7
    return time_features(day,history) + [
        float(day.is_month_start),float(day.is_month_end),day.day,day.days_in_month-day.day,
        np.sin(weekly),np.cos(weekly),np.median(history[-28:]),
        np.median(rent_history) if rent_history else 0.,
    ]


def fit_flow(series,name="Random Forest"):
    if len(series)<60:
        raise ValueError("Для навчання потрібно щонайменше 60 календарних днів.")
    if name in CANDIDATES[:2]:
        return FlowModel(name)
    estimators = {
        "Random Forest":RandomForestRegressor(n_estimators=60,min_samples_leaf=3,random_state=42),
        "HistGradientBoosting":HistGradientBoostingRegressor(max_iter=65,max_leaf_nodes=12,
                                                            early_stopping=False,random_state=42),
    }
    values = series.to_numpy()
    rent = series.attrs.get("rent",{})
    X = [calendar_features(series.index[i],values[i-28:i],rent,series.index[i-1]) for i in range(28,len(series))]
    with threadpool_limits(limits=1):
        return FlowModel(name,estimators[name].fit(X,values[28:]))


def predict_flow(model, history, horizon):
    past = list(history.iloc[-28:])
    days = pd.date_range(history.index[-1]+pd.Timedelta(days=1), periods=horizon)
    if isinstance(model,FlowModel):
        if model.name==CANDIDATES[0]:
            return pd.Series(np.median(past),index=days)
        if model.name==CANDIDATES[1]:
            return pd.Series(np.resize(past[-7:],horizon),index=days)
    with threadpool_limits(limits=1):
        for day in days:
            if isinstance(model,FlowModel):
                features = calendar_features(day,past,history.attrs.get("rent",{}),history.index[-1])
                value = model.estimator.predict([features])[0]
            else:
                # Історичні версії MLflow очікують початковий набір із восьми ознак.
                value = model.predict([time_features(day,past)])[0]
            past.append(float(value))
    return pd.Series(past[-horizon:], index=days)


def select_flow(train,horizon):
    # Три послідовні неперетинні блоки. Незалежний тест не передається.
    if len(train)<60+3*horizon:
        raise ValueError("Для трьох блоків валідації потрібна довша історія.")
    rows = []
    for fold in range(3):
        end = len(train)-(3-fold)*horizon
        history,valid = train.iloc[:end],train.iloc[end:end+horizon]
        for name in CANDIDATES:
            prediction = predict_flow(fit_flow(history,name),history,horizon)
            rows.append({"Модель":name,"Блок":fold+1,"Кінець навчання":history.index[-1],
                         "Початок перевірки":valid.index[0],"Кінець перевірки":valid.index[-1],
                         **errors(valid,prediction)})
    report = pd.DataFrame(rows)
    winner = report.groupby("Модель",sort=False).MAE.mean().idxmin()
    return fit_flow(train,winner),report


def errors(actual,pred):
    return {"MAE":mean_absolute_error(actual,pred),"RMSE":np.sqrt(mean_squared_error(actual,pred)),
            "R²":r2_score(actual,pred) if actual.nunique()>1 else np.nan}


def backtest(train, truth, model=None):
    if model is None:
        model = fit_flow(train)
    prediction = predict_flow(model,train,len(truth))
    if not prediction.index.equals(truth.index):
        raise ValueError("Тестовий період має йти відразу після навчального.")
    baseline = np.resize(train.iloc[-7:].to_numpy(),len(truth))
    return pd.DataFrame({"Факт":truth,"Обрана модель":prediction,"Базовий":baseline}), errors(truth,prediction), errors(truth,baseline)

