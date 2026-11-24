from pathlib import Path
import os
import sqlite3
import subprocess
import sys
from contextlib import closing

import joblib
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import mlflow
import numpy as np
from tempfile import TemporaryDirectory
from mlflow import MlflowClient

STORE = Path(__file__).resolve().parents[1]/"data"/"mlflow"


def relocate_artifacts(store):
    """Після розпакування прив’язуємо URI історичних запусків до локальної копії."""
    database = store/"registry.db"
    artifacts = store/"artifacts"
    if not database.is_file() or not artifacts.is_dir():
        return
    target = artifacts.as_uri()
    with closing(sqlite3.connect(database)) as con, con:
        row = con.execute("select artifact_location from experiments where name=?",("Financial Ops",)).fetchone()
        if not row or row[0]==target or not row[0].startswith("file:"):
            return
        old = row[0].rstrip("/")
        if not old.endswith("/mlflow/artifacts"):
            return
        for table,column in (("runs","artifact_uri"),("model_versions","source"),("model_versions","storage_location")):
            con.execute(f"update {table} set {column}=? || substr({column},?) where substr({column},1,?)=?",
                        (target,len(old)+1,len(old)+1,old+"/"))
        con.execute("update experiments set artifact_location=? where name=?", (target,"Financial Ops"))


def configure(store=STORE):
    store = Path(store).resolve()
    store.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    relocate_artifacts(store)
    mlflow.set_tracking_uri("sqlite:///"+(store/"registry.db").as_posix())
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Financial Ops")
    return experiment.experiment_id if experiment else client.create_experiment("Financial Ops",artifact_location=(store/"artifacts").as_uri())


def register(model,name,params,metrics,store=STORE):
    experiment = configure(store)
    client = MlflowClient()
    with mlflow.start_run(experiment_id=experiment,run_name=name) as run, TemporaryDirectory() as folder:
        mlflow.log_params(params)
        mlflow.log_metrics({key.replace("²","2"):float(value) for key,value in metrics.items() if np.isfinite(value)})
        file = Path(folder)/"model.joblib"
        joblib.dump(model,file)
        mlflow.log_artifact(str(file),"model")
        from mlflow.exceptions import MlflowException
        try:
            client.create_registered_model(name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        version = client.create_model_version(name,mlflow.get_artifact_uri("model"),run.info.run_id)
    return int(version.version)


def load_version(name,version,store=STORE):
    configure(store)
    client = MlflowClient()
    entry = client.get_model_version(name,str(version))
    file = client.download_artifacts(entry.run_id,"model/model.joblib")
    return joblib.load(file)


if __name__ == "__main__":
    configure()
    args = [sys.executable,"-m","mlflow","server","--backend-store-uri",mlflow.get_tracking_uri(),
            "--host","127.0.0.1","--port","5000"]
    raise SystemExit(subprocess.call(args))
