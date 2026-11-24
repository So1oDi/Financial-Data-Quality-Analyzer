from pathlib import Path
from time import perf_counter
from uuid import uuid4

import duckdb
import pandas as pd

from core.metrics import evaluate
from finance.data import DIMS, FIELDS, RULES, amount_review, transform

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/financial.duckdb"
SOURCE = {d: ("transaction_date" if d == "date" else d) for d in DIMS}
AGG = """sum(case when transaction_type='Дохід' then amount else 0 end) as income,
sum(case when transaction_type='Витрата' then amount else 0 end) as expense,
sum(case when transaction_type='Дохід' then amount else -amount end) as net_flow"""


def load(raw, path=DB, labels=None):
    run = uuid4().hex
    steps = []
    with duckdb.connect(str(path)) as con:
        con.execute("""create table if not exists etl_log(run varchar, started timestamp default current_timestamp,
                    stage varchar, seconds double, processed bigint, rejected bigint, status varchar)""")
        stage = "Extract"
        start = perf_counter()
        try:
            if not set(FIELDS).issubset(raw):
                raise ValueError("Для ETL потрібні всі 11 полів фінансових транзакцій.")
            source = raw.astype("string")
            steps.append((stage, perf_counter()-start,len(raw),0,"Успішно"))
            stage, start = "Transform", perf_counter()
            accepted, rejected = transform(raw)
            scores = []
            for name, frame in [("До",raw),("Після",accepted)]:
                metrics, overall, _ = evaluate(frame, RULES, ("transaction_id",))
                scores.extend((name,key,value) for key,value in {**metrics,"Інтегральна":overall}.items())
            quality = pd.DataFrame(scores,columns=["Стан","Метрика","Оцінка"])
            steps.append((stage,perf_counter()-start,len(raw),len(rejected),"Успішно"))
            stage, start = "Load", perf_counter()
            con.begin()
            for schema in ("mart","staging","raw"):
                con.execute(f"drop schema if exists {schema} cascade")
                con.execute(f"create schema {schema}")
            for table, frame in [("raw.transactions",source),("staging.transactions",accepted),
                                 ("staging.amount_review",amount_review(accepted)),("staging.rejected",rejected.astype("string")),("staging.quality",quality)]:
                con.register("incoming",frame)
                con.execute(f"create table {table} as select * from incoming")
                con.unregister("incoming")
            con.execute("create table raw.evaluation_labels(transaction_id varchar primary key, suspicious boolean)")
            if labels is not None:
                con.register("labels_input",labels.rename("suspicious").rename_axis("transaction_id").reset_index())
                con.execute("insert into raw.evaluation_labels select transaction_id,suspicious from labels_input")
            joins, keys = [], []
            for dim, col in SOURCE.items():
                dtype = "date" if dim == "date" else "varchar"
                con.execute(f"create table mart.dim_{dim}(id integer primary key, value {dtype} unique not null)")
                con.execute(f"""insert into mart.dim_{dim} select row_number() over(order by value), value
                    from (select distinct cast({col} as {dtype}) as value from staging.transactions)""")
                joins.append(f"join mart.dim_{dim} d_{dim} on d_{dim}.value=s.{col}")
                keys.append(f"{dim}_id integer not null references mart.dim_{dim}(id)")
            con.execute(f"""create table mart.fact_transactions(transaction_id varchar primary key,
                {','.join(keys)}, amount decimal(14,2) check(amount>0), currency varchar, status varchar, description varchar)""")
            dim_ids = ",".join(f"d_{d}.id" for d in DIMS)
            con.execute(f"""insert into mart.fact_transactions select s.transaction_id,{dim_ids},
                s.amount,s.currency,s.status,s.description from staging.transactions s {' '.join(joins)}""")
            if con.execute("select count(*) from mart.fact_transactions").fetchone()[0] != len(accepted):
                raise ValueError("Порушення цілісності вимірів.")
            fields = ",".join(f"d_{d}.value as {col}" for d,col in SOURCE.items())
            lookup = " ".join(f"join mart.dim_{d} d_{d} on f.{d}_id=d_{d}.id" for d in DIMS)
            con.execute(f"""create view mart.operations as select f.transaction_id,{fields},f.amount,f.currency,
                          f.status,f.description from mart.fact_transactions f {lookup}""")
            for view, group in [("daily","transaction_date"),("monthly","date_trunc('month',transaction_date)"),
                                ("categories","category"),("departments","department")]:
                con.execute(f"""create view mart.{view} as select {group} as period,{AGG} from mart.operations
                             where status='Проведено' group by 1 order by 1""")
            steps.append((stage,perf_counter()-start,len(accepted),len(rejected),"Успішно"))
            _log(con,run,steps)
            con.commit()
        except Exception as error:
            if stage == "Load":
                con.rollback()
            _log(con,run,steps+[(stage,perf_counter()-start,len(raw),0,str(error))])
            raise
    return steps


def _log(con,run,steps):
    con.executemany("insert into etl_log(run,stage,seconds,processed,rejected,status) values (?,?,?,?,?,?)",
                    [(run,*step) for step in steps])


def read(query, path=DB):
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute(query).df()

