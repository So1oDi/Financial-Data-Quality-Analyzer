import pandas as pd

from core.validation import dates, inspect, missing

METRICS = {
    "Повнота": (20, "Частка заповнених комірок серед усіх комірок."),
    "Унікальність": (15, "Частка рядків, які не є повторними входженнями повного рядка або заповненого ключа."),
    "Валідність": (20, "Частка успішних перевірок типів, дат, адрес та числових обмежень."),
    "Узгодженість": (15, "Частка успішних перевірок пробілів і належності до словника категорій."),
    "Точність": (20, "Частка еталонних комірок, відтворених за однозначним ключем. Відсутні та неоднозначні збіги знижують оцінку."),
    "Своєчасність": (10, "Частка рядків із коректною датою в межах заданого періоду до дати оцінки включно."),
}


def percentage(success, total):
    return 100 * success / total if total else None


def accuracy(frame, reference, keys):
    if reference is None or not keys or not set(keys).issubset(reference.columns):
        return None
    columns = [col for col in reference if col in frame and col not in keys]
    if not columns:
        return None
    expected = reference.loc[~pd.concat([missing(reference[key]) for key in keys], axis=1).any(axis=1), [*keys, *columns]]
    if expected.duplicated(list(keys)).any():
        return None
    actual = frame.loc[~pd.concat([missing(frame[key]) for key in keys], axis=1).any(axis=1), [*keys, *columns]]
    actual = actual.loc[~actual.duplicated(list(keys), keep=False)]
    expected = expected.set_index(list(keys))
    actual = actual.set_index(list(keys)).reindex(expected.index)
    comparable = ~expected.apply(missing)
    equal = actual.eq(expected).fillna(False)
    return percentage(int((equal & comparable).to_numpy().sum()), int(comparable.to_numpy().sum()))


def evaluate(frame, rules=None, keys=(), anomaly_columns=(), method="IQR", reference=None, freshness=None):
    issues, totals, duplicates = inspect(frame, rules, keys, anomaly_columns, method)
    valid, valid_bad = totals["valid"]
    consistent, consistent_bad = totals["consistent"]
    timely = None
    if freshness:
        column, as_of, days = freshness
        end = pd.Timestamp(as_of, tz="UTC").normalize()
        parsed = dates(frame[column]).dt.normalize()
        if parsed.notna().any():
            timely = percentage(int(parsed.between(end - pd.Timedelta(days=days), end).sum()), len(frame))
    scores = {
        "Повнота": percentage(frame.size - totals["missing"], frame.size),
        "Унікальність": percentage(len(frame) - duplicates, len(frame)),
        "Валідність": percentage(valid - valid_bad, valid),
        "Узгодженість": percentage(consistent - consistent_bad, consistent),
        "Точність": accuracy(frame, reference, keys),
        "Своєчасність": timely,
    }
    available = [(score, METRICS[name][0]) for name, score in scores.items() if score is not None]
    overall = percentage(sum(score * weight for score, weight in available), 100 * sum(weight for _, weight in available))
    return scores, overall, issues

