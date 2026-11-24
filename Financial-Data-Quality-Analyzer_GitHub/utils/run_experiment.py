"""Відтворює приклади даних та фактичні вимірювання для курсової."""
from datetime import date
from pathlib import Path

from core.analytics import benchmark, relative_accuracy
from core.cleaning import clean, demo_operations
from core.metrics import evaluate
from utils.data_generator import KEYS, NUMERIC, RULES, SIZES, generate

ROOT = Path(__file__).resolve().parents[1]


def main():
    lines = [
        "# Практичний експеримент", "",
        "Генератор із seed 42 створює чисті замовлення та копію з контрольованими дефектами. "
        "У кожному сценарії застосовано однаковий пакет операцій із кнопки «Заповнити чергу методами експерименту». "
        "Заміна медіаною наближує відсутні значення. Точні ціни та кількості можна відновити лише за зовнішнім джерелом.",
        "",
        "Аналітичні операції — сума добутків кількості та ціни, середнє підсумків за замовленням, "
        "кількість унікальних клієнтів і сума продажів за категорією. Відсутні або нечислові суми не беруть участі "
        "в агрегації продажів. Від’ємні та аномальні числові значення впливають на агрегати до очищення.",
        "",
        "Час — медіана семи запусків після одного прогріву, у мілісекундах. "
        "Вимірюється лише агрегація. Час залежить від обладнання та навантаження.",
        "",
    ]
    for size in SIZES:
        reference, damaged = generate(size)
        if size == SIZES[0]:
            reference.to_csv(ROOT / "data/sales_reference_1000.csv", index=False, encoding="utf-8-sig")
            damaged.to_csv(ROOT / "data/sales_damaged_1000.csv", index=False, encoding="utf-8-sig")
        repaired, _ = clean(damaged, demo_operations())
        measurements = [benchmark(frame) for frame in (damaged, repaired, reference)]
        before, after, truth = [item[0] for item in measurements]
        options = dict(rules=RULES, keys=KEYS, anomaly_columns=NUMERIC, reference=reference,
                       freshness=("order_date", date(2025, 3, 31), 90))
        _, initial, initial_issues = evaluate(damaged, **options)
        _, final, final_issues = evaluate(repaired, **options)
        lines.extend([
            f"## {size:,} еталонних замовлень", "",
            f"Інтегральна оцінка — {initial:.2f}% → {final:.2f}%. "
            f"Спрацювання — {len(initial_issues):,} → {len(final_issues):,}. "
            f"Видалено рядків — {len(damaged) - len(repaired):,}.", "",
            "| Показник | До | Після | Еталон | Точність до | Точність після |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, expected in truth.items():
            lines.append(f"| {name} | {before[name]:,.2f} | {after[name]:,.2f} | {expected:,.2f} "
                         f"| {relative_accuracy(before[name], expected):.2f}% "
                         f"| {relative_accuracy(after[name], expected):.2f}% |")
        lines.append("| Медіана часу, мс | " + " | ".join(f"{item[2]:.3f}" for item in measurements) + " | — | — |")
        lines.extend(["", "| Категорія | Продажі до | Продажі після | Еталон |", "|---|---:|---:|---:|"])
        categories = sorted(set().union(*(set(item[1].index) for item in measurements)))
        for category in categories:
            values = " | ".join(f"{item[1].get(category, 0):,.2f}" for item in measurements)
            lines.append(f"| {category.strip()} | {values} |")
        lines.append("")
    lines.extend([
        "## Інтерпретація", "",
        "Великі аномальні ціни та дублікати зміщують суму продажів і середній чек. "
        "Нормалізація пробілів і зіставлення зі словником об’єднують різні записи однієї категорії. "
        "Той самий клієнт може бути присутнім в інших замовленнях, "
        "тому виправлення рядків не обов’язково змінює кількість клієнтів.", "",
        "Точність показника — `100 × max(0, 1 − |факт − еталон| / |еталон|)`. "
        "За нульового еталона двом нулям відповідає 100%, інакше відносна точність недоступна.",
    ])
    (ROOT / "docs/EXPERIMENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

