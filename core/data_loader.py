"""Локальне читання CSV та XLSX з обмеженнями розміру."""
import csv
import io
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

MAX_ROWS = 250_000
MAX_CELLS = 2_500_000


def _decode(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "UTF-16"
    try:
        return raw.decode("utf-8-sig"), "UTF-8"
    except UnicodeDecodeError:
        return raw.decode("cp1251"), "Windows-1251"


def load_data(raw, name):
    if not raw or len(raw) > 25 * 1024**2:
        raise ValueError("Файл порожній або перевищує 25 МБ.")
    suffix = Path(name).suffix.lower()
    metadata = {"Файл": Path(name).name}
    try:
        if suffix == ".csv":
            content, encoding = _decode(raw)
            if "\x00" in content:
                raise ValueError
            try:
                separator = csv.Sniffer().sniff(content[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                # Пошкоджений рядок може завадити Sniffer розпізнати роздільник.
                header = next(csv.reader(io.StringIO(content)), [])
                separator = "," if len(header) > 1 else max(";\t|,", key=content.splitlines()[0].count)
            reader = csv.reader(io.StringIO(content), delimiter=separator, strict=True)
            header = next((row for row in reader if row), [])
            if not header or len(header) > 500:
                raise ValueError
            for count, row in enumerate((row for row in reader if row), 1):
                if len(row) != len(header) or count > MAX_ROWS or count * len(header) > MAX_CELLS:
                    raise ValueError
            frame = pd.read_csv(io.StringIO(content), sep=separator, keep_default_na=False, na_values=[""])
            metadata.update({"Кодування": encoding, "Роздільник": repr(separator)})
        elif suffix == ".xlsx":
            with ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) > 3000 or sum(item.file_size for item in members) > 100 * 1024**2:
                    raise ValueError
            with pd.ExcelFile(io.BytesIO(raw), engine="openpyxl", engine_kwargs={"keep_links": False}) as book:
                sheet = book.book.worksheets[0]
                if (sheet.max_column or 0) > 500:
                    raise ValueError
                limit = min(MAX_ROWS, MAX_CELLS // max(sheet.max_column or 500, 1))
                if (sheet.max_row or 0) > limit + 1:
                    raise ValueError
                frame = pd.read_excel(book, 0, nrows=limit + 1, keep_default_na=False, na_values=[""])
                if len(frame) > limit:
                    raise ValueError
                metadata["Аркуш"] = book.sheet_names[0]
        else:
            raise ValueError("Підтримуються файли CSV та XLSX.")
    except (BadZipFile, InvalidFileException, ParseError, UnicodeError, csv.Error, ValueError,
            OSError, IndexError, KeyError, SyntaxError) as error:
        raise ValueError("Не вдалося прочитати файл. Перевірте формат, структуру й розмір таблиці.") from error
    if not len(frame.columns) or len(frame.columns) > 500 or frame.size > MAX_CELLS:
        raise ValueError("Таблиця має містити від 1 до 500 стовпців і до 2,5 млн комірок.")
    frame.columns = frame.columns.map(str)
    return frame, metadata

