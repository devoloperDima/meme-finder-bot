"""Расширение запроса объяснениями зумерского сленга.

Плотная модель не знает, что «сигма» — это мем про независимого одиночку с Патриком Бейтманом:
на запрос «сигма» её сходство со всеми мемами почти одинаковое (0.45–0.51). Словарь
(slang.json, написан вручную) дописывает к запросу смысл слова и типичные образы мемов.

Ключи длиной до 3 символов совпадают только целым словом (иначе «пон» ловит «понедельник»),
длиннее — по началу слова («сигм» → сигма, сигмы, сигмой); ключи с пробелом — подстрокой.
"""

import json
import re
from functools import cache
from pathlib import Path

TOKEN_RE = re.compile(r"[\w🔥]+", re.UNICODE)


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


@cache
def _glossary() -> list[tuple[list[str], str]]:
    entries = json.loads((Path(__file__).with_name("slang.json")).read_text(encoding="utf-8"))
    return [([_norm(k) for k in e["keys"]], e["meaning"]) for e in entries]


def _matches(key: str, text: str, tokens: list[str]) -> bool:
    if " " in key:
        return key in text
    if len(key) <= 3:
        return key in tokens
    return any(t.startswith(key) for t in tokens)


def explain(query: str) -> list[str]:
    """Объяснения сленговых слов, найденных в запросе."""
    text = _norm(query)
    tokens = TOKEN_RE.findall(text)
    return [meaning for keys, meaning in _glossary() if any(_matches(k, text, tokens) for k in keys)]


def expand(query: str) -> str:
    meanings = explain(query)
    return f"{query}. {' '.join(meanings)}" if meanings else query
