import re
from dataclasses import dataclass
from typing import Dict, Tuple

URL_RE = re.compile(r"https?://[^\s]+")
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
MONEY_RE = re.compile(r"[$¥€£]\s?\d[\d,]*(?:\.\d+)?")
CODE_RE = re.compile(r"`[^`]+`|[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)")
NUM_RE = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?%?\b")

PATTERNS = [
    ("URL", URL_RE),
    ("EMAIL", EMAIL_RE),
    ("MONEY", MONEY_RE),
    ("CODE", CODE_RE),
    ("NUM", NUM_RE),
]


@dataclass
class ProtectedText:
    text: str
    mapping: Dict[str, str]


def protect(text: str, max_placeholders: int = 100) -> ProtectedText:
    mapping: Dict[str, str] = {}
    protected = text

    for _, pattern in PATTERNS:
        def repl(match):
            if len(mapping) >= max_placeholders:
                return match.group(0)
            key = f"<PH{len(mapping)}>"
            mapping[key] = match.group(0)
            return key
        protected = pattern.sub(repl, protected)

    return ProtectedText(protected, mapping)


def restore(text: str, mapping: Dict[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text
