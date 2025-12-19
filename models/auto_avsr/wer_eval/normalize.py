import re
import unicodedata

_PUNCT = r"[\.,;:!?…\"']"
_SPACE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = re.sub(_PUNCT, "", s)
    s = _SPACE.sub(" ", s).strip()
    return s
