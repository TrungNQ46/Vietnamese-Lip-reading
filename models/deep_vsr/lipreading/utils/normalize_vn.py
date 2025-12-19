import re
import unicodedata

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[.,;:!?…\"']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s