from jiwer import wer
from lipreading.utils.normalize_vn import normalize_text

def compute_wer(refs, hyps):
    refs = [normalize_text(r) for r in refs]
    hyps = [normalize_text(h) for h in hyps]
    return wer(refs, hyps)