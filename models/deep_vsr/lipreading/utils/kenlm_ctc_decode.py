from pyctcdecode import build_ctcdecoder
from .normalize_vn import normalize_text

def build_decoder(labels, kenlm_bin):
    return build_ctcdecoder(labels, kenlm_model_path=kenlm_bin)

def decode_ctc(log_probs, decoder, alpha=0.8, beta=0.5):
    text = decoder.decode(log_probs, alpha=alpha, beta=beta)
    return normalize_text(text)