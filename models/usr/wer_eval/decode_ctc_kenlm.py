import argparse, json, os
import numpy as np
from pyctcdecode import build_ctcdecoder
from .normalize import normalize_text

def load_labels(vocab_path: str):
    # supports vocab.txt (one token per line) OR vocab.json mapping index->token
    if vocab_path.endswith(".json"):
        import json as _json
        m = _json.load(open(vocab_path, "r", encoding="utf-8"))
        # ensure list in index order
        items = sorted(((int(k), v) for k,v in m.items()), key=lambda x: x[0])
        return [v for _,v in items]
    else:
        return [l.rstrip("\n") for l in open(vocab_path, "r", encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctc_jsonl", required=True, help="Each line: {utt_id, ref, logprobs_path, vocab_path}")
    ap.add_argument("--lm_bin", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    # build decoder once using labels from first line
    first = json.loads(open(args.ctc_jsonl, "r", encoding="utf-8").readline())
    labels = load_labels(first["vocab_path"])
    decoder = build_ctcdecoder(labels, kenlm_model_path=args.lm_bin)

    with open(args.ctc_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            it = json.loads(line)
            lp = np.load(it["logprobs_path"])
            hyp = decoder.decode(lp, alpha=args.alpha, beta=args.beta)
            out = {
                "utt_id": it.get("utt_id"),
                "ref": it.get("ref",""),
                "hyp": normalize_text(hyp),
                "alpha": args.alpha,
                "beta": args.beta
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
