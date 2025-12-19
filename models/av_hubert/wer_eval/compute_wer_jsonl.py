import argparse, json
from jiwer import wer
from .normalize import normalize_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--ref_key", default="ref")
    ap.add_argument("--hyp_key", default="hyp")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    refs, hyps = [], []
    n = 0
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if args.ref_key not in it or args.hyp_key not in it:
                continue
            ref = normalize_text(it[args.ref_key])
            hyp = normalize_text(it[args.hyp_key])
            if ref == "" and hyp == "":
                continue
            refs.append(ref)
            hyps.append(hyp)
            n += 1
            if args.limit and n >= args.limit:
                break
    if not refs:
        raise SystemExit("No valid ref/hyp pairs found.")
    w = wer(refs, hyps)
    print(f"Samples: {len(refs)}")
    print(f"WER: {w*100:.2f}%")

if __name__ == "__main__":
    main()
