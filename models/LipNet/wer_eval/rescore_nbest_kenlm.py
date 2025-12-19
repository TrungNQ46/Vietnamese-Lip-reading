import argparse, json
import kenlm
from .normalize import normalize_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbest_jsonl", required=True, help="Each line: {utt_id, ref, cands:[{text, am_score}, ...]}")
    ap.add_argument("--lm_bin", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    lm = kenlm.Model(args.lm_bin)

    def lm_score(txt: str) -> float:
        # kenlm.Model.score returns log10
        return float(lm.score(txt, bos=True, eos=True))

    with open(args.nbest_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            it = json.loads(line)
            cands = it.get("cands") or []
            best_hyp = ""
            best_s = -1e18
            for c in cands:
                txt = normalize_text(c.get("text",""))
                am = float(c.get("am_score", 0.0))
                s = am + args.alpha * lm_score(txt) + args.beta * (len(txt.split()) if txt else 0)
                if s > best_s:
                    best_s = s
                    best_hyp = txt
            out = {
                "utt_id": it.get("utt_id"),
                "ref": it.get("ref",""),
                "hyp": best_hyp,
                "alpha": args.alpha,
                "beta": args.beta
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
