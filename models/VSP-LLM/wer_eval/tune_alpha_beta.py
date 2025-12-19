import argparse, json, itertools, os, tempfile, subprocess
import numpy as np
from jiwer import wer
from .normalize import normalize_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nbest","ctc"], required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--lm_bin", required=True)
    ap.add_argument("--alpha_grid", default="0.0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--beta_grid", default="-1.0,-0.5,0.0,0.5,1.0")
    args = ap.parse_args()

    alphas=[float(x) for x in args.alpha_grid.split(",")]
    betas=[float(x) for x in args.beta_grid.split(",")]

    best = (1e9, None, None)
    # lazy import to avoid deps unless needed
    if args.mode=="nbest":
        import kenlm
        lm = kenlm.Model(args.lm_bin)
        def lm_score(txt): return float(lm.score(txt, bos=True, eos=True))
        items=[json.loads(l) for l in open(args.in_jsonl,"r",encoding="utf-8")]
        for a,b in itertools.product(alphas, betas):
            refs=[]; hyps=[]
            for it in items:
                cands=it.get("cands") or []
                best_h=""; best_s=-1e18
                for c in cands:
                    txt=normalize_text(c.get("text",""))
                    am=float(c.get("am_score",0.0))
                    s=am + a*lm_score(txt) + b*(len(txt.split()) if txt else 0)
                    if s>best_s:
                        best_s=s; best_h=txt
                refs.append(normalize_text(it.get("ref","")))
                hyps.append(best_h)
            w=wer(refs, hyps)
            if w<best[0]:
                best=(w,a,b)
        print(f"BEST WER={best[0]*100:.2f}% alpha={best[1]} beta={best[2]}")
    else:
        # CTC: delegate to decode_ctc_kenlm for each grid is heavy; do direct pyctcdecode here
        from pyctcdecode import build_ctcdecoder
        import numpy as np
        def load_labels(vocab_path: str):
            if vocab_path.endswith(".json"):
                m=json.load(open(vocab_path,"r",encoding="utf-8"))
                items=sorted(((int(k),v) for k,v in m.items()), key=lambda x:x[0])
                return [v for _,v in items]
            else:
                return [l.rstrip("\n") for l in open(vocab_path,"r",encoding="utf-8")]
        first=json.loads(open(args.in_jsonl,"r",encoding="utf-8").readline())
        labels=load_labels(first["vocab_path"])
        decoder=build_ctcdecoder(labels, kenlm_model_path=args.lm_bin)
        items=[json.loads(l) for l in open(args.in_jsonl,"r",encoding="utf-8")]
        for a,b in itertools.product(alphas, betas):
            refs=[]; hyps=[]
            for it in items:
                lp=np.load(it["logprobs_path"])
                hyp=decoder.decode(lp, alpha=a, beta=b)
                refs.append(normalize_text(it.get("ref","")))
                hyps.append(normalize_text(hyp))
            w=wer(refs, hyps)
            if w<best[0]:
                best=(w,a,b)
        print(f"BEST WER={best[0]*100:.2f}% alpha={best[1]} beta={best[2]}")

if __name__ == "__main__":
    main()
