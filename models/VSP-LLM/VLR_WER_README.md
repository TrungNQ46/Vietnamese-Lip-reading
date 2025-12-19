# vsp_llm VLR WER integration

This folder adds a Vietnamese WER reduction pipeline (Normalize + KenLM rescoring + WER).

## What you need to produce from the model
- If this model is **NBEST/S2S**: produce a JSONL file where each line is:
  { "utt_id": "...", "ref": "...", "cands":[{"text":"...", "am_score": -12.3}, ...] }

- If this model is **CTC**: produce a JSONL file where each line is:
  { "utt_id": "...", "ref": "...", "logprobs_path": "path/to.npy", "vocab_path": "vocab.txt|vocab.json" }

Then run the scripts under `wer_eval/`.

## Quick commands
### Tune alpha/beta on VALID
python -m wer_eval.tune_alpha_beta --mode nbest --in_jsonl work/valid.jsonl --lm_bin lm/vi_5gram.binary

### Decode/rescore on TEST
python -m wer_eval.rescore_nbest_kenlm --nbest_jsonl work/test_nbest.jsonl --lm_bin lm/vi_5gram.binary --alpha 0.8 --beta 0.5 --out_jsonl work/test_rescored.jsonl
python -m wer_eval.compute_wer_jsonl --jsonl work/test_rescored_or_decoded.jsonl
