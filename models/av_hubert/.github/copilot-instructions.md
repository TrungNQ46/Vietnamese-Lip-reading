# Copilot / AI agent instructions for AV-HuBERT

Purpose: give an AI coding agent the minimal, specific knowledge to be immediately productive in this repository.

Quick architecture (high level)
- Core model code lives in `avhubert/` — the main model is registered as `av_hubert` in [avhubert/hubert.py](avhubert/hubert.py).
- ASR / finetune wrappers are in [avhubert/hubert_asr.py](avhubert/hubert_asr.py); decoding utilities live in [avhubert/sequence_generator.py](avhubert/sequence_generator.py) and `infer_s2s.py`.
- Config-driven workflows use Hydra config files under `conf/` (e.g., `conf/s2s_decode.yaml`).
- Data prep scripts are under `preparation/`; clustering (pretrain labels) is under `clustering/`.
- This project embeds a vendored `fairseq/` copy; model training uses `fairseq-hydra-train` and fairseq task/registry mechanisms.

Important developer workflows (concrete)
- Setup: create a conda env and run `pip install -r requirements.txt`, then `cd fairseq && pip install -e ./` (see [README.md](README.md)).
- Pretrain command (from README):
  - `fairseq-hydra-train --config-dir /path/to/conf/ --config-name conf-name task.data=/path/to/data task.label_dir=/path/to/label model.label_rate=100 hydra.run.dir=/path/to/exp common.user_dir=`pwd``
- Finetune command (seq2seq): use `fairseq-hydra-train` with `model.w2v_path` pointing to the pretrained checkpoint (see README examples).
- Decode: use `python -B infer_s2s.py --config-dir ./conf/ --config-name <conf> dataset.gen_subset=test common_eval.path=/path/to/checkpoint common_eval.results_path=/path/to/exp common.user_dir=`pwd`` and override `override.modalities=['video']|['audio']|['audio','video']`.

Project-specific conventions and patterns
- Hydra + `common.user_dir=`pwd`` is required so Fairseq can discover local `avhubert` modules. Keep `common.user_dir` present in examples.
- Many modules use a DBG switch (checking `sys.argv`) to allow running files directly vs package imports — be careful with relative imports when editing tests or scripts.
- Model registration names to know: `av_hubert` (main model), `av_hubert_ctc` (CTC variant). Use these keys when inspecting training logs or config files.
- Masking and data-flow: features are prepared with scripts in `preparation/` and clustering outputs (`*.km`) in `clustering/` are referenced by configs via `task.label_dir` and `task.data`.

Integration points and external dependencies
- Relies on Fairseq internals: tasks, checkpoint_utils, hydra config parsing. Edits to model/task code must preserve expected function signatures (e.g., `build_model`, `forward`, `upgrade_state_dict_named`).
- Native dependencies: OpenCV (`cv2`) is used for video loading in `avhubert/utils.py`.
- Optional accelerators: NVIDIA apex (for fp16) and PyArrow for large dataset handling; see `fairseq/README.md`.

What to look at when changing model behavior
- Start with [avhubert/hubert.py](avhubert/hubert.py) and [avhubert/hubert_asr.py](avhubert/hubert_asr.py): these define configs (dataclasses), registration, and the build flow.
- Changes that affect tensor shapes usually require updates to `sequence_generator.py` and `decoder.py` (beam/decoder interaction).
- When adding config options, add them to the appropriate dataclass and to the YAML files under `conf/`.

Example quick tasks for an AI agent
- Add a new config flag: update the dataclass (e.g., `AVHubertConfig`), add default in `conf/` and a line in the README command examples.
- Small refactor: keep public API stable — do not change registered model names or task keys unless updating configs and examples.

References (where patterns are exemplified)
- [README.md](README.md) — setup, pretrain/finetune/decode commands
- [avhubert/hubert.py](avhubert/hubert.py) — model registration and config dataclass
- [avhubert/hubert_asr.py](avhubert/hubert_asr.py) — finetune/ASR wrappers and build flow
- [avhubert/sequence_generator.py](avhubert/sequence_generator.py) — decoding/beam search specifics
- `conf/` — Hydra config examples and override patterns used in CLI

If anything above is unclear or you want more detail (e.g., exact lines demonstrating `common.user_dir` usage or example Hydra overrides), tell me which part to expand and I will update this file.
