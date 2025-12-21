import argparse
import json
import torch

from datamodule.data_module import DataModule
from datamodule.transforms import TextTransform

# Import ModelModule from lightning (uses same wiring as eval.py)
from lightning import ModelModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint file")
    p.add_argument("--modality", required=True, choices=["audio", "video"]) 
    p.add_argument("--root-dir", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decode-snr-target", type=float, default=999999)
    return p.parse_args()


def _load_state_dict_to_model(model, ckpt):
    state = ckpt
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]

    try:
        model.model.load_state_dict(state)
    except Exception:
        try:
            model.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint into model: {e}")


def sample_to_text(tt, target_tensor):
    # target_tensor expected to be 1-D tensor of token ids
    if isinstance(target_tensor, torch.Tensor):
        target_tensor = target_tensor.cpu()
    # handle extra trailing dimensions (e.g. [L,1])
    try:
        target_tensor = target_tensor.squeeze()
    except Exception:
        pass
    text = tt.post_process(target_tensor)
    # remove special tokens used by the model
    if isinstance(text, str):
        text = text.replace("<eos>", "").replace("<blank>", "").strip()
    return text


def main():
    args = parse_args()

    # Prepare minimal args object expected by DataModule / ModelModule
    # ModelModule and DataModule access attributes on args; we'll set what is needed.
    # Keep other attributes at reasonable defaults.
    required_attrs = dict(
        modality=args.modality,
        root_dir=args.root_dir,
        test_file=args.test_file,
        decode_snr_target=args.decode_snr_target,
        lr=1e-3,
        weight_decay=0.03,
        warmup_epochs=5,
        max_epochs=1,
        max_frames=1600,
        ctc_weight=0.1,
    )
    # Build a simple namespace-like object
    class A: pass

    a = A()
    for k, v in required_attrs.items():
        setattr(a, k, v)

    # Instantiate model module and load weights
    model_module = ModelModule(a)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    _load_state_dict_to_model(model_module, ckpt)

    device = torch.device(args.device)
    model_module.to(device)
    model_module.model.to(device)
    model_module.eval()

    # Data
    dm = DataModule(a)
    dl = dm.test_dataloader()

    tt = TextTransform()

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in dl:
            # `item` from test_dataloader is a single sample dict with keys 'input' and 'target'
            if "input" in item:
                inp = item["input"]
                tgt = item["target"]
                utt_id = item.get("id", None)

                if hasattr(inp, "to"):
                    inp = inp.to(device)

                with torch.no_grad():
                    # ModelModule.forward expects a tensor (it unsqueezes internally)
                    hyp = model_module.forward(inp)

                ref = sample_to_text(tt, tgt)
                out = {"id": utt_id, "ref": ref, "hyp": hyp}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            else:
                # Defensive: support small-batch dicts created by collate (inputs/targets)
                if "inputs" in item:
                    inputs = item["inputs"]
                    targets = item["targets"]
                    for i in range(inputs.size(0)):
                        inp = inputs[i]
                        tgt = targets[i]
                        if hasattr(inp, "to"):
                            inp = inp.to(device)
                        with torch.no_grad():
                            hyp = model_module.forward(inp)
                        ref = sample_to_text(tt, tgt)
                        out = {"id": None, "ref": ref, "hyp": hyp}
                        fout.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
