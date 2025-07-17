import torch
from collections import OrderedDict

ckpt_path = "./exps/pretrain.model"
out_path = "./exps/sr.model"
checkpoint = torch.load(ckpt_path, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    full_state = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    full_state = checkpoint["state_dict"]
else:
    full_state = checkpoint

encoder_state = OrderedDict()
for key, val in full_state.items():
    if key.startswith("speaker_encoder."):
        new_key = key.replace("speaker_encoder.", "")
        encoder_state[new_key] = val

torch.save(encoder_state, out_path)
print(f"✅ Saved speaker_encoder-only model to {out_path}")
