"""Make unused FlashAttention optional; leave native flex/torch paths intact."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
path = root / 'wan_va/modules/model.py'
source = path.read_text(encoding='utf-8')
original = hashlib.sha256(path.read_bytes()).hexdigest()
old = 'try:\n    from flash_attn_interface import flash_attn_func\nexcept:\n    from flash_attn import flash_attn_func\n\n'
branch = "        elif attn_mode == 'flashattn':\n            self.attn_op = flash_attn_func"
replacement = "        elif attn_mode == 'flashattn':\n            try:\n                from flash_attn_interface import flash_attn_func\n            except ImportError:\n                from flash_attn import flash_attn_func\n            self.attn_op = flash_attn_func"
assert source.count(old) == 1 and source.count(branch) == 1
source = source.replace(old, '').replace(branch, replacement)
path.write_text(source, encoding='utf-8')
(root.parent / 'optional_flash_import.json').write_text(json.dumps(dict(
    original_sha256=original, patched_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    change='lazy FlashAttention import only; native flex training and torch inference unchanged'
), indent=2), encoding='utf-8')

if '--mixed-batch' in sys.argv:
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = '        device,\n    ):\n        torch._inductor'
    call = '                               device=hidden_states.device\n'
    sequence = '        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)'
    assert source.count(signature) == source.count(call) == source.count(sequence) == 1
    source = source.replace(signature, '        device,\n        video_lengths=None,\n    ):\n        torch._inductor')
    source = source.replace(call, "                               device=hidden_states.device,\n                               video_lengths=input_dict.get('video_lengths'),\n")
    source = source.replace(sequence, '''        if video_lengths is not None:
            lengths = video_lengths.detach().cpu()
            valid_video = (torch.arange(L_F)[None, :] < lengths[:, None])
            valid_action = (torch.arange(A_F)[None, :] < lengths[:, None])
            valid_video = valid_video[:, :, None, None].expand(-1, -1, L_H // patch_size[1], L_W // patch_size[2]).flatten()
            valid_action = valid_action[:, :, None, None].expand(-1, -1, A_H, A_W).flatten()
            latent_seq_id = latent_seq_id.masked_fill(~valid_video, -1)
            action_seq_id = action_seq_id.masked_fill(~valid_action, -1)
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)''')
    path.write_text(source, encoding='utf-8')
    (root.parent / 'mixed_batch_mask.json').write_text(json.dumps(dict(
        original_sha256=original, patched_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        change='Mark padded Query/action tokens invalid in native self/cross attention; retain native causal predicate'
    ), indent=2), encoding='utf-8')
