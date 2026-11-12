"""Cache fixed human descriptions with the native LingBot text encoder."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--descriptions", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source / "wan_va"))
    from modules.utils import load_text_encoder, load_tokenizer
    from wan_va_server import VA_Server
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    start = time.perf_counter()
    encoder = load_text_encoder(str(args.model / "text_encoder"), torch.bfloat16, "cuda")
    encoder.eval().requires_grad_(False)
    tokenizer = load_tokenizer(str(args.model / "tokenizer"))
    context = SimpleNamespace(device=torch.device("cuda"), dtype=torch.bfloat16,
                              text_encoder=encoder, tokenizer=tokenizer)
    descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    prompts = {key: descriptions[key][0] for key in selection["human_ids"]}
    embeddings = {}
    with torch.inference_mode():
        for key, prompt in [("__empty__", ""), *prompts.items()]:
            embeddings[key] = VA_Server._get_t5_prompt_embeds(
                context, prompt=prompt, max_sequence_length=512)[0].cpu()
            assert torch.isfinite(embeddings[key]).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(embeddings={k: v for k, v in embeddings.items() if k != "__empty__"},
                    empty=embeddings["__empty__"], prompts=prompts,
                    description_selection="first official human description, frozen per task",
                    max_sequence_length=512), args.output)
    result = dict(prompts=len(prompts), seconds=time.perf_counter() - start,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  input="human task description only", shape=list(embeddings["__empty__"].shape))
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
