import os
import re
import torch
from collections import defaultdict


CKPT_PATH = "Experiments/MotionStreamer_t2m_272_msa_rag_t5_trans662048_latent_retr_every2layer_top3_ddpm_cfg/net_Iter100000.pth"


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "net", "module", "trans", "generator"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[Info] Found state_dict under key: {key}")
                return ckpt[key]

        # 如果本身就是 state_dict
        if all(isinstance(k, str) for k in ckpt.keys()):
            print("[Info] Checkpoint itself seems to be a state_dict.")
            return ckpt

    raise ValueError("Cannot find a valid state_dict in checkpoint.")


def clean_key(k):
    if k.startswith("module."):
        k = k[len("module."):]
    return k


def is_self_attn_name(name):
    patterns = [
        "self_attn", "self_attention", "selfattn",
        "attn", "sa", "causal_attn", "causal_attention"
    ]
    negative = ["cross", "retr", "rag", "encoder_attn", "enc_attn"]
    lname = name.lower()
    return any(p in lname for p in patterns) and not any(n in lname for n in negative)


def is_cross_attn_name(name):
    patterns = [
        "cross_attn", "cross_attention", "crossattn",
        "encoder_attn", "enc_attn",
        "retr_attn", "rag_attn", "retrieval_attn",
        "ca"
    ]
    lname = name.lower()
    return any(p in lname for p in patterns)


def extract_layer_id(key):
    """
    尝试从参数名中提取 layer/block id.
    支持:
    layers.0.xxx
    blocks.3.xxx
    transformer.layers.5.xxx
    decoder.layers.2.xxx
    """
    patterns = [
        r"(?:layers|blocks|h|decoder\.layers|transformer\.layers)\.(\d+)\.",
        r"layer_(\d+)",
        r"block_(\d+)",
    ]

    for p in patterns:
        m = re.search(p, key)
        if m:
            return int(m.group(1))

    return None


def inspect_state_dict_order(state_dict):
    layer_items = defaultdict(list)

    for raw_k in state_dict.keys():
        k = clean_key(raw_k)
        layer_id = extract_layer_id(k)
        if layer_id is None:
            continue

        parts = k.split(".")
        attn_type = None

        if is_cross_attn_name(k):
            attn_type = "CROSS"
        elif is_self_attn_name(k):
            attn_type = "SELF"

        if attn_type is not None:
            layer_items[layer_id].append((k, attn_type))

    if not layer_items:
        print("[Warning] No layer attention parameters found from state_dict names.")
        return

    print("\n==============================")
    print("Attention modules found by layer")
    print("==============================")

    for layer_id in sorted(layer_items.keys()):
        items = layer_items[layer_id]

        # 去重：只保留模块路径，不看具体 weight/bias
        module_paths = []
        seen = set()

        for k, attn_type in items:
            # 去掉最后的参数名，如 weight/bias
            path = ".".join(k.split(".")[:-1])
            if path not in seen:
                seen.add(path)
                module_paths.append((path, attn_type))

        print(f"\nLayer {layer_id}:")
        for path, attn_type in module_paths:
            print(f"  [{attn_type}] {path}")

        ordered_types = []
        for _, t in module_paths:
            if not ordered_types or ordered_types[-1] != t:
                ordered_types.append(t)

        if "SELF" in ordered_types and "CROSS" in ordered_types:
            first_self = ordered_types.index("SELF")
            first_cross = ordered_types.index("CROSS")

            if first_cross < first_self:
                print("  ==> Likely order: CROSS-ATTENTION before SELF-ATTENTION")
            else:
                print("  ==> Likely order: SELF-ATTENTION before CROSS-ATTENTION")
        elif "SELF" in ordered_types:
            print("  ==> Only SELF-ATTENTION found in this layer")
        elif "CROSS" in ordered_types:
            print("  ==> Only CROSS-ATTENTION found in this layer")


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print(f"[Info] Loading checkpoint: {CKPT_PATH}")
    state_dict = load_state_dict(CKPT_PATH)

    print(f"[Info] Number of tensors: {len(state_dict)}")

    print("\n==============================")
    print("Matched attention-related keys")
    print("==============================")

    for k in state_dict.keys():
        kk = clean_key(k)
        lname = kk.lower()
        if any(s in lname for s in ["attn", "attention", "cross", "retr", "rag"]):
            print(kk)

    inspect_state_dict_order(state_dict)


if __name__ == "__main__":
    main()