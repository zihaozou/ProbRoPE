import re

def _normalize_checkpoint_keys(state_dict):
    """Normalize parameter prefixes and retain compatible camera-encoder tensors."""
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("vace"):
            continue
        if name.split(".")[0] in [
            "pose_patch_embedding",
            "face_adapter",
            "face_encoder",
            "motion_encoder",
        ]:
            continue
        name_ = name
        if name_.startswith("model."):
            name_ = name_[len("model.") :]


        if ".cam_encoder.weight" in name_ or ".cam_encoder.bias" in name_:

            if ".cam_encoder.conv." not in name_:
                continue

        state_dict_[name_] = state_dict[name]
    return state_dict_


_SELF_ATTN_PROJ_RE = re.compile(r"^(blocks\.\d+\.self_attn)\.(q|k|v|o)\.(weight|bias)$")
_SELF_ATTN_NORM_RE = re.compile(r"^(blocks\.\d+\.self_attn)\.(norm_q|norm_k)\.(weight)$")
_FFN_RE = re.compile(r"^(blocks\.\d+\.ffn)\.(0|2)\.(weight|bias)$")


_DFUSE_FUSION_PROJ_RE = re.compile(
    r"^(blocks\.\d+\.fusion_attn)\.(q|k|v|o)_list\.(\d+)\.(weight|bias)$"
)
_DFUSE_FUSION_NORM_RE = re.compile(
    r"^(blocks\.\d+\.fusion_attn)\.(norm_q|norm_k)_list\.(\d+)\.(weight)$"
)
_DFUSE_FFN_RE = re.compile(
    r"^(blocks\.\d+\.ffn)\.ffn_list\.(\d+)\.(0|2)\.(weight|bias)$"
)


def _get_num_modalities_from_config(model_name="dfuse"):
    """Read num_modalities from the first matching MODEL_CONFIGS entry.

    The training script patches MODEL_CONFIGS before model loading, so this
    picks up the user-specified value.  Falls back to 1 (model class default).
    """
    from dfuse.configs import MODEL_CONFIGS

    for cfg in MODEL_CONFIGS:
        if cfg.get("model_name") == model_name:
            return cfg["extra_kwargs"].get("num_modalities", 1)
    return 1


def DFuseStateDictConverter(state_dict):
    """Expand D-FUSE expert weights to the configured modality count."""
    base_sd = _normalize_checkpoint_keys(state_dict)


    ckpt_max_idx = -1
    for key in base_sd:
        for pat, idx_group in (
            (_DFUSE_FUSION_PROJ_RE, 3),
            (_DFUSE_FUSION_NORM_RE, 3),
            (_DFUSE_FFN_RE, 2),
        ):
            m = pat.match(key)
            if m:
                ckpt_max_idx = max(ckpt_max_idx, int(m.group(idx_group)))
                break

    if ckpt_max_idx < 0:

        return base_sd

    ckpt_num_modalities = ckpt_max_idx + 1
    target_num_modalities = _get_num_modalities_from_config(
        "dfuse"
    )

    if target_num_modalities <= ckpt_num_modalities:
        return base_sd

    print(
        f"[D-FUSEConverter] Replicating fusion_attn/ffn slot 0 → slots "
        f"{ckpt_num_modalities}..{target_num_modalities - 1} "
        f"(checkpoint has {ckpt_num_modalities} modality, "
        f"model expects {target_num_modalities})."
    )

    new_sd = dict(base_sd)
    for key, tensor in base_sd.items():
        m = _DFUSE_FUSION_PROJ_RE.match(key)
        if m:
            prefix, proj, idx, suffix = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if idx == 0:
                for i in range(ckpt_num_modalities, target_num_modalities):
                    new_sd[f"{prefix}.{proj}_list.{i}.{suffix}"] = tensor.clone()
            continue

        m = _DFUSE_FUSION_NORM_RE.match(key)
        if m:
            prefix, norm, idx, suffix = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if idx == 0:
                for i in range(ckpt_num_modalities, target_num_modalities):
                    new_sd[f"{prefix}.{norm}_list.{i}.{suffix}"] = tensor.clone()
            continue

        m = _DFUSE_FFN_RE.match(key)
        if m:
            prefix, idx, layer_idx, suffix = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            if idx == 0:
                for i in range(ckpt_num_modalities, target_num_modalities):
                    new_sd[f"{prefix}.ffn_list.{i}.{layer_idx}.{suffix}"] = tensor.clone()
            continue

    return new_sd
