from __future__ import annotations


def load_policy_checkpoint(policy, path: str, device: str = "cpu") -> dict:
    import torch

    target = policy
    get_model = getattr(policy, "_get_model", None)
    if callable(get_model):
        try:
            target = get_model()
        except Exception:
            target = policy

    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    def _strip_state_dict(s: dict, prefixes: tuple[str, ...]) -> dict:
        if not prefixes:
            return s

        def _strip_prefix(key: str) -> str:
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
                        break
            return key

        return {_strip_prefix(k): v for k, v in s.items()}

    try:
        expected_keys = set(target.state_dict().keys())
    except Exception:
        expected_keys = None

    prefix_candidates = [
        (),
        ("policy.",),
        ("module.",),
        ("policy.", "module."),
        ("model.",),
        ("policy.", "model."),
        ("module.", "model."),
        ("policy.", "module.", "model."),
    ]

    best_state = None
    best_prefixes: tuple[str, ...] = ()
    best_score = None
    if expected_keys is not None:
        for prefixes in prefix_candidates:
            cand = _strip_state_dict(state, prefixes)
            matched = sum(1 for k in cand.keys() if k in expected_keys)
            unexpected = len(cand) - matched
            score = (matched, -unexpected)
            if best_score is None or score > best_score:
                best_score = score
                best_state = cand
                best_prefixes = prefixes
    else:
        best_state = _strip_state_dict(state, ("policy.", "module.", "model."))
        best_prefixes = ("policy.", "module.", "model.")

    res = target.load_state_dict(best_state, strict=False)
    return {
        "path": path,
        "missing_keys": list(getattr(res, "missing_keys", [])),
        "unexpected_keys": list(getattr(res, "unexpected_keys", [])),
        "applied_prefixes": list(best_prefixes),
    }

