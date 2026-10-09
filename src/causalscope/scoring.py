import torch


def score_components(
    model,
    x,
    keys,
    calibration,
    config,
    temperature=1.0,
    excluded=None,
    full_candidates=False,
):
    original = model(x)
    probability = (original["root_logits"] / temperature).softmax(-1)
    gains = torch.zeros_like(probability)
    available = torch.ones_like(probability, dtype=torch.bool)
    if excluded is not None:
        available &= ~excluded
    screened = (
        probability.masked_fill(~available, -1)
        .topk(min(config["candidates"], probability.shape[1]), dim=1)
        .indices
    )
    if config["counterfactual"]:
        for index, component in enumerate(model.schema["components"]):
            selected = available[:, index] & (
                (screened == index).any(-1) | full_candidates
            )
            if not selected.any():
                continue
            replacement = calibration.replacement(x, keys, component)
            gain = (original["step"] - model(replacement)["step"]).clamp_min(0)
            gains[:, index] = torch.where(selected, gain, 0)
        ranking = config["alpha"] * probability + (1 - config["alpha"]) * gains / (
            gains.sum(-1, keepdim=True) + 1e-6
        )
    else:
        ranking = probability
    ranking = ranking.masked_fill(~available, -1)
    missing_nodes = (x[:, :, :, 1].mean(-1) > 0.5).float().mean(-1)
    abstain = (probability.max(-1).values < config["probability_threshold"]) | (
        missing_nodes > config["missing_threshold"]
    )
    if config["counterfactual"]:
        abstain |= gains.max(-1).values < config["gain_threshold"]
    return {
        "scores": ranking,
        "probabilities": probability,
        "gains": gains,
        "step": original["step"],
        "fault_probability": original["fault_logit"].sigmoid(),
        "abstain": abstain,
        "missing_fraction": missing_nodes,
    }


def rank_components(model, x, keys, calibration, config, temperature=1.0):
    result = score_components(model, x, keys, calibration, config, temperature)
    order = result["scores"].argsort(dim=1, descending=True, stable=True)
    result["initial_order"] = order
    if config["iterative"] and config["counterfactual"]:
        excluded = torch.zeros_like(result["scores"], dtype=torch.bool)
        excluded.scatter_(1, order[:, :1], True)
        replacement = x.clone()
        for i, component in enumerate(model.schema["components"]):
            selected = order[:, 0] == i
            if selected.any():
                restored = calibration.replacement(x, keys, component)
                replacement[selected] = restored[selected]
        second = score_components(
            model, replacement, keys, calibration, config, temperature, excluded
        )
        next_order = second["scores"].argsort(dim=1, descending=True, stable=True)
        order = torch.cat([order[:, :1], next_order[:, :-1]], dim=1)
    result["order"] = order
    return result
