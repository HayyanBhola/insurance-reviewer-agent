"""
Shared measurement helpers: latency percentiles, cost in dollars, faithfulness.

Prices are per 1 million tokens, from the providers' model pages (check them when
prices change). Models not listed here show tokens only, with cost "unknown".
"""

import math

PRICES_PER_MILLION = {  # model name prefix: (input $, output $)
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
}


def percentile(values, p):
    """p in 0..100, nearest-rank method. Returns None for an empty list."""
    if not values:
        return None
    v = sorted(values)
    k = max(0, math.ceil(p / 100 * len(v)) - 1)
    return v[k]


def cost_usd(usage):
    """usage: {model_name: {"input_tokens": .., "output_tokens": ..}} -> (dollars or None, unpriced models)."""
    total, unpriced = 0.0, []
    for model, u in usage.items():
        price = next((p for prefix, p in PRICES_PER_MILLION.items() if model.startswith(prefix)), None)
        if price is None:
            unpriced.append(model)
            continue
        total += u.get("input_tokens", 0) / 1e6 * price[0] + u.get("output_tokens", 0) / 1e6 * price[1]
    return (None if unpriced and total == 0 else total), unpriced


def faithfulness(reports):
    """Share of AI citations whose quote was found in the policy text.
    Returns (rate or None, verified, total)."""
    cits = [c for r in reports for c in r.coverage.citations]
    if not cits:
        return None, 0, 0
    ok = sum(c.verified for c in cits)
    return ok / len(cits), ok, len(cits)
