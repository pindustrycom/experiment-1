"""USD per million tokens. Editable in Settings (llm.pricing); these are the defaults."""
DEFAULT_PRICING = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write": 5.0},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20, "cache_write": 2.5},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-fable-5-1": {"input": 10.0, "output": 50.0, "cache_read": 0.25, "cache_write": 12.5},
}
FALLBACK = {"input": 10.0, "output": 50.0, "cache_read": 1.0, "cache_write": 12.5}  # conservative for unknown models


def cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0, cache_write: int = 0, pricing: dict | None = None) -> float:
    p = (pricing or DEFAULT_PRICING).get(model, FALLBACK)
    return (input_tokens * p["input"] + output_tokens * p["output"] + cache_read * p["cache_read"] + cache_write * p["cache_write"]) / 1e6
