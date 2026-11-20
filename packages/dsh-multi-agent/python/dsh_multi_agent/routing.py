"""Deterministic model selection from host-supplied task facts, not model output."""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class RoutingPolicy:
    enabled: bool = False
    fastModel: str = ""
    strongModel: str = ""
    fastMaxInputBytes: int = 12000
    fastTimeoutSeconds: int = 120
    strongTimeoutSeconds: int = 300

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("modelRouting.enabled must be boolean")
        for name in ("fastModel", "strongModel"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 256 or any(ord(c) < 32 for c in value):
                raise ValueError("modelRouting model must be a single-line model ID")
            if value != value.strip():
                raise ValueError("modelRouting model cannot have surrounding whitespace")
        for name, low, high in (("fastMaxInputBytes", 256, 64000),
                                ("fastTimeoutSeconds", 30, 300), ("strongTimeoutSeconds", 30, 300)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("Invalid modelRouting." + name)

    def select(self, *, provider, default_model, input_bytes, low_risk=False, reason="unknown_task"):
        if type(low_risk) is not bool:
            raise ValueError("low_risk must be a host boolean")
        if type(input_bytes) is not int or input_bytes < 0:
            raise ValueError("input_bytes must be a non-negative integer")
        identity = {"version": 1, **asdict(self), "provider": provider, "default_model": default_model}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        tier, model, why, timeout = "default", default_model, "routing_disabled", self.strongTimeoutSeconds
        if self.enabled:
            tier, model, why = "strong", self.strongModel or default_model, reason
            if low_risk:
                if input_bytes > self.fastMaxInputBytes:
                    why = "input_exceeds_fast_budget"
                elif not self.fastModel:
                    why = "fast_model_not_configured"
                else:
                    tier, model, timeout = "fast", self.fastModel, self.fastTimeoutSeconds
        return dict(tier=tier, model=model, provider=provider, reason=why,
                    timeout_seconds=timeout, input_bytes=input_bytes, policy_fingerprint=fingerprint)
