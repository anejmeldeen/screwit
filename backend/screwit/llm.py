"""Claude client factory shared by the agent (P4), perception (P3) and codegen (X7). See AGENTS.md decision 4.

SCREWIT_LLM    openrouter | bedrock | anthropic | fake   (default fake, so nothing calls out by accident)
SCREWIT_MODEL  model id for the chosen provider; required for every real provider, no default is guessed
               (OpenRouter ids look like "anthropic/claude-sonnet-4.5")
Credentials come from the environment only: OPENROUTER_API_KEY (openrouter), AWS credentials
(bedrock), ANTHROPIC_API_KEY (anthropic).

OpenRouter exposes Anthropic's Messages API (same request and response format, streaming and tool
use) at https://openrouter.ai/api with the key as a Bearer token, so every caller keeps using the
anthropic SDK unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDERS = {"openrouter", "bedrock", "anthropic", "fake"}
REAL_PROVIDERS = PROVIDERS - {"fake"}
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class LLMNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str | None
    aws_region: str | None


def config() -> LLMConfig:
    provider = os.environ.get("SCREWIT_LLM", "fake")
    if provider not in PROVIDERS:
        raise LLMNotConfigured(
            f"SCREWIT_LLM must be one of {', '.join(sorted(PROVIDERS))}, got {provider!r}")
    model = os.environ.get("SCREWIT_MODEL")
    if provider != "fake" and not model:
        raise LLMNotConfigured(f"SCREWIT_MODEL must be set when SCREWIT_LLM={provider}")
    return LLMConfig(provider=provider, model=model,
                     aws_region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))


def client():
    """An anthropic SDK client (Messages API) for the configured provider. Callers handle `fake` themselves."""
    cfg = config()
    if cfg.provider == "fake":
        raise LLMNotConfigured("SCREWIT_LLM=fake: use the caller's fake implementation")
    import anthropic

    if cfg.provider == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMNotConfigured("OPENROUTER_API_KEY is not set")
        # api_key=None so the SDK never falls back to an ANTHROPIC_API_KEY from the environment.
        return anthropic.Anthropic(base_url=OPENROUTER_BASE_URL, auth_token=key, api_key=None,
                                   default_headers={"X-Title": "Screw It"})
    if cfg.provider == "bedrock":
        return anthropic.AnthropicBedrock(aws_region=cfg.aws_region)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMNotConfigured("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic()
