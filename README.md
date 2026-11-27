# pi-openai-model-discovery

A Pi package that discovers models from custom OpenAI-compatible providers without a static `models` array, and enriches them with pricing and capability metadata.

## Install

From this directory:

```bash
pi install .
```

Or load the extension for one run:

```bash
pi -e ./extensions/openai-model-discovery.ts
```

## Configuration

Add OpenAI-compatible providers to `~/.pi/agent/models.json` without listing `models`:

```json
{
  "providers": {
    "cliproxyapi": {
      "baseUrl": "http://127.0.0.1:8317/v1",
      "api": "openai-completions",
      "apiKey": "sk-..."
    }
  }
}
```

The extension also honors `PI_CODING_AGENT_DIR` (including leading `~` expansion).

- Providers with a non-empty static `models` array are left unchanged.
- If `api` is omitted, it defaults to `openai-completions`. Providers with explicit non-OpenAI `api` values (such as `anthropic-messages` or `google-generative-ai`) are skipped unless explicitly targeted by `providerId`.
- `apiKey` and custom `headers` support Pi's value syntax: literals, `$ENV_VAR` / `${ENV_VAR}` interpolation (`$$` and `$!` escapes), and `!command` shell outputs.
- `Authorization: Bearer <key>` is sent by default whenever an API key is resolved, unless `"authHeader": false` or an explicit `Authorization` header is configured in `headers`.
- Loopback servers (`127.0.0.1`, `localhost`, `::1`) without an `apiKey` or stored credential still discover models without sending an `Authorization` header and automatically supply a harmless local key so Pi marks the provider available. Providers authenticating purely via custom `headers` with `"authHeader": false` are similarly marked configured without sending a Bearer token.
- Provider-level `compat` settings from `models.json` are merged onto discovered models. Set `PI_OFFLINE=1` to skip startup network calls and restore cached models immediately. On `/reload`, previously registered dynamic providers are cleanly unregistered before re-registering.

## How discovery & enrichment work

1. **Startup & refresh discovery**: On Pi startup (including `pi --list-models`, `pi -p`, `--model`, and interactive sessions) and during Pi's model-catalog refresh (`/model`, `/reload`), the extension queries `GET {baseUrl}/models` concurrently across all eligible providers with a bounded timeout (default 15s).
2. **Gateway-native metadata**: Understands 9router's `context_length`, `max_completion_tokens`, and nested `capabilities` fields. When `/models` returns only bare IDs, it makes an optional `?client_version=pi` request for gateways such as CLIProxyAPI, which exposes richer `models[].slug`, `context_window`, `max_tokens`, `input_modalities`, and `supported_reasoning_levels` metadata.
3. **External pricing & capability enrichment**: Fetches pricing and capability metadata from [Wei-Shaw/model-price-repo](https://github.com/Wei-Shaw/model-price-repo) (configurable or disableable via `pricingUrl`) using **no provider auth headers** and caches it in memory and on disk (`openai-model-discovery-pricing-cache.json`, 6-hour TTL). Model IDs are matched with exact lookup, lowercase lookup, stripped `provider/` prefixes (e.g. `cc/gpt-5.5` -> `gpt-5.5`), stripped `:variant` suffixes, and fallback date suffix stripping.
4. **Priority order**: `overrides` -> endpoint-native metadata -> external pricing/capability catalog -> `defaults` -> built-in defaults.
   - Maps `contextWindow`, `maxTokens`, `input` modalities (`["text", "image"]` for vision/image models), per-million-token `cost` including long-context `cost.tiers` (both `tiered_pricing` and `*_above_272k_tokens` / multipliers), and `reasoning` + `thinkingLevelMap` (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
5. **Non-chat model filtering**: Filters out common utility endpoints (`embedding`, `tts`, `whisper`, `transcribe`, `moderation`, `realtime`, `rerank`, `auto-review`, `dall-e`, `imagen`, `flux`, `sora`, `veo`, `gpt-image`) while keeping multimodal chat models such as `gemini-2.5-flash-image`.
6. **Persistence & fallbacks**: Successful non-empty catalogs are persisted in Pi's `models-store.json` (preserving `thinkingLevelMap`, `cost.tiers`, and `compat`). If discovery fails or returns no models, the last successful cached catalog is reused. If pricing metadata fails to load, stale disk cache or configured defaults are used without blocking model discovery.

## Optional extension config

Create `~/.pi/agent/pi-openai-model-discovery.json` (JSONC supported) to customize behavior:

```json
{
  "pricingUrl": "https://raw.githubusercontent.com/Wei-Shaw/model-price-repo/main/model_prices_and_context_window.json",
  "pricingCacheTtlMs": 21600000,
  "requestTimeoutMs": 15000,
  "include": [".*"],
  "exclude": [
    "(?:^|[-_/])(?:embedding|embeddings|tts|whisper|transcribe|transcription|moderation|realtime|rerank|auto-review)(?:$|[-_/])",
    "(?:^|[-_/])(?:dall-e|imagen|flux|sora|veo|gpt-image)(?:$|[-_/])"
  ],
  "defaults": {
    "contextWindow": 128000,
    "maxTokens": 16384,
    "input": ["text"],
    "reasoning": true,
    "thinkingLevelMap": { "xhigh": "xhigh", "max": "max" }
  },
  "overrides": {
    "my-custom-model": {
      "contextWindow": 200000,
      "reasoning": true
    }
  }
}
```

- Set `"pricingUrl": null` (or `""`) to disable external pricing lookups.
- Set `"exclude": []` to disable default non-chat model exclusions.
- Set `"providerId": "cliproxyapi"` to scope discovery to a single provider from `models.json`.

## Troubleshooting

- **Stale `auth.json` OAuth entry shadowing `models.json` `apiKey`**: In Pi, a stored `type: "oauth"` credential in `~/.pi/agent/auth.json` takes precedence over `models.json`'s `apiKey` and causes Pi to treat an API-key-only provider as unconfigured. The extension logs a warning on `stderr` when it detects this; run `/logout <provider>` or remove that provider's entry from `auth.json`.
- **Scoped `/model` list**: If `enabledModels` is set in `~/.pi/agent/settings.json`, Pi's `/model` selector opens in scoped view; press `Tab` to view all discovered models.

## Development

```bash
npm test
npm run check
```
