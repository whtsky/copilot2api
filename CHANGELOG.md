# Changelog

## [Unreleased]

### Features

- Add local amp search support (`webSearch2`) using the Copilot Responses API with `web_search` tool (`gpt-5-mini` by default), and page extraction (`extractWebPageContent`) via Jina Reader, so amp CLI web search works without a paid ampcode.com account
- Auto-upgrade models to the best available variant (e.g. `claude-opus-4.7` → `claude-opus-4.7-1m-internal`) based on upstream model list, enabling features like `effort: high` that require extended variants

## [0.3.1] - 2026-04-26

### Bug Fixes

- Fix Anthropic thinking signatures being emitted as a separate block instead of attached to the currently open thinking block
- Fix Docker image crash (`exec /copilot2api: no such file or directory`) caused by dynamically-linked binary in `scratch` image — add `CGO_ENABLED=0` to CI cross-compilation
- Fix Docker multi-arch build: arm64 image was shipping the amd64 binary due to `ARG TARGETARCH=amd64` default overriding buildx's automatic platform arg
- Fix CI triggering redundant runs on tag pushes — `on: push` now scoped to `main` branch only

### CI

- Add Docker smoke test — `docker run --version` gate before pushing to prevent broken images from reaching the registry

### Docs

- Refresh README quick start and examples

## [0.3.0] - 2026-04-03

### Features

- Add Gemini-compatible `/v1beta/models` endpoints for local `gemini-cli` usage, including `generateContent`, `streamGenerateContent`, and `countTokens`
- Expose the full upstream model list on the Gemini `/v1beta/models` surface instead of limiting the listing to a small allowlist
- Add smart fallback routing between `/v1/chat/completions` and `/v1/responses`, so requests can still work when a model only supports one of the two OpenAI-compatible endpoints
- Improve OpenAI request conversion compatibility across the two endpoints, including better handling for system instructions, structured output, tool choice, reasoning state, and `previous_response_id`
- Improve Claude Code native `/v1/messages` compatibility by removing unsupported passthrough fields before forwarding requests upstream
- Add AmpCode support: chat completions via `/amp/v1/*` and `/api/provider/*` route through Copilot API; management routes (`/api/*`) and login redirects reverse-proxy to `ampcode.com`

## [0.2.0]

### Performance

- Batch SSE flushes in Anthropic streaming — flush once per upstream event instead of per translated event (~3-5x fewer syscalls)
- Flush at SSE event boundaries in native `/v1/messages` passthrough instead of every line (~3x fewer syscalls)
- Defer model alias body re-encode to only the native passthrough path — Responses and Chat Completions paths skip the JSON round-trip entirely
- Remove unnecessary `string()` copy in `writeSSEEvent`

### Architecture

- Consolidate models cache — single upstream `/models` fetch populates both raw JSON (for proxying) and parsed model info (for capability detection), eliminating duplicate HTTP calls
- Remove dead `internal/cache` package after consolidation
- Centralize request body size limit as `upstream.MaxRequestBody` constant (was magic number `10<<20` in 3 files)
- Consistent SSE header setup via `sse.BeginSSE()` across all streaming paths

### Logging

- nginx-style single access log per request at completion with method, endpoint, model, route, duration
- Downgrade client disconnect / context cancellation errors from ERROR to WARN via `upstream.LogRequestError`
- Add `duration_ms` to token refresh logs
- Promote key request lifecycle logs to Info level (was all Debug — invisible in default mode)
- Remove noisy per-chunk/per-event debug logs from streaming hot path
- Add `route` field to Anthropic access log (`native`, `responses`, `chat_completions`)
- Add `endpoint` field to Anthropic access log for consistency with proxy handler
- Add models cache miss debug log

### Bug Fixes

- Fix split choices in OpenAI Chat Completions responses — merge text and tool_calls from separate choices into a single Anthropic message
- Fix `AnthropicContentBlockDelta` / `AnthropicMessageDelta` type confusion in streaming events
- Remove hardcoded "Thinking..." placeholder text in thinking blocks
- Request usage in streaming chunks (`stream_options.include_usage`) so `message_delta` gets real output token counts

### Features

- 1M context window support — automatically appends `-1m` suffix when `anthropic-beta: context-1m-...` header is detected
- Document 1M context window usage in README

## [0.1.0]

- Initial commit
