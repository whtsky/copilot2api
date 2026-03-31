# Proxy Smart Routing: ChatCompletions ↔ Responses Conversion

## Problem

The proxy handler (`proxy/handler.go`) blindly passes requests through to the upstream Copilot API at the same endpoint the client requested. If a model only supports `/responses` but the client hits `/v1/chat/completions` (or vice versa), the upstream returns an error.

The Anthropic handler already has smart routing based on model capabilities. The proxy handler does not.

## Solution

Add capability-based routing to the proxy handler: when the requested endpoint isn't supported by the model, convert the request to a supported format, route to the supported endpoint, and convert the response back.

## Scope

- Smart routing for the OpenAI-compatible proxy endpoints (`/v1/chat/completions`, `/v1/responses`)
- New ChatCompletions ↔ Responses pairwise converters (request, response, and streaming)
- Extract shared OpenAI/Responses types to `internal/types/`
- Add `PickEndpoint` helper to `internal/models/`

### Out of scope

- Changes to the Anthropic handler routing logic (it works)
- Moving Anthropic conversion code
- Gemini or other future provider support
- Router abstraction layer

## Design

### 1. `internal/types/` — Shared request/response types

Move from `anthropic/types.go` to a new `internal/types/` package:

- `OpenAIChatCompletionsRequest`, `OpenAIChatCompletionsResponse`, `OpenAIChatCompletionChunk` and their sub-types (messages, tools, choices, usage, etc.)
- `ResponsesRequest`, `ResponsesResult`, `ResponseStreamEvent` and their sub-types (input items, output items, tool definitions, etc.)

Keep in `anthropic/types.go`:

- `AnthropicMessagesRequest`, `AnthropicMessagesResponse`, `AnthropicStreamEvent` and all Anthropic-specific types

Both `proxy/` and `anthropic/` import from `internal/types/`.

### 2. `internal/models/` — `PickEndpoint` helper

Add to `models.go`:

```go
// PickEndpoint returns the first endpoint from preferred that the model
// supports, or "" if none match.
func PickEndpoint(info *Info, preferred []string) string
```

This uses the existing `SupportsEndpoint` function internally. Returns empty string when info is nil (unknown model), letting callers fall back to default behavior.

### 3. `proxy/convert.go` — ChatCompletions ↔ Responses conversion

New file with two conversion directions:

**Chat Completions → Responses:**

- `ConvertChatToResponsesRequest(req types.OpenAIChatCompletionsRequest) types.ResponsesRequest` — maps messages to input items, tools to tool definitions, model/temperature/etc.
- `ConvertResponsesResultToChatResponse(result types.ResponsesResult) types.OpenAIChatCompletionsResponse` — maps output items back to choices, usage, etc.
- Stream back-conversion: translate Responses stream events into Chat Completions chunks

**Responses → Chat Completions:**

- `ConvertResponsesToChatRequest(req types.ResponsesRequest) types.OpenAIChatCompletionsRequest` — maps input items to messages, tool definitions to tools
- `ConvertChatResponseToResponsesResult(resp types.OpenAIChatCompletionsResponse) types.ResponsesResult` — maps choices back to output items
- Stream back-conversion: translate Chat Completions chunks into Responses stream events

### 4. `proxy/handler.go` — Smart routing logic

Update `handlePassthrough` to:

1. Parse request body to extract `model` field
2. Look up model capabilities via `modelsCache.GetInfo()`
3. Determine target endpoint:
   - `/chat/completions` requested → prefer `["/chat/completions", "/responses"]`
   - `/responses` requested → prefer `["/responses", "/chat/completions"]`
   - Call `models.PickEndpoint(modelInfo, preferred)`
4. If target matches requested endpoint OR target is empty (unknown model) → passthrough as today
5. If target differs → parse full request, convert, route to target, convert response back

Both streaming and non-streaming paths need conversion support.

### 5. `anthropic/` — Update imports

Change type references in all `anthropic/` files from local types to `internal/types/`. No logic changes. The Anthropic-specific types stay in `anthropic/types.go`.

## Conversion Mapping Notes

### Chat Completions → Responses

| Chat Completions | Responses |
|---|---|
| `messages[]` | `input[]` (map role/content to input items) |
| `messages[].role = "system"` | `instructions` or system input item |
| `tools[]` | `tools[]` (function → function tool) |
| `tool_choice` | `tool_choice` |
| `model` | `model` |
| `temperature` | `temperature` |
| `top_p` | `top_p` |
| `max_tokens` / `max_completion_tokens` | `max_output_tokens` |
| `stream` | `stream` |

### Responses → Chat Completions

The reverse mapping. Input items become messages, tool definitions become tools, output items become choices.

## Edge Cases

- **Unknown model (capabilities fetch failed):** Passthrough to requested endpoint as today. Best effort — matches current behavior.
- **Model supports neither endpoint:** Passthrough to requested endpoint. The upstream will return an error, which we forward as-is. No point inventing a better error when the model genuinely doesn't support either format.
- **Model supports both endpoints:** Passthrough to requested endpoint (no conversion needed).

## Rollout

The conversion is transparent to clients. If a model doesn't support the requested endpoint, the proxy converts silently. Degraded/unknown cases fall back to passthrough (current behavior).
