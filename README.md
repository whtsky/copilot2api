# copilot2api

A lightweight Go proxy that exposes GitHub Copilot as OpenAI-compatible, Anthropic-compatible, Gemini-compatible, and AmpCode-compatible API endpoints.

## Features

- **OpenAI API Compatible**: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses`
- **Embeddings Support**: Native OpenAI-compatible `/v1/embeddings` endpoint
- **Anthropic API Compatible**: `/v1/messages`
- **Gemini API Compatible**: `/v1beta/models`, `/v1beta/models/{model}:generateContent`, `/v1beta/models/{model}:streamGenerateContent`, `/v1beta/models/{model}:countTokens`
- **AmpCode Compatible**: `/amp/v1/*` routes for chat, `/api/provider/*` for provider-specific calls, management proxied to `ampcode.com`
- **Streaming Support**: Full SSE streaming for both OpenAI and Anthropic formats
- **Anthropic Routing**: Uses native `/v1/messages` when the model supports it, otherwise routes via `/responses` or `/chat/completions`
- **Auto Authentication**: GitHub Device Flow OAuth with automatic token refresh
- **Usage Monitoring**: Built-in `/usage` endpoint for quota tracking
- **Models Cache**: 5-minute cache for `/v1/models` and Anthropic model capability lookups

## Quick Start

### Docker

```bash
docker run -it --rm \
  -p 127.0.0.1:7777:7777 \
  -v ~/.config/copilot2api:/root/.config/copilot2api \
  ghcr.io/whtsky/copilot2api:latest
```

The volume mount persists your GitHub credentials across container restarts. The examples publish the port on `127.0.0.1` only so the proxy stays local by default.

<details>
<summary>Docker Compose</summary>

```yaml
services:
  copilot2api:
    image: ghcr.io/whtsky/copilot2api:latest
    ports:
      - "127.0.0.1:7777:7777"
    volumes:
      - ${HOME}/.config/copilot2api:/root/.config/copilot2api
```

Start it with:

```bash
docker compose up
```

</details>

### Download a release binary

```bash
# Example: macOS Apple Silicon
curl -L -o copilot2api \
  https://github.com/whtsky/copilot2api/releases/latest/download/copilot2api-darwin-arm64

# Example: Linux x64
# curl -L -o copilot2api \
#   https://github.com/whtsky/copilot2api/releases/latest/download/copilot2api-linux-amd64

chmod +x copilot2api
./copilot2api
```

Download the asset that matches your platform from [GitHub Releases](https://github.com/whtsky/copilot2api/releases/latest). Published binaries use names like `copilot2api-linux-amd64`, `copilot2api-linux-arm64`, `copilot2api-darwin-amd64`, `copilot2api-darwin-arm64`, `copilot2api-windows-amd64.exe`, and `copilot2api-windows-arm64.exe`.

On first run, both Docker and downloaded binaries prompt GitHub Device Flow authentication:

```
🔐 GitHub Authentication Required
Please visit: https://github.com/login/device
Enter code: XXXX-XXXX

Waiting for authorization...
✅ Authentication successful!
```

Server starts on `http://127.0.0.1:7777` by default.

## Security

⚠️ **This proxy is designed for local development only.**

- Does **not** implement API key validation — any request is accepted
- Do not expose publicly — it becomes an open proxy consuming your Copilot quota
- Credentials are stored in `~/.config/copilot2api/credentials.json`

## Usage with Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:7777",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_MODEL": "claude-opus-4.6",
    "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4.5",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "permissions": {
    "deny": [
      "WebSearch"
    ]
  }
}
```

### Model Auto-Upgrade

copilot2api automatically upgrades models to the best available variant in the upstream model list. For example, if `claude-opus-4.7-1m-internal` is available, a request for `claude-opus-4.7` will automatically use it. This enables features like 1M context windows and `effort: high` that are only supported by extended variants.

The upgrade priority is: `-1m-internal` > `-1m` > base model.

## Usage with Codex

Add to `~/.codex/config.toml`:

```toml
model = "gpt-5.3-codex"
model_provider = "copilot2api"
model_reasoning_effort = "high"
web_search = "disabled"

[model_providers.copilot2api]
name = "copilot2api"
base_url = "http://127.0.0.1:7777/v1"
wire_api = "responses"
api_key = "dummy"
```

## Usage with Gemini CLI

Add to `~/.gemini/.env`:

```env
GOOGLE_GEMINI_BASE_URL=http://127.0.0.1:7777
GEMINI_API_KEY=dummy
GEMINI_MODEL=claude-opus-4.6-1m
```

## Usage with AmpCode

Set the `AMP_URL` environment variable to point at copilot2api:

```bash
AMP_URL=http://127.0.0.1:7777/amp amp
```

Or add to `~/.config/amp/settings.json`:

```json
{
  "amp.url": "http://127.0.0.1:7777/amp"
}
```

Chat completions, tool calls, and image input all route through Copilot API. Login and management routes (threads, telemetry) are proxied to `ampcode.com` — a free amp account is required for authentication.

Web search (`webSearch2`) is handled locally via the Copilot Responses API with `web_search` tool (using `gpt-5-mini` by default). Page extraction (`extractWebPageContent`) uses [Jina Reader](https://jina.ai/reader/) — set `JINA_API_KEY` for higher rate limits (optional). No paid ampcode.com account needed.

<details>
<summary>Usage with curl</summary>

```bash
# OpenAI chat completion
curl http://localhost:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.3-codex","messages":[{"role":"user","content":"Hello!"}]}'

# Anthropic message
curl http://localhost:7777/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: dummy" \
  -d '{"model":"claude-sonnet-4.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'

# List models
curl http://localhost:7777/v1/models

# Check usage/quota
curl http://localhost:7777/usage
```

</details>

<details>
<summary>Usage with SDKs</summary>

### OpenAI Python SDK

```python
import openai

client = openai.OpenAI(
    api_key="dummy",
    base_url="http://127.0.0.1:7777/v1"
)

response = client.chat.completions.create(
    model="gpt-5.3-codex",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="dummy",
    base_url="http://127.0.0.1:7777"
)

message = client.messages.create(
    model="claude-sonnet-4.6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

</details>

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI Chat Completions (streaming & non-streaming) |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/models` | GET | List available models (5min cache) |
| `/v1/embeddings` | POST | Generate embeddings (string or array input) |
| `/v1/messages` | POST | Anthropic Messages API (streaming & non-streaming) |
| `/v1beta/models` | GET | List Gemini-compatible models |
| `/v1beta/models/{model}:generateContent` | POST | Gemini Generate Content |
| `/v1beta/models/{model}:streamGenerateContent` | POST | Gemini Generate Content streaming SSE |
| `/v1beta/models/{model}:countTokens` | POST | Gemini token counting estimate |
| `/amp/v1/chat/completions` | POST | AmpCode chat completions (via Copilot API) |
| `/amp/v1/models` | GET | AmpCode model listing |
| `/api/provider/*` | POST | AmpCode provider-specific routes |
| `/api/internal?webSearch2` | POST | AmpCode web search (via Copilot Responses API) |
| `/api/internal?extractWebPageContent` | POST | AmpCode page extraction (via Jina Reader) |
| `/api/*` | ANY | AmpCode management proxy to ampcode.com |
| `/usage` | GET | Copilot usage and quota info |

## Configuration

### CLI Flags

```
./copilot2api [options]

  -host string       Server host (default "127.0.0.1")
  -port int          Server port (default 7777)
  -token-dir string  Token storage directory (default ~/.config/copilot2api)
  -debug             Enable debug logging
  -version           Show version and exit
```

### Environment Variables

Environment variables are used as defaults when flags are not provided:

| Variable | Description | Default |
|----------|-------------|---------|
| `COPILOT2API_HOST` | Server host | `127.0.0.1` |
| `COPILOT2API_PORT` | Server port | `7777` |
| `COPILOT2API_TOKEN_DIR` | Token storage directory | `~/.config/copilot2api` |
| `COPILOT2API_DEBUG` | Enable debug logging (`true`/`false`, `1`/`0`) | `false` |
| `COPILOT2API_NO_MODEL_UPGRADE` | Disable model auto-upgrade (`true`/`false`, `1`/`0`) | `false` |
| `JINA_API_KEY` | Jina API key for amp page extraction (optional) | — |

CLI flags take precedence over environment variables.

## How It Works

1. Authenticates with GitHub via Device Flow OAuth
2. Exchanges GitHub token for Copilot API token (auto-refreshes)
3. Proxies OpenAI-format requests directly to Copilot API
4. Routes Anthropic Messages requests by model capabilities (native `/v1/messages`, translated `/responses`, or translated `/chat/completions`)
5. Automatically detects API endpoint from token (Individual/Business/Enterprise)

## Development

```bash
go test ./...              # Run tests
go build -o copilot2api .  # Build
```

## License

MIT
