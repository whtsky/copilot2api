# Configurable Model Routing for Unsupported Provider Model Names Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Let copilot2api exact-match and rewrite provider-incompatible model IDs such as `codex-auto-review` to a configurable supported model such as `gpt-5.6-luna`.

**Architecture:** Add a validated JSON model-route map loaded from `COPILOT2API_MODEL_ROUTES` (with a matching CLI flag). Apply one exact-match rewrite to OpenAI-compatible request bodies before the existing smart endpoint router runs. Keep the route internal: `/v1/models` remains the upstream catalog, and native Anthropic/Gemini model handling is unchanged.

**Tech Stack:** Go 1.26, `encoding/json`, existing `proxy.Handler` and `internal/models` packages, Docker environment configuration.

---

## Evidence and Root Cause

- GitHub issue #31255 reports that Codex's auto-review request sends the literal model `codex-auto-review`, while the upstream accepts different model IDs.
- The live local proxy's `/v1/models` response contains `gpt-5.6-luna` and `gpt-5.6-sol`, but not `codex-auto-review`.
- A direct `POST /v1/responses` through the live proxy with `model: "codex-auto-review"` returned HTTP 400 with `model_not_supported`.
- Codex CLI 0.144.6 reproduced the same failure with `approvals_reviewer = "auto_review"`, `approval_policy = "on-request"`, read-only sandboxing, and a harmless `curl` command. Codex reported: `Automatic approval review failed` with the same `model_not_supported` response.
- A control request using `gpt-5.6-luna` and supported `reasoning.effort: "low"` returned HTTP 200.
- Current `proxy/handler.go` only extracts the model to select `/responses` vs `/chat/completions`; unknown IDs pass through unchanged. Current `internal/upstream/client.go` injects `contextTier` but has no model rewrite layer.

## Configuration Contract

Use an exact-match JSON object:

```yaml
environment:
  COPILOT2API_MODEL_ROUTES: '{"codex-auto-review":"gpt-5.6-luna"}'
```

Semantics:

- Unset or empty: use the built-in `codex-auto-review` → `gpt-5.6-luna` route.
- Non-empty JSON: replace the built-in route set with the explicitly configured map; use `{}` to disable all routes.
- Keys and values are model IDs; matching is exact and case-sensitive.
- Each source maps directly to one target; do not recursively chain routes.
- Empty IDs and self-maps are configuration errors; malformed JSON fails startup with a useful error.
- Routing applies to `POST /v1/responses` and `POST /v1/chat/completions`, including the existing `/amp/v1/*` and provider-specific OpenAI routes after prefix stripping.
- Bodies without a string `model` field, invalid JSON, and unrelated endpoints pass through to the existing validation/error path.
- The upstream response is not rewritten. The request is sent with the effective target model; the upstream catalog is not polluted with pseudo-model aliases.

---

## Implementation Tasks

### Task 1: Add a validated model-route type

**Objective:** Parse and apply exact-match model routes without coupling configuration parsing to HTTP handling.

**Files:**
- Create: `internal/models/routing.go`
- Create: `internal/models/routing_test.go`

**Step 1: Write failing tests**

Cover:

- empty input selects the built-in compatibility route; `{}` explicitly disables routing;
- valid JSON parses and rewrites `codex-auto-review` to `gpt-5.6-luna`;
- unknown model is unchanged;
- missing/non-string `model` is unchanged;
- malformed JSON, empty source/target, and self-maps return errors;
- route resolution is one-hop and exact-match.

**Step 2: Run the focused tests**

```bash
go test ./internal/models -run 'Test(ModelRoutes|RewriteModel)' -v
```

Expected: FAIL because the route type and parser do not exist.

**Step 3: Implement the minimal API**

Use a small value type, for example:

```go
type ModelRoutes struct {
    routes map[string]string
}

func ParseModelRoutes(raw string) (ModelRoutes, error)
func (r ModelRoutes) Resolve(model string) (string, bool)
func (r ModelRoutes) RewriteJSON(body []byte) (rewritten []byte, from string, to string, changed bool)
```

`RewriteJSON` should preserve the original bytes when no route applies. It should only replace the top-level JSON `model` string and should not recursively inspect messages, tools, metadata, or arbitrary nested objects.

**Step 4: Run the focused tests again**

```bash
go test ./internal/models -run 'Test(ModelRoutes|RewriteModel)' -v
```

Expected: PASS.

---

### Task 2: Load model routes from startup configuration

**Objective:** Make routing configurable in both Docker and binary deployments, with fail-fast validation.

**Files:**
- Modify: `main.go:31-71` for the flag and environment default
- Modify: `main.go:78-136` for parsing and handler wiring

**Step 1: Add the configuration input**

Add a `-model-routes` string flag whose environment fallback is `COPILOT2API_MODEL_ROUTES`. Keep the existing precedence convention: a non-empty CLI flag wins over the environment variable.

Parse the final string with `models.ParseModelRoutes` after logging is initialized and before authentication/server startup. On error, log the configuration error and exit without starting the HTTP server.

**Step 2: Pass the parsed routes to the OpenAI proxy handler**

Update the `proxy.NewHandler` call to receive the immutable `models.ModelRoutes` value. Do not put this setting into the Copilot credential directory or modify the upstream token format.

**Step 3: Verify startup configuration locally**

Run the normal package tests and a binary help check:

```bash
go test ./...
go build -o /tmp/copilot2api-model-routing .
/tmp/copilot2api-model-routing -h
```

Expected: the new flag is listed and existing tests remain green.

---

### Task 3: Rewrite before smart endpoint routing

**Objective:** Ensure the target model, not the unsupported source alias, is used for both model capability lookup and the upstream request.

**Files:**
- Modify: `proxy/handler.go:20-32` to store routes and update `NewHandler`
- Modify: `proxy/handler.go:82-107` to rewrite the already-buffered body before `resolveTargetEndpoint`
- Modify: `proxy/handler_test.go`

**Step 1: Add the handler regression test**

Use the existing fake-upstream test pattern. Configure:

```json
{"codex-auto-review":"gpt-5.6-luna"}
```

Have `/models` advertise only `gpt-5.6-luna` on `/v1/responses`, send a client request to `/v1/chat/completions` with `model: "codex-auto-review"`, and assert that:

- the fake upstream receives `/responses` through the existing smart router;
- the converted request body contains `model: "gpt-5.6-luna"`;
- the client still receives the expected Chat Completions response shape.

Add a direct `/responses` non-streaming case and a streaming case so the rewrite is proven on both body paths.

**Step 2: Run the new tests before implementation**

```bash
go test ./proxy -run 'Test.*ModelRoute' -v
```

Expected: FAIL because `Handler` does not yet have route state or rewrite logic.

**Step 3: Implement the narrow integration**

After `handlePassthrough` has read and size-checked `bodyBytes`, and before `handlePassthroughBody` calls `resolveTargetEndpoint`:

1. Apply `ModelRoutes.RewriteJSON` only for `/chat/completions` and `/responses`.
2. Reset `r.Body` and `r.ContentLength` when the body changes, because streaming passthrough reads `r.Body` later.
3. Log only structured `from`, `to`, and endpoint fields; never log the full request body.
4. Leave `/v1/models`, `/v1/embeddings`, `/v1/messages`, and Gemini URL/model handling unchanged.

This placement is important: rewriting only inside `upstream.Client.Do` would fix the request but would leave `resolveTargetEndpoint` looking up the unsupported source ID and could bypass endpoint conversion for future route targets.

**Step 4: Run the focused and full tests**

```bash
go test ./proxy -run 'Test.*ModelRoute' -v
go test ./...
```

Expected: PASS, with no regressions in existing smart-routing or streaming tests.

---

### Task 4: Document the operational configuration

**Objective:** Make the workaround discoverable and preserve the user-visible behavior contract.

**Files:**
- Modify: `README.md:111-126` (Codex configuration example)
- Modify: `README.md:241-267` (flags and environment table)
- Modify: `CHANGELOG.md:3-4` under `## [Unreleased]`

**Step 1: Document the environment example**

Show the optional Docker/binary configuration:

```bash
COPILOT2API_MODEL_ROUTES='{"codex-auto-review":"gpt-5.6-luna"}'
```

Explain that the target must be a model exposed by the upstream `/v1/models` catalog and that changing the environment requires restarting the process/container.

**Step 2: Document the CLI flag and exact-match semantics**

State that the map is exact-match, one-hop, request-only routing and that the source alias is not added to `/v1/models`.

**Step 3: Add an Unreleased changelog entry**

Use a short user-facing feature bullet describing configurable model routing. Do not invent a release version.

---

### Task 5: Verify the real Codex regression path

**Objective:** Prove the shipped behavior fixes the original integration, not only unit tests.

**Files:**
- No source files; use an isolated build/container configuration.

**Step 1: Verify the configured target exists**

```bash
curl -fsS http://127.0.0.1:7777/v1/models -H 'Authorization: Bearer dummy'
```

Confirm the chosen target ID is present before enabling the route.

**Step 2: Run the Codex reproduction with routing enabled**

Use the same Codex setup that reproduced the failure:

```bash
codex exec --ephemeral --skip-git-repo-check --json \
  -C /tmp -m gpt-5.6-sol -s read-only \
  -c 'approval_policy="on-request"' \
  -c 'approvals_reviewer="auto_review"' \
  -c 'model_provider="copilot2api"' \
  -c 'model_providers.copilot2api.base_url="http://127.0.0.1:7777/v1"' \
  -c 'model_providers.copilot2api.wire_api="responses"' \
  -c 'model_providers.copilot2api.api_key="dummy"' \
  'Run exactly this harmless command with the shell tool: curl -fsSI --max-time 5 https://example.com.'
```

Expected: no `model_not_supported` auto-review failure; the proxy log records a `codex-auto-review` → `gpt-5.6-luna` route and Codex receives a normal reviewer result. The reviewer may still deny an action for policy reasons; the acceptance criterion is that denial must be a real review decision, not model-name transport failure.

**Step 3: Verify the default and explicit-disable paths**

Run the same request with no `COPILOT2API_MODEL_ROUTES` setting and confirm the built-in route works. Then run with `COPILOT2API_MODEL_ROUTES='{}'` and confirm the original unsupported source model produces the previous 400 response.

**Step 4: Clean up verification artifacts**

Remove temporary binaries, logs, containers, and configuration directories created for the check. Do not leave a restart-disabled test container or a build tree behind.

---

## Acceptance Criteria

- With no route configuration, `go test ./...` and existing request behavior remain unchanged except for the built-in Codex compatibility route.
- With `COPILOT2API_MODEL_ROUTES='{"codex-auto-review":"gpt-5.6-luna"}'`, exact source requests to `/v1/responses` and `/v1/chat/completions` are sent upstream as `gpt-5.6-luna`.
- With `COPILOT2API_MODEL_ROUTES='{}'`, model routing is disabled and the previous passthrough behavior is restored.
- Smart endpoint routing sees the target model before selecting an upstream endpoint.
- Streaming and non-streaming requests both work.
- Malformed or unsafe route configuration fails fast with an actionable startup error.
- `/v1/models` does not claim that `codex-auto-review` is an upstream model.
- The real Codex auto-review path no longer fails solely because of the pseudo-model ID.

## Rollout Note

After the implementation is released, add the environment setting only when a deployment wants a custom route set, update the image digest, and restart the live proxy. The built-in Codex route requires no deployment-side environment setting.
