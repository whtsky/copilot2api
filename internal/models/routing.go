package models

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	defaultModelRouteSource = "codex-auto-review"
	defaultModelRouteTarget = "gpt-5.6-luna"
)

// ModelRoutes maps client-visible model IDs to upstream model IDs.
//
// The map is intentionally private and immutable after parsing so request
// handling can safely share one value across all handlers.
type ModelRoutes struct {
	routes map[string]string
}

// ParseModelRoutes parses a JSON object of source-to-target model IDs.
// An empty value selects the built-in compatibility route. A non-empty value
// replaces the built-in route set, allowing an operator to override or disable
// defaults with explicit configuration (for example, {}).
func ParseModelRoutes(raw string) (ModelRoutes, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ModelRoutes{routes: map[string]string{
			defaultModelRouteSource: defaultModelRouteTarget,
		}}, nil
	}

	var routes map[string]string
	if err := json.Unmarshal([]byte(raw), &routes); err != nil {
		return ModelRoutes{}, fmt.Errorf("model routes must be a JSON object: %w", err)
	}
	if routes == nil {
		return ModelRoutes{}, fmt.Errorf("model routes must be a JSON object, got null")
	}

	for source, target := range routes {
		if source == "" || source != strings.TrimSpace(source) {
			return ModelRoutes{}, fmt.Errorf("model route source must be a non-empty model ID without surrounding whitespace: %q", source)
		}
		if target == "" || target != strings.TrimSpace(target) {
			return ModelRoutes{}, fmt.Errorf("model route target for %q must be a non-empty model ID without surrounding whitespace", source)
		}
		if source == target {
			return ModelRoutes{}, fmt.Errorf("model route %q cannot target itself", source)
		}
	}

	return ModelRoutes{routes: routes}, nil
}

// Resolve returns the configured target for model, if any.
func (r ModelRoutes) Resolve(model string) (string, bool) {
	target, ok := r.routes[model]
	return target, ok
}

// RewriteJSON rewrites only a top-level string model field. It returns the
// original body unchanged when the JSON is invalid, the model is missing or
// non-string, or no route matches.
func (r ModelRoutes) RewriteJSON(body []byte) (rewritten []byte, from string, to string, changed bool) {
	if len(r.routes) == 0 || len(body) == 0 {
		return body, "", "", false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil || payload == nil {
		return body, "", "", false
	}

	rawModel, ok := payload["model"]
	if !ok {
		return body, "", "", false
	}

	var source string
	if err := json.Unmarshal(rawModel, &source); err != nil {
		return body, "", "", false
	}

	target, ok := r.Resolve(source)
	if !ok {
		return body, "", "", false
	}

	rawTarget, err := json.Marshal(target)
	if err != nil {
		return body, "", "", false
	}
	payload["model"] = rawTarget

	rewritten, err = json.Marshal(payload)
	if err != nil {
		return body, "", "", false
	}
	return rewritten, source, target, true
}
