package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"

	"github.com/whtsky/copilot2api/internal/upstream"
)

// modelAliases maps common model name variants to their Copilot equivalents.
var modelAliases = map[string]string{
	"claude-haiku-4-5-20251001":  "claude-haiku-4.5",
	"claude-haiku-4.5-20251001":  "claude-haiku-4.5",
	"claude-sonnet-4-20250514":   "claude-sonnet-4",
	"claude-sonnet-4.5-20250514": "claude-sonnet-4.5",
	"claude-opus-4-20250514":     "claude-opus-4.5",
	// Hyphen-separated version aliases (e.g. Zed sends claude-opus-4-6)
	"claude-opus-4-6":      "claude-opus-4.6",
	"claude-opus-4-6-fast": "claude-opus-4.6-fast",
	"claude-sonnet-4-6":    "claude-sonnet-4.6",
	"claude-haiku-4-5":     "claude-haiku-4.5",
	"claude-opus-4-5":      "claude-opus-4.5",
	"claude-sonnet-4-5":    "claude-sonnet-4.5",
}

// versionHyphenRe matches hyphen-separated version numbers like "4-6" or "4-5"
// that appear after a letter segment (e.g. "opus-4-6"). Both digits must be single
// to avoid matching date components like "04-14" or "20-25" in "2025-04-14".
var versionHyphenRe = regexp.MustCompile(`([a-zA-Z]-)(\d)-(\d)([^0-9]|$)`)

// resolveModelAlias returns the canonical model ID, checking aliases if needed.
// If no exact alias match is found, it tries replacing hyphen-separated version
// numbers with dot-separated ones (e.g. "4-6" -> "4.6").
func resolveModelAlias(modelID string) string {
	if alias, ok := modelAliases[modelID]; ok {
		return alias
	}
	// Try normalizing hyphen-separated versions to dot-separated
	normalized := versionHyphenRe.ReplaceAllString(modelID, "${1}${2}.${3}${4}")
	if normalized != modelID {
		if alias, ok := modelAliases[normalized]; ok {
			return alias
		}
		return normalized
	}
	return modelID
}

// getModelInfo returns cached model info, fetching from upstream if needed.
func (h *Handler) getModelInfo(ctx context.Context, modelID string) (*ModelInfo, bool) {
	modelID = resolveModelAlias(modelID)

	models, err := h.models.Get(ctx, func(ctx context.Context) (map[string]*ModelInfo, error) {
		models, err := h.fetchModels(ctx)
		if err != nil {
			return nil, err
		}
		slog.Debug("models cache refreshed", "count", len(models))
		return models, nil
	})

	if err != nil {
		slog.Error("failed to fetch models for capability detection", "error", err)
		return nil, true
	}

	return models[modelID], false
}

func modelSupportsEndpoint(info *ModelInfo, endpoint string) bool {
	if info == nil {
		return false
	}

	target := normalizeModelEndpoint(endpoint)
	for _, ep := range info.SupportedEndpoints {
		if normalizeModelEndpoint(ep) == target {
			return true
		}
	}
	return false
}

func normalizeModelEndpoint(endpoint string) string {
	normalized := strings.TrimSpace(endpoint)
	normalized = strings.TrimPrefix(normalized, "/v1")
	if normalized == "" {
		return "/"
	}
	if !strings.HasPrefix(normalized, "/") {
		normalized = "/" + normalized
	}
	return normalized
}

// fetchModels fetches the models list from the upstream API.
func (h *Handler) fetchModels(ctx context.Context) (map[string]*ModelInfo, error) {
	_, respData, err := h.upstream.Do(ctx, upstream.Request{
		Method:   "GET",
		Endpoint: "/models",
	})
	if err != nil {
		return nil, fmt.Errorf("models request failed: %w", err)
	}

	var modelsResp ModelsListResponse
	if err := json.Unmarshal(respData, &modelsResp); err != nil {
		return nil, fmt.Errorf("failed to parse models response: %w", err)
	}

	models := make(map[string]*ModelInfo, len(modelsResp.Data))
	for i := range modelsResp.Data {
		models[modelsResp.Data[i].ID] = &modelsResp.Data[i]
	}

	return models, nil
}
