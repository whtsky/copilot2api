package anthropic

import (
	"context"
	"log/slog"
	"regexp"

	"github.com/whtsky/copilot2api/internal/models"
)

// modelAliases maps model name variants to their Copilot equivalents.
// Only needed for non-obvious mappings that can't be derived algorithmically.
var modelAliases = map[string]string{
	// Non-obvious version mappings (pre-4.5 naming used single-digit versions)
	"claude-opus-4":  "claude-opus-4.5",
	"claude-sonnet-4": "claude-sonnet-4", // identity, but here for documentation
}

// versionHyphenRe matches hyphen-separated version numbers like "4-6" or "4-5"
// that appear after a letter segment (e.g. "opus-4-6"). Both digits must be single
// to avoid matching date components like "04-14" or "20-25" in "2025-04-14".
var versionHyphenRe = regexp.MustCompile(`([a-zA-Z]-)(\d)-(\d)([^0-9]|$)`)

// dateSuffixRe matches an 8-digit date suffix like "-20250514" or "-20251001"
// at the end of a model ID (optionally followed by more digits for timestamps).
var dateSuffixRe = regexp.MustCompile(`-(\d{8,})$`)

// context1mRe matches the "context-1m" token in the anthropic-beta header,
// used by Claude Code to signal the 1M context window variant.
var context1mRe = regexp.MustCompile(`\bcontext-1m\b`)

// resolveModelAlias returns the canonical model ID for Copilot's model list.
// It applies the following transformations in order:
//  1. Strip date suffixes (e.g. "-20250514")
//  2. Normalize hyphen-separated versions to dot-separated (e.g. "4-6" → "4.6")
//  3. Check explicit alias overrides for non-obvious mappings
func resolveModelAlias(modelID string) string {
	// Step 1: Strip date suffix (e.g. "claude-opus-4-6-20250514" → "claude-opus-4-6")
	stripped := dateSuffixRe.ReplaceAllString(modelID, "")
	if stripped == "" {
		stripped = modelID // safety: don't strip everything
	}

	// Step 2: Normalize hyphen-separated versions to dot-separated
	normalized := versionHyphenRe.ReplaceAllString(stripped, "${1}${2}.${3}${4}")

	// Step 3: Check explicit aliases (e.g. "claude-opus-4" → "claude-opus-4.5")
	if alias, ok := modelAliases[normalized]; ok {
		return alias
	}

	// If normalization changed anything, return the normalized form
	if normalized != modelID {
		return normalized
	}

	return modelID
}

// getModelInfo returns cached model info, fetching from upstream if needed.
func (h *Handler) getModelInfo(ctx context.Context, modelID string) (*models.Info, bool) {
	modelID = resolveModelAlias(modelID)

	infoMap, err := h.models.GetInfo(ctx)
	if err != nil {
		slog.Error("failed to fetch models for capability detection", "error", err)
		return nil, true
	}

	return infoMap[modelID], false
}

func modelSupportsEndpoint(info *models.Info, endpoint string) bool {
	return models.SupportsEndpoint(info, endpoint)
}
