package anthropic

import (
	"testing"

	"github.com/whtsky/copilot2api/internal/models"
)

func TestModelSupportsEndpoint_NormalizedV1Prefix(t *testing.T) {
	info := &models.Info{SupportedEndpoints: []string{"/messages", "/responses"}}

	if !modelSupportsEndpoint(info, "/v1/messages") {
		t.Fatal("expected /v1/messages to match /messages")
	}

	if !modelSupportsEndpoint(info, "/responses") {
		t.Fatal("expected /responses to be supported")
	}

	if modelSupportsEndpoint(info, "/v1/chat/completions") {
		t.Fatal("did not expect /v1/chat/completions to be supported")
	}
}

func TestResolveModelAlias(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		// Hyphen-separated versions are normalized to dots
		{"claude-opus-4-6", "claude-opus-4.6"},
		{"claude-opus-4-6-fast", "claude-opus-4.6-fast"},
		{"claude-sonnet-4-6", "claude-sonnet-4.6"},
		{"claude-haiku-4-5", "claude-haiku-4.5"},
		{"claude-opus-4-5", "claude-opus-4.5"},
		{"claude-sonnet-4-5", "claude-sonnet-4.5"},

		// Date suffixes are stripped, then version normalization applied
		{"claude-haiku-4-5-20251001", "claude-haiku-4.5"},
		{"claude-haiku-4.5-20251001", "claude-haiku-4.5"},
		{"claude-sonnet-4-20250514", "claude-sonnet-4"},
		{"claude-opus-4-6-20250514", "claude-opus-4.6"},
		{"claude-opus-4.6-20250514", "claude-opus-4.6"},
		{"claude-sonnet-4-6-20250514", "claude-sonnet-4.6"},
		{"claude-sonnet-4.6-20250514", "claude-sonnet-4.6"},
		{"claude-sonnet-4-5-20250514", "claude-sonnet-4.5"},
		{"claude-opus-4-5-20250514", "claude-opus-4.5"},
		{"claude-opus-4.5-20250514", "claude-opus-4.5"},

		// Non-obvious mapping via explicit alias
		{"claude-opus-4-20250514", "claude-opus-4.5"},

		// Future models: should work automatically without new explicit aliases
		{"claude-opus-4-7-20260101", "claude-opus-4.7"},
		{"claude-sonnet-5-0-20260601", "claude-sonnet-5.0"},

		// Generic normalizer: unknown model with hyphen version
		{"claude-sonnet-4-6-fast", "claude-sonnet-4.6-fast"},

		// Already canonical — no change
		{"claude-opus-4.6", "claude-opus-4.6"},
		{"claude-sonnet-4", "claude-sonnet-4"},

		// No version numbers to normalize
		{"claude-sonnet", "claude-sonnet"},

		// Hyphenated dates must NOT be corrupted
		{"claude-sonnet-4-2025-04-14", "claude-sonnet-4-2025-04-14"},
		{"claude-3-5-sonnet-2025-04-14", "claude-3.5-sonnet-2025-04-14"},

		// Non-Claude models pass through unchanged
		{"gpt-5.3-codex", "gpt-5.3-codex"},
		{"gpt-5.4", "gpt-5.4"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := resolveModelAlias(tt.input)
			if got != tt.want {
				t.Errorf("resolveModelAlias(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestUpgradeModel(t *testing.T) {
	available := map[string]*models.Info{
		"claude-opus-4.7":              {ID: "claude-opus-4.7"},
		"claude-opus-4.7-1m-internal":  {ID: "claude-opus-4.7-1m-internal"},
		"claude-opus-4.6":              {ID: "claude-opus-4.6"},
		"claude-opus-4.6-1m":           {ID: "claude-opus-4.6-1m"},
		"claude-sonnet-4.6":            {ID: "claude-sonnet-4.6"},
	}

	tests := []struct {
		input string
		want  string
	}{
		{"claude-opus-4.7", "claude-opus-4.7-1m-internal"},
		{"claude-opus-4.6", "claude-opus-4.6-1m"},
		{"claude-sonnet-4.6", "claude-sonnet-4.6"},
		{"claude-sonnet-5.0", "claude-sonnet-5.0"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := upgradeModel(tt.input, available)
			if got != tt.want {
				t.Errorf("upgradeModel(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}
