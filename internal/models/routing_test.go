package models

import (
	"encoding/json"
	"testing"
)

func TestParseModelRoutes_DefaultsCodexAutoReviewToLuna(t *testing.T) {
	routes, err := ParseModelRoutes("")
	if err != nil {
		t.Fatalf("ParseModelRoutes() error = %v", err)
	}

	got, ok := routes.Resolve("codex-auto-review")
	if !ok || got != "gpt-5.6-luna" {
		t.Fatalf("default route = (%q, %v), want (%q, true)", got, ok, "gpt-5.6-luna")
	}
}

func TestParseModelRoutes_ExplicitConfigReplacesDefaults(t *testing.T) {
	routes, err := ParseModelRoutes(`{"custom-source":"custom-target"}`)
	if err != nil {
		t.Fatalf("ParseModelRoutes() error = %v", err)
	}

	if _, ok := routes.Resolve("codex-auto-review"); ok {
		t.Fatal("explicit route configuration should replace the default route set")
	}
	if got, ok := routes.Resolve("custom-source"); !ok || got != "custom-target" {
		t.Fatalf("custom route = (%q, %v), want (%q, true)", got, ok, "custom-target")
	}

	emptyRoutes, err := ParseModelRoutes(`{}`)
	if err != nil {
		t.Fatalf("ParseModelRoutes({}) error = %v", err)
	}
	if _, ok := emptyRoutes.Resolve("codex-auto-review"); ok {
		t.Fatal("empty explicit route configuration should disable the built-in route")
	}
}

func TestParseModelRoutes_RejectsInvalidConfig(t *testing.T) {
	tests := []struct {
		name string
		raw  string
	}{
		{name: "malformed json", raw: `{not-json`},
		{name: "empty source", raw: `{"":"target"}`},
		{name: "empty target", raw: `{"source":""}`},
		{name: "self map", raw: `{"same":"same"}`},
		{name: "whitespace source", raw: `{" source":"target"}`},
		{name: "whitespace target", raw: `{"source":" target"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ParseModelRoutes(tt.raw); err == nil {
				t.Fatalf("ParseModelRoutes(%q) returned nil error", tt.raw)
			}
		})
	}
}

func TestModelRoutesRewriteJSON(t *testing.T) {
	routes, err := ParseModelRoutes("")
	if err != nil {
		t.Fatal(err)
	}

	body := []byte(`{"model":"codex-auto-review","input":[{"type":"message","content":"review"}],"metadata":{"model":"codex-auto-review"}}`)
	rewritten, from, to, changed := routes.RewriteJSON(body)
	if !changed || from != "codex-auto-review" || to != "gpt-5.6-luna" {
		t.Fatalf("RewriteJSON metadata = (%q, %q, %v), want source, target, true", from, to, changed)
	}

	var payload struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(rewritten, &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Model != "gpt-5.6-luna" {
		t.Fatalf("rewritten model = %q, want %q", payload.Model, "gpt-5.6-luna")
	}
	if string(rewritten) == string(body) {
		t.Fatal("RewriteJSON returned the original body")
	}
}

func TestModelRoutesRewriteJSONLeavesUnmatchedBodiesUnchanged(t *testing.T) {
	routes, err := ParseModelRoutes("")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name string
		body []byte
	}{
		{name: "unknown model", body: []byte(`{"model":"gpt-5.6-sol"}`)},
		{name: "missing model", body: []byte(`{"input":[]}`)},
		{name: "non-string model", body: []byte(`{"model":42}`)},
		{name: "invalid json", body: []byte(`{broken`)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rewritten, from, to, changed := routes.RewriteJSON(tt.body)
			if changed || from != "" || to != "" {
				t.Fatalf("RewriteJSON metadata = (%q, %q, %v), want empty, empty, false", from, to, changed)
			}
			if string(rewritten) != string(tt.body) {
				t.Fatalf("RewriteJSON changed unmatched body from %q to %q", tt.body, rewritten)
			}
		})
	}
}
