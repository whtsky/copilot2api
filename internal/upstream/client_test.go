package upstream

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"
)

func TestInjectContextTier_AddsField(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	body := []byte(`{"model":"gpt-4o","messages":[]}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	var tier string
	if err := json.Unmarshal(parsed["contextTier"], &tier); err != nil {
		t.Fatal("contextTier not found or not string:", err)
	}
	if tier != "long_context" {
		t.Errorf("got %q, want %q", tier, "long_context")
	}
}

func TestInjectContextTier_PreservesExisting(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	body := []byte(`{"model":"gpt-4o","contextTier":"default"}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	var tier string
	if err := json.Unmarshal(parsed["contextTier"], &tier); err != nil {
		t.Fatal(err)
	}
	if tier != "default" {
		t.Errorf("got %q, want %q (should not override)", tier, "default")
	}
}

func TestInjectContextTier_DisabledWhenEmpty(t *testing.T) {
	c := &Client{DefaultContextTier: ""}
	body := []byte(`{"model":"gpt-4o"}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if _, exists := parsed["contextTier"]; exists {
		t.Error("contextTier should not be injected when DefaultContextTier is empty")
	}
}

func TestInjectContextTier_NilBody(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	result := c.injectContextTier(nil)
	if result != nil {
		t.Error("expected nil for nil input")
	}
}

func TestInjectContextTier_InvalidJSON(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	body := []byte(`not json at all`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	if string(data) != "not json at all" {
		t.Errorf("invalid JSON should pass through unchanged, got %q", string(data))
	}
}

func TestInjectContextTier_PreservesOtherFields(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	body := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":100}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}

	if _, ok := parsed["model"]; !ok {
		t.Error("model field lost")
	}
	if _, ok := parsed["messages"]; !ok {
		t.Error("messages field lost")
	}
	if _, ok := parsed["max_tokens"]; !ok {
		t.Error("max_tokens field lost")
	}
	if _, ok := parsed["contextTier"]; !ok {
		t.Error("contextTier not added")
	}
}

func TestInjectContextTier_CheckerApproves(t *testing.T) {
	c := &Client{
		DefaultContextTier: "long_context",
		LongContextChecker: func(modelID string) bool {
			return modelID == "gpt-5.5"
		},
	}
	body := []byte(`{"model":"gpt-5.5","messages":[]}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if _, exists := parsed["contextTier"]; !exists {
		t.Error("contextTier should be injected for approved model")
	}
}

func TestInjectContextTier_CheckerRejects(t *testing.T) {
	c := &Client{
		DefaultContextTier: "long_context",
		LongContextChecker: func(modelID string) bool {
			return modelID == "gpt-5.5"
		},
	}
	body := []byte(`{"model":"gpt-5-mini","messages":[]}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if _, exists := parsed["contextTier"]; exists {
		t.Error("contextTier should NOT be injected for rejected model")
	}
}

func TestInjectContextTier_CheckerNoModelField(t *testing.T) {
	c := &Client{
		DefaultContextTier: "long_context",
		LongContextChecker: func(modelID string) bool {
			return modelID != ""
		},
	}
	body := []byte(`{"messages":[]}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if _, exists := parsed["contextTier"]; exists {
		t.Error("contextTier should NOT be injected when no model field and checker rejects empty")
	}
}

func TestInjectContextTier_NilCheckerForceMode(t *testing.T) {
	c := &Client{DefaultContextTier: "long_context"}
	body := []byte(`{"model":"any-model","messages":[]}`)

	result := c.injectContextTier(bytes.NewReader(body))
	data, _ := io.ReadAll(result)

	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	if _, exists := parsed["contextTier"]; !exists {
		t.Error("contextTier should be injected unconditionally when checker is nil (force mode)")
	}
}
