package anthropic

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

func TestReadSSEEventMultiLineData(t *testing.T) {
	input := strings.Join([]string{
		"event: response.output_text.delta",
		"data: {\"type\":\"response.output_text.delta\",",
		"data: \"delta\":\"hello\"}",
		"",
	}, "\n")

	reader := bufio.NewReader(strings.NewReader(input))
	event, err := readSSEEvent(reader)
	if err != nil {
		t.Fatalf("readSSEEvent returned error: %v", err)
	}
	if event == nil {
		t.Fatal("readSSEEvent returned nil event")
	}

	if event.Event != "response.output_text.delta" {
		t.Fatalf("event type = %q, want %q", event.Event, "response.output_text.delta")
	}

	wantData := "{\"type\":\"response.output_text.delta\",\n\"delta\":\"hello\"}"
	if event.Data != wantData {
		t.Fatalf("event data = %q, want %q", event.Data, wantData)
	}
}

func TestReadSSEEventEOFWithoutData(t *testing.T) {
	reader := bufio.NewReader(strings.NewReader(""))
	event, err := readSSEEvent(reader)
	if err == nil {
		t.Fatal("expected EOF error")
	}
	if err != io.EOF {
		t.Fatalf("error = %v, want io.EOF", err)
	}
	if event != nil {
		t.Fatalf("event = %#v, want nil", event)
	}
}

func TestNormalizeNativeMessagesBody_RemovesCacheControlScope(t *testing.T) {
	body := []byte(`{
		"model": "claude-opus-4-6-20250514",
		"context_management": {"type": "auto"},
		"system": [
			{"type": "text", "text": "one"},
			{"type": "text", "text": "two", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "workspace"}}
		],
		"messages": [
			{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "scope": "tool"}}]}
		],
		"max_tokens": 16
	}`)

	normalized, err := normalizeNativeMessagesBody(body, "claude-opus-4.6", true)
	if err != nil {
		t.Fatalf("normalizeNativeMessagesBody returned error: %v", err)
	}

	info := inspectCacheControl(normalized)
	if info.ScopeCount != 0 {
		t.Fatalf("ScopeCount = %d, want 0; paths=%v", info.ScopeCount, info.ScopePaths)
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(normalized, &decoded); err != nil {
		t.Fatalf("failed to decode normalized body: %v", err)
	}

	if decoded["model"] != "claude-opus-4.6" {
		t.Fatalf("model = %v, want claude-opus-4.6", decoded["model"])
	}
	if _, ok := decoded["context_management"]; ok {
		t.Fatalf("context_management still present")
	}

	system := decoded["system"].([]interface{})
	cacheControl := system[1].(map[string]interface{})["cache_control"].(map[string]interface{})
	if cacheControl["type"] != "ephemeral" {
		t.Fatalf("system cache_control.type = %v, want ephemeral", cacheControl["type"])
	}
	if cacheControl["ttl"] != "1h" {
		t.Fatalf("system cache_control.ttl = %v, want 1h", cacheControl["ttl"])
	}
	if _, ok := cacheControl["scope"]; ok {
		t.Fatalf("system cache_control.scope still present")
	}

	messages := decoded["messages"].([]interface{})
	parts := messages[0].(map[string]interface{})["content"].([]interface{})
	messageCacheControl := parts[0].(map[string]interface{})["cache_control"].(map[string]interface{})
	if _, ok := messageCacheControl["scope"]; ok {
		t.Fatalf("message cache_control.scope still present")
	}
}
