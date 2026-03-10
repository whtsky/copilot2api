package anthropic

import "testing"

// contentDelta extracts an *AnthropicContentDelta from a stream event's Delta field.
// Fails the test if Delta is nil or not of the expected type.
func contentDelta(t *testing.T, event AnthropicStreamEvent) *AnthropicContentDelta {
	t.Helper()
	if event.Delta == nil {
		t.Fatal("Expected non-nil delta")
	}
	cd, ok := event.Delta.(*AnthropicContentDelta)
	if !ok {
		t.Fatalf("Expected *AnthropicContentDelta, got %T", event.Delta)
	}
	return cd
}

// messageDelta extracts an *AnthropicMessageDelta from a stream event's Delta field.
// Fails the test if Delta is nil or not of the expected type.
func messageDelta(t *testing.T, event AnthropicStreamEvent) *AnthropicMessageDelta {
	t.Helper()
	if event.Delta == nil {
		t.Fatal("Expected non-nil delta")
	}
	md, ok := event.Delta.(*AnthropicMessageDelta)
	if !ok {
		t.Fatalf("Expected *AnthropicMessageDelta, got %T", event.Delta)
	}
	return md
}

// isContentDelta checks if a stream event's Delta is an *AnthropicContentDelta matching the given type.
func isContentDelta(event AnthropicStreamEvent, deltaType string) bool {
	cd, ok := event.Delta.(*AnthropicContentDelta)
	return ok && cd.Type == deltaType
}
