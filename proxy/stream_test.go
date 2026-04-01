package proxy

import (
	"errors"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsStreamingRequest(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expected bool
	}{
		{
			name:     "streaming request with stream true",
			body:     `{"model": "gpt-4", "messages": [], "stream": true}`,
			expected: true,
		},
		{
			name:     "streaming request with stream true and spaces",
			body:     `{"model": "gpt-4", "messages": [], "stream": true }`,
			expected: true,
		},
		{
			name:     "non-streaming request with stream false",
			body:     `{"model": "gpt-4", "messages": [], "stream": false}`,
			expected: false,
		},
		{
			name:     "non-streaming request without stream parameter",
			body:     `{"model": "gpt-4", "messages": []}`,
			expected: false,
		},
		{
			name:     "empty body",
			body:     "",
			expected: false,
		},
		{
			name:     "malformed JSON",
			body:     `{"model": "gpt-4", "messages": [], "stream":`,
			expected: false,
		},
		{
			name:     "stream true in message content should not match",
			body:     `{"model": "gpt-4", "messages": [{"role": "user", "content": "set \"stream\": true in the body"}], "stream": false}`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isStreamingRequest([]byte(tt.body))
			if result != tt.expected {
				t.Errorf("isStreamingRequest() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsStreamingRequest_NoBody(t *testing.T) {
	result := isStreamingRequest(nil)
	if result {
		t.Error("isStreamingRequest() should return false for nil body")
	}
}

func TestIsResponsesTerminationEvent(t *testing.T) {
	tests := []struct {
		name string
		line string
		want bool
	}{
		{name: "completed exact", line: "event: response.completed", want: true},
		{name: "completed with whitespace", line: "  event: response.completed\r\n", want: true},
		{name: "error exact", line: "event: error", want: true},
		{name: "non termination", line: "event: response.output_text.delta", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isResponsesTerminationEvent(tt.line); got != tt.want {
				t.Fatalf("isResponsesTerminationEvent(%q) = %v, want %v", tt.line, got, tt.want)
			}
		})
	}
}

// streamResponseHelper calls streamResponse on a zero-value Handler to keep tests simple.
func streamResponseHelper(t *testing.T, ssePayload string, endpoint string) string {
	t.Helper()
	h := &Handler{}
	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(ssePayload))
	if err := h.streamResponse(rec, body, endpoint); err != nil {
		t.Fatalf("streamResponse returned error: %v", err)
	}
	return rec.Body.String()
}

func TestStreamResponse_ChatCompletions_DataDone(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n"
	got := streamResponseHelper(t, sse, "/chat/completions")
	if !strings.Contains(got, "data: [DONE]") {
		t.Fatal("expected data: [DONE] in output")
	}
}

func TestStreamResponse_Responses_CompletedEvent(t *testing.T) {
	sse := strings.Join([]string{
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"hi"}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{}}`,
		"",
	}, "\n")
	got := streamResponseHelper(t, sse, "/responses")
	if !strings.Contains(got, "event: response.completed") {
		t.Fatal("expected event: response.completed in output")
	}
	if !strings.Contains(got, `"type":"response.completed"`) {
		t.Fatal("expected response.completed data line in output")
	}
}

// Regression test: data: [DONE] before response.completed must NOT terminate a
// /responses stream early — the proxy must still forward response.completed.
func TestStreamResponse_Responses_DataDoneBeforeCompleted(t *testing.T) {
	sse := strings.Join([]string{
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"hi"}`,
		"",
		"data: [DONE]",
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{}}`,
		"",
	}, "\n")
	got := streamResponseHelper(t, sse, "/responses")
	if !strings.Contains(got, "event: response.completed") {
		t.Fatalf("response.completed must be forwarded even when data: [DONE] appears first.\nGot:\n%s", got)
	}
}

func TestStreamResponse_Responses_ErrorTermination(t *testing.T) {
	sse := strings.Join([]string{
		"event: error",
		`data: {"type":"error","message":"something broke"}`,
		"",
	}, "\n")
	got := streamResponseHelper(t, sse, "/responses")
	if !strings.Contains(got, "event: error") {
		t.Fatal("expected event: error in output")
	}
}

func TestStreamResponse_Responses_IncompleteTermination(t *testing.T) {
	sse := strings.Join([]string{
		"event: response.incomplete",
		`data: {"type":"response.incomplete"}`,
		"",
	}, "\n")
	got := streamResponseHelper(t, sse, "/responses")
	if !strings.Contains(got, "event: response.incomplete") {
		t.Fatal("expected event: response.incomplete in output")
	}
}

func TestHeadersSentError(t *testing.T) {
	inner := errors.New("read: connection reset")
	hse := &headersSentError{err: inner}

	if hse.Error() != inner.Error() {
		t.Fatalf("Error() = %q, want %q", hse.Error(), inner.Error())
	}
	if !errors.Is(hse, inner) {
		t.Fatal("Unwrap should expose the inner error")
	}

	// errors.As should match *headersSentError
	var target *headersSentError
	if !errors.As(hse, &target) {
		t.Fatal("errors.As should match *headersSentError")
	}
}

// --- Issue 5: Stream EOF without terminal event ---

func TestStreamResponsesAsChatChunks_NoTerminalEvent(t *testing.T) {
	// Simulate a stream that ends without a terminal event
	sseData := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-4"}}` + "\n\n" +
		"event: response.output_text.delta\n" +
		`data: {"type":"response.output_text.delta","delta":"hello"}` + "\n\n"

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseData))

	err := streamResponsesAsChatChunks(rec, body)
	if err == nil {
		t.Fatal("expected error when stream ends without terminal event")
	}
	if !strings.Contains(err.Error(), "terminal event") {
		t.Errorf("error should mention terminal event, got: %s", err.Error())
	}
}

func TestStreamResponsesAsChatChunks_WithTerminalEvent(t *testing.T) {
	// Normal stream with terminal event should succeed
	sseData := "event: response.created\n" +
		`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-4"}}` + "\n\n" +
		"event: response.completed\n" +
		`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-4","status":"completed"}}` + "\n\n"

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseData))

	err := streamResponsesAsChatChunks(rec, body)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestStreamChatChunksAsResponsesEvents_NoTerminalEvent(t *testing.T) {
	// Chat stream that ends without [DONE] should error
	sseData := `data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` + "\n\n" +
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}` + "\n\n"

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseData))

	err := streamChatChunksAsResponsesEvents(rec, body)
	if err == nil {
		t.Fatal("expected error when chat stream ends without [DONE]")
	}
	if !strings.Contains(err.Error(), "[DONE]") {
		t.Errorf("error should mention [DONE], got: %s", err.Error())
	}
}
