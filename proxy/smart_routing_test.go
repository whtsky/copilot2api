package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/types"
	"github.com/whtsky/copilot2api/internal/upstream"
)

// --- extractModelField tests ---

func TestExtractModelField(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"valid model", `{"model":"gpt-4","messages":[]}`, "gpt-4"},
		{"no model field", `{"messages":[]}`, ""},
		{"invalid JSON", `{broken`, ""},
		{"empty body", ``, ""},
		{"model is empty string", `{"model":""}`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractModelField([]byte(tt.body))
			if got != tt.want {
				t.Errorf("extractModelField() = %q, want %q", got, tt.want)
			}
		})
	}
}

// --- resolveTargetEndpoint tests ---

func TestResolveTargetEndpoint_NilModelsCache(t *testing.T) {
	h := &Handler{}
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"model":"gpt-4"}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (passthrough for nil cache)", got)
	}
}

func TestResolveTargetEndpoint_NoModelField(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "gpt-4", "supported_endpoints": []string{"/v1/chat/completions"}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"messages":[]}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (no model field)", got)
	}
}

func TestResolveTargetEndpoint_UnknownModel(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "gpt-4", "supported_endpoints": []string{"/v1/chat/completions"}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"model":"unknown-model"}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (unknown model → passthrough)", got)
	}
}

func TestResolveTargetEndpoint_ModelSupportsBoth(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "gpt-4", "supported_endpoints": []string{"/v1/chat/completions", "/v1/responses"}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	// Request /chat/completions → model supports it → no conversion
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"model":"gpt-4"}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (model supports both → preferred)", got)
	}

	// Request /responses → model supports it → no conversion
	req = httptest.NewRequest("POST", "/v1/responses", nil)
	got = h.resolveTargetEndpoint(req, "/responses", []byte(`{"model":"gpt-4"}`))
	if got != "/responses" {
		t.Errorf("got %q, want /responses (model supports both → preferred)", got)
	}
}

func TestResolveTargetEndpoint_NeedsConversion(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "o3-mini", "supported_endpoints": []string{"/v1/responses"}},
				{"id": "legacy-model", "supported_endpoints": []string{"/v1/chat/completions"}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	// o3-mini only supports /responses → /chat/completions request → convert to /responses
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"model":"o3-mini"}`))
	if got != "/responses" {
		t.Errorf("got %q, want /responses (o3-mini only supports responses)", got)
	}

	// legacy-model only supports /chat/completions → /responses request → convert to /chat/completions
	req = httptest.NewRequest("POST", "/v1/responses", nil)
	got = h.resolveTargetEndpoint(req, "/responses", []byte(`{"model":"legacy-model"}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (legacy-model only supports chat)", got)
	}
}

func TestResolveTargetEndpoint_ModelSupportsNeither(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "embed-model", "supported_endpoints": []string{"/v1/embeddings"}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	got := h.resolveTargetEndpoint(req, "/chat/completions", []byte(`{"model":"embed-model"}`))
	if got != "/chat/completions" {
		t.Errorf("got %q, want /chat/completions (supports neither → passthrough)", got)
	}
}

// --- Smart routing integration tests ---

func TestSmartRouting_ChatToResponsesNonStreaming(t *testing.T) {
	// Fake upstream that serves /models and /responses
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "resp-only-model", "supported_endpoints": []string{"/v1/responses"}},
				},
			})
		case "/responses":
			// Verify we received a Responses API request
			body, _ := io.ReadAll(r.Body)
			var req map[string]interface{}
			if err := json.Unmarshal(body, &req); err != nil {
				t.Errorf("Failed to parse request to /responses: %v", err)
			}
			if req["model"] != "resp-only-model" {
				t.Errorf("model = %v, want resp-only-model", req["model"])
			}
			// Return a Responses API result
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "resp_result_1",
				"model":  "resp-only-model",
				"status": "completed",
				"output": []map[string]interface{}{
					{
						"type": "message",
						"content": []map[string]interface{}{
							{"type": "output_text", "text": "Hello from responses"},
						},
					},
				},
				"usage": map[string]interface{}{
					"input_tokens":  10,
					"output_tokens": 5,
				},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, "unexpected path: %s", r.URL.Path)
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	// Send a Chat Completions request for a model that only supports /responses
	chatBody := `{"model":"resp-only-model","messages":[{"role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(chatBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/chat/completions")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}

	// Verify the response is in Chat Completions format
	var resp map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if resp["object"] != "chat.completion" {
		t.Errorf("object = %v, want chat.completion", resp["object"])
	}
	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("Expected non-empty choices array")
	}
}

func TestSmartRouting_ResponsesToChatNonStreaming(t *testing.T) {
	// Fake upstream that serves /models and /chat/completions
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "chat-only-model", "supported_endpoints": []string{"/v1/chat/completions"}},
				},
			})
		case "/chat/completions":
			// Verify we received a Chat Completions request
			body, _ := io.ReadAll(r.Body)
			var req map[string]interface{}
			json.Unmarshal(body, &req)
			if req["model"] != "chat-only-model" {
				t.Errorf("model = %v, want chat-only-model", req["model"])
			}
			// Return Chat Completions response
			text := "Hello from chat"
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":      "chatcmpl_1",
				"object":  "chat.completion",
				"model":   "chat-only-model",
				"choices": []map[string]interface{}{
					{
						"index":         0,
						"finish_reason": "stop",
						"message": map[string]interface{}{
							"role":    "assistant",
							"content": text,
						},
					},
				},
				"usage": map[string]interface{}{
					"prompt_tokens":     15,
					"completion_tokens": 5,
					"total_tokens":      20,
				},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	// Send a Responses API request for a model that only supports /chat/completions
	responsesBody := `{"model":"chat-only-model","input":[{"type":"message","role":"user","content":"hi"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(responsesBody))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/responses")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}

	// Verify the response is in Responses API format
	var resp map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if resp["status"] != "completed" {
		t.Errorf("status = %v, want completed", resp["status"])
	}
}

func TestSmartRouting_PassthroughWhenModelSupportsEndpoint(t *testing.T) {
	var requestedPath string
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "gpt-4", "supported_endpoints": []string{"/v1/chat/completions", "/v1/responses"}},
				},
			})
		default:
			requestedPath = r.URL.Path
			json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	body := `{"model":"gpt-4","messages":[],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/chat/completions")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	// Should have hit /chat/completions directly (passthrough), not /responses
	if requestedPath != "/chat/completions" {
		t.Errorf("upstream path = %q, want /chat/completions", requestedPath)
	}
}

// --- Streaming conversion integration tests ---

func TestSmartRouting_ChatToResponsesStreaming(t *testing.T) {
	sseData := strings.Join([]string{
		"event: response.created",
		`data: {"type":"response.created","response":{"id":"resp_s1","model":"resp-only","status":"in_progress"}}`,
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"Hi","output_index":0}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{"id":"resp_s1","model":"resp-only","status":"completed","usage":{"input_tokens":5,"output_tokens":2}}}`,
		"",
	}, "\n")

	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "resp-only", "supported_endpoints": []string{"/v1/responses"}},
				},
			})
		case "/responses":
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(sseData))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	body := `{"model":"resp-only","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/chat/completions")

	result := rec.Body.String()
	// Should contain Chat Completions format chunks
	if !strings.Contains(result, "chat.completion.chunk") {
		t.Errorf("Expected chat.completion.chunk in output, got:\n%s", result)
	}
	if !strings.Contains(result, "data: [DONE]") {
		t.Errorf("Expected data: [DONE] in output, got:\n%s", result)
	}
}

func TestSmartRouting_ResponsesToChatStreaming(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"id":"chatcmpl_s1","object":"chat.completion.chunk","model":"chat-only","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		`data: {"id":"chatcmpl_s1","object":"chat.completion.chunk","model":"chat-only","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
		"data: [DONE]",
		"",
	}, "\n")

	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "chat-only", "supported_endpoints": []string{"/v1/chat/completions"}},
				},
			})
		case "/chat/completions":
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(sseData))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, 5*time.Minute),
	}

	body := `{"model":"chat-only","input":[{"type":"message","role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/responses")

	result := rec.Body.String()
	// Should contain Responses API format events
	if !strings.Contains(result, "event: response.created") {
		t.Errorf("Expected event: response.created in output, got:\n%s", result)
	}
	if !strings.Contains(result, "event: response.completed") {
		t.Errorf("Expected event: response.completed in output, got:\n%s", result)
	}
}

// --- streamResponsesAsChatChunks tests ---

func TestStreamResponsesAsChatChunks_Basic(t *testing.T) {
	sseInput := strings.Join([]string{
		"event: response.created",
		`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-4","status":"in_progress"}}`,
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"Hello"}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-4","status":"completed"}}`,
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamResponsesAsChatChunks(rec, body)
	if err != nil {
		t.Fatalf("streamResponsesAsChatChunks error: %v", err)
	}

	result := rec.Body.String()
	if !strings.Contains(result, "chat.completion.chunk") {
		t.Errorf("Expected chat.completion.chunk, got:\n%s", result)
	}
	if !strings.Contains(result, "data: [DONE]") {
		t.Error("Expected data: [DONE]")
	}
}

func TestStreamResponsesAsChatChunks_IgnoresDataDone(t *testing.T) {
	sseInput := strings.Join([]string{
		"event: response.created",
		`data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-4"}}`,
		"",
		"data: [DONE]",
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","delta":"Hello"}`,
		"",
		"event: response.completed",
		`data: {"type":"response.completed","response":{"id":"resp_1","status":"completed"}}`,
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamResponsesAsChatChunks(rec, body)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	result := rec.Body.String()
	// Should still contain the text delta (data: [DONE] should not terminate)
	if !strings.Contains(result, "Hello") {
		t.Errorf("Expected Hello after data: [DONE], got:\n%s", result)
	}
}

func TestStreamResponsesAsChatChunks_EventTypeFromSSELine(t *testing.T) {
	// Some implementations put event type only in the SSE "event:" line,
	// not in the JSON data
	sseInput := strings.Join([]string{
		"event: response.created",
		`data: {"response":{"id":"resp_1","model":"gpt-4"}}`,
		"",
		"event: response.completed",
		`data: {"response":{"id":"resp_1","status":"completed"}}`,
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamResponsesAsChatChunks(rec, body)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	result := rec.Body.String()
	if !strings.Contains(result, "data: [DONE]") {
		t.Errorf("Expected data: [DONE], got:\n%s", result)
	}
}

// --- streamChatChunksAsResponsesEvents tests ---

func TestStreamChatChunksAsResponsesEvents_Basic(t *testing.T) {
	sseInput := strings.Join([]string{
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		"data: [DONE]",
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamChatChunksAsResponsesEvents(rec, body)
	if err != nil {
		t.Fatalf("streamChatChunksAsResponsesEvents error: %v", err)
	}

	result := rec.Body.String()
	if !strings.Contains(result, "event: response.created") {
		t.Errorf("Expected response.created, got:\n%s", result)
	}
	if !strings.Contains(result, "event: response.output_text.delta") {
		t.Errorf("Expected response.output_text.delta, got:\n%s", result)
	}
	if !strings.Contains(result, "event: response.completed") {
		t.Errorf("Expected response.completed, got:\n%s", result)
	}
}

func TestStreamChatChunksAsResponsesEvents_DeferredTermination(t *testing.T) {
	// finish_reason on one chunk, usage on the next (empty choices)
	sseInput := strings.Join([]string{
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		"data: [DONE]",
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamChatChunksAsResponsesEvents(rec, body)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	result := rec.Body.String()
	if !strings.Contains(result, "event: response.completed") {
		t.Errorf("Expected response.completed, got:\n%s", result)
	}
	// Should contain usage info in the completion event
	if !strings.Contains(result, `"input_tokens":10`) {
		t.Errorf("Expected input_tokens in completion event, got:\n%s", result)
	}
}

func TestStreamChatChunksAsResponsesEvents_DoneWithoutUsage(t *testing.T) {
	// finish_reason seen but [DONE] arrives before usage chunk
	sseInput := strings.Join([]string{
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"data: [DONE]",
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamChatChunksAsResponsesEvents(rec, body)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	result := rec.Body.String()
	// Should still emit response.completed even without usage
	if !strings.Contains(result, "event: response.completed") {
		t.Errorf("Expected response.completed, got:\n%s", result)
	}
}

func TestStreamChatChunksAsResponsesEvents_DoneWithoutFinishReason(t *testing.T) {
	// Edge case: [DONE] without any finish_reason
	sseInput := strings.Join([]string{
		`data: {"id":"chatcmpl_1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
		"data: [DONE]",
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	body := io.NopCloser(strings.NewReader(sseInput))
	err := streamChatChunksAsResponsesEvents(rec, body)
	if err != nil {
		t.Fatalf("error: %v", err)
	}

	result := rec.Body.String()
	// Should emit a fallback response.completed
	if !strings.Contains(result, "event: response.completed") {
		t.Errorf("Expected response.completed, got:\n%s", result)
	}
}

// --- writeResponsesSSEEvent tests ---

func TestWriteResponsesSSEEvent(t *testing.T) {
	var buf bytes.Buffer
	event := types.ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "hello",
	}

	err := writeResponsesSSEEvent(&buf, event)
	if err != nil {
		t.Fatalf("writeResponsesSSEEvent error: %v", err)
	}

	result := buf.String()
	if !strings.HasPrefix(result, "event: response.output_text.delta\n") {
		t.Errorf("Expected event: prefix, got: %q", result)
	}
	if !strings.Contains(result, "data: ") {
		t.Errorf("Expected data: prefix, got: %q", result)
	}
	if !strings.HasSuffix(result, "\n\n") {
		t.Errorf("Expected double newline suffix, got: %q", result)
	}
}
