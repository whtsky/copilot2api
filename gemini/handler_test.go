package gemini

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/upstream"
)

type stubTokenProvider struct {
	baseURL string
}

func (s *stubTokenProvider) GetToken(_ context.Context) (string, error) {
	return "test-token", nil
}

func (s *stubTokenProvider) GetBaseURL() string {
	return s.baseURL
}

func TestHandler_ModelsListsGeminiMethods(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{
				{"id": "claude-opus-4.6-1m", "supported_endpoints": []string{"/responses"}},
				{"id": "gpt-5.4", "supported_endpoints": []string{"/chat/completions"}},
				{"id": "gpt-4o", "supported_endpoints": []string{"/chat/completions"}},
				{"id": "o3", "supported_endpoints": []string{}},
			},
		})
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{upstream: uc, models: models.NewCache(uc, 5*time.Minute)}

	req := httptest.NewRequest(http.MethodGet, "/v1beta/models", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp ModelListResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Models) != 4 {
		t.Fatalf("expected 4 models, got %d", len(resp.Models))
	}
	for _, modelID := range []string{"models/claude-opus-4.6-1m", "models/gpt-5.4", "models/gpt-4o", "models/o3"} {
		found := false
		for _, m := range resp.Models {
			if m.Name == modelID {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected model %s in list", modelID)
		}
	}
}

func TestHandler_GenerateContent_NonStreaming(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{{"id": "gpt-5.4", "supported_endpoints": []string{"/chat/completions"}}},
			})
		case "/chat/completions":
			var req map[string]interface{}
			json.NewDecoder(r.Body).Decode(&req)
			if req["model"] != "gpt-5.4" {
				t.Fatalf("unexpected model: %v", req["model"])
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "chatcmpl-1",
				"object": "chat.completion",
				"model":  "gpt-5.4",
				"choices": []map[string]interface{}{{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "hello from gpt-5.4",
					},
					"finish_reason": "stop",
				}},
				"usage": map[string]interface{}{
					"prompt_tokens":     10,
					"completion_tokens": 4,
					"total_tokens":      14,
				},
			})
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{upstream: uc, models: models.NewCache(uc, 5*time.Minute)}

	body := "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"hello\"}]}]}"
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gpt-5.4:generateContent", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var resp GenerateContentResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got := resp.Candidates[0].Content.Parts[0].Text; got != "hello from gpt-5.4" {
		t.Fatalf("unexpected text: %q", got)
	}
	if resp.UsageMetadata == nil || resp.UsageMetadata.TotalTokenCount != 14 {
		t.Fatalf("unexpected usage: %+v", resp.UsageMetadata)
	}
}

func TestHandler_StreamGenerateContent_SSE(t *testing.T) {
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{{"id": "gpt-4o", "supported_endpoints": []string{"/chat/completions"}}},
			})
			return
		}
		if r.URL.Path == "/chat/completions" {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello \"}}]}\n\n"))
			_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":2,\"total_tokens\":10}}\n\n"))
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
			return
		}
		http.NotFound(w, r)
	}))
	defer fakeUpstream.Close()

	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{upstream: uc, models: models.NewCache(uc, 5*time.Minute)}

	body := "{\"contents\":[{\"role\":\"user\",\"parts\":[{\"text\":\"hello\"}]}]}"
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gpt-4o:streamGenerateContent?alt=sse", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("expected text/event-stream, got %q", rec.Header().Get("Content-Type"))
	}
	bodyText := rec.Body.String()
	if !strings.Contains(bodyText, "data: {") {
		t.Fatalf("expected SSE data event, got: %s", bodyText)
	}
	if !strings.Contains(bodyText, "hello world") {
		t.Fatalf("expected merged Gemini text payload, got: %s", bodyText)
	}
}

func TestStreamChatAsGemini_MergesToolCallFragmentsByIndex(t *testing.T) {
	body := strings.Join([]string{
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-5.4\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\"}}]}}]}",
		"",
		"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-5.4\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"file_path\\\":\\\"README.md\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}",
		"",
		"data: [DONE]",
		"",
	}, "\n")

	rec := httptest.NewRecorder()
	if err := streamChatAsGemini(rec, strings.NewReader(body)); err != nil {
		t.Fatalf("streamChatAsGemini returned error: %v", err)
	}

	result := rec.Body.String()
	if !strings.Contains(result, "read_file") {
		t.Fatalf("expected merged tool name in output: %s", result)
	}
	if !strings.Contains(result, "file_path") {
		t.Fatalf("expected merged tool args in output: %s", result)
	}

	line := strings.TrimSpace(result)
	line = strings.TrimPrefix(line, "data: ")
	var resp GenerateContentResponse
	if err := json.NewDecoder(strings.NewReader(line)).Decode(&resp); err != nil && err != io.EOF {
		t.Fatalf("failed to decode output payload: %v", err)
	}
	if len(resp.Candidates) != 1 || len(resp.Candidates[0].Content.Parts) != 1 {
		t.Fatalf("unexpected response shape: %+v", resp)
	}
	if resp.Candidates[0].Content.Parts[0].FunctionCall == nil {
		t.Fatalf("expected function call in response: %+v", resp)
	}
}
