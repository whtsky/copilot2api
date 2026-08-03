package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/upstream"
)

func TestModelRoute_DefaultRunsBeforeSmartRouting(t *testing.T) {
	var mu sync.Mutex
	var receivedEndpoint string
	var receivedModel string

	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "gpt-5.6-luna", "supported_endpoints": []string{"/v1/responses"}},
				},
			})
		case "/responses":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("read /responses body: %v", err)
			}
			var request struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(body, &request); err != nil {
				t.Errorf("decode /responses body: %v", err)
			}
			mu.Lock()
			receivedEndpoint = r.URL.Path
			receivedModel = request.Model
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "resp_model_route",
				"model":  request.Model,
				"status": "completed",
				"output": []map[string]interface{}{
					{
						"type": "message",
						"content": []map[string]interface{}{
							{"type": "output_text", "text": "approved"},
						},
					},
				},
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer fakeUpstream.Close()

	routes, err := models.ParseModelRoutes("")
	if err != nil {
		t.Fatal(err)
	}
	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, time.Minute),
		modelRoutes: routes,
	}

	body := `{"model":"codex-auto-review","messages":[{"role":"user","content":"review"}],"stream":false}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/chat/completions")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	mu.Lock()
	defer mu.Unlock()
	if receivedEndpoint != "/responses" {
		t.Fatalf("upstream endpoint = %q, want /responses", receivedEndpoint)
	}
	if receivedModel != "gpt-5.6-luna" {
		t.Fatalf("upstream model = %q, want gpt-5.6-luna", receivedModel)
	}
}

func TestModelRoute_DefaultUpdatesStreamingBody(t *testing.T) {
	var receivedModel string
	fakeUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"data": []map[string]interface{}{
					{"id": "gpt-5.6-luna", "supported_endpoints": []string{"/v1/responses"}},
				},
			})
		case "/responses":
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Errorf("read /responses body: %v", err)
			}
			var request struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(body, &request); err != nil {
				t.Errorf("decode /responses body: %v", err)
			}
			receivedModel = request.Model
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream\",\"model\":\"gpt-5.6-luna\",\"status\":\"in_progress\"}}\n\n")
			fmt.Fprint(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"model\":\"gpt-5.6-luna\",\"status\":\"completed\"}}\n\n")
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer fakeUpstream.Close()

	routes, err := models.ParseModelRoutes("")
	if err != nil {
		t.Fatal(err)
	}
	tp := &stubTokenProvider{baseURL: fakeUpstream.URL}
	uc := upstream.NewClient(tp, nil, false)
	h := &Handler{
		upstream:    uc,
		modelsCache: models.NewCache(uc, time.Minute),
		modelRoutes: routes,
	}

	body := `{"model":"codex-auto-review","input":[],"stream":true}`
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.handlePassthrough(rec, req, "/responses")

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body: %s", rec.Code, rec.Body.String())
	}
	if receivedModel != "gpt-5.6-luna" {
		t.Fatalf("upstream streaming model = %q, want gpt-5.6-luna", receivedModel)
	}
}
