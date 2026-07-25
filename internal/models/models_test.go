package models

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/whtsky/copilot2api/internal/upstream"
)

func TestPickEndpoint(t *testing.T) {
	tests := []struct {
		name      string
		info      *Info
		preferred []string
		want      string
	}{
		{
			name:      "nil info returns empty",
			info:      nil,
			preferred: []string{"/chat/completions"},
			want:      "",
		},
		{
			name:      "model supports first preferred",
			info:      &Info{ID: "gpt-4", SupportedEndpoints: []string{"/v1/chat/completions", "/v1/responses"}},
			preferred: []string{"/chat/completions", "/responses"},
			want:      "/chat/completions",
		},
		{
			name:      "model supports second preferred",
			info:      &Info{ID: "o3-mini", SupportedEndpoints: []string{"/v1/responses"}},
			preferred: []string{"/chat/completions", "/responses"},
			want:      "/responses",
		},
		{
			name:      "model supports neither",
			info:      &Info{ID: "embedding-model", SupportedEndpoints: []string{"/v1/embeddings"}},
			preferred: []string{"/chat/completions", "/responses"},
			want:      "",
		},
		{
			name:      "empty preferred list",
			info:      &Info{ID: "gpt-4", SupportedEndpoints: []string{"/v1/chat/completions"}},
			preferred: []string{},
			want:      "",
		},
		{
			name:      "normalizes /v1 prefix in preferred",
			info:      &Info{ID: "gpt-4", SupportedEndpoints: []string{"/v1/chat/completions"}},
			preferred: []string{"/v1/chat/completions"},
			want:      "/v1/chat/completions",
		},
		{
			name:      "normalizes no prefix in supported endpoints",
			info:      &Info{ID: "gpt-4", SupportedEndpoints: []string{"/chat/completions"}},
			preferred: []string{"/v1/chat/completions"},
			want:      "/v1/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PickEndpoint(tt.info, tt.preferred)
			if got != tt.want {
				t.Errorf("PickEndpoint() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSupportsEndpoint(t *testing.T) {
	tests := []struct {
		name     string
		info     *Info
		endpoint string
		want     bool
	}{
		{
			name:     "nil info",
			info:     nil,
			endpoint: "/chat/completions",
			want:     false,
		},
		{
			name:     "exact match with /v1 prefix",
			info:     &Info{SupportedEndpoints: []string{"/v1/chat/completions"}},
			endpoint: "/v1/chat/completions",
			want:     true,
		},
		{
			name:     "match without /v1 prefix",
			info:     &Info{SupportedEndpoints: []string{"/v1/chat/completions"}},
			endpoint: "/chat/completions",
			want:     true,
		},
		{
			name:     "no match",
			info:     &Info{SupportedEndpoints: []string{"/v1/responses"}},
			endpoint: "/chat/completions",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SupportsEndpoint(tt.info, tt.endpoint)
			if got != tt.want {
				t.Errorf("SupportsEndpoint() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNormalizeEndpoint(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"/v1/chat/completions", "/chat/completions"},
		{"/chat/completions", "/chat/completions"},
		{"chat/completions", "/chat/completions"},
		{"/v1/responses", "/responses"},
		{"", "/"},
		{"  /v1/chat/completions  ", "/chat/completions"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := normalizeEndpoint(tt.input)
			if got != tt.want {
				t.Errorf("normalizeEndpoint(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestSupportsLongContext(t *testing.T) {
	tests := []struct {
		name string
		info *Info
		want bool
	}{
		{
			name: "nil info",
			info: nil,
			want: false,
		},
		{
			name: "nil capabilities",
			info: &Info{ID: "gpt-5-mini"},
			want: false,
		},
		{
			name: "nil limits",
			info: &Info{ID: "gpt-5-mini", Capabilities: &Capabilities{}},
			want: false,
		},
		{
			name: "below threshold",
			info: &Info{ID: "gpt-5-mini", Capabilities: &Capabilities{
				Limits: &Limits{MaxContextWindowTokens: 200_000},
			}},
			want: false,
		},
		{
			name: "at threshold",
			info: &Info{ID: "gpt-5.4", Capabilities: &Capabilities{
				Limits: &Limits{MaxContextWindowTokens: 500_000},
			}},
			want: true,
		},
		{
			name: "above threshold (1M model)",
			info: &Info{ID: "gpt-5.5", Capabilities: &Capabilities{
				Limits: &Limits{MaxContextWindowTokens: 1_050_000},
			}},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := SupportsLongContext(tt.info)
			if got != tt.want {
				t.Errorf("SupportsLongContext() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCacheUsesModelAccessAndDeveloperCLIFallback(t *testing.T) {
	const billingNotice = `"Your billing plan has changed to usage-based billing and model multipliers no longer apply. Please update your client to the latest version to see the new billing information."`

	type requestHeaders struct {
		authorization string
		integrationID string
		intent        string
		interaction   string
		contentType   string
	}
	var requests []requestHeaders
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, requestHeaders{
			authorization: r.Header.Get("Authorization"),
			integrationID: r.Header.Get("Copilot-Integration-Id"),
			intent:        r.Header.Get("Openai-Intent"),
			interaction:   r.Header.Get("X-Interaction-Type"),
			contentType:   r.Header.Get("Content-Type"),
		})
		w.Header().Set("Content-Type", "application/json")
		if len(requests) == 1 {
			_, _ = w.Write([]byte(billingNotice))
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-5","supported_endpoints":["/responses"]}]}`))
	}))
	defer server.Close()

	provider := &modelCatalogTokenProvider{baseURL: server.URL}
	client := upstream.NewClient(provider, &http.Transport{}, false)
	cache := NewCache(client, time.Minute)

	raw, err := cache.GetRaw(context.Background())
	if err != nil {
		t.Fatalf("GetRaw() error = %v", err)
	}
	if len(requests) != 2 {
		t.Fatalf("expected primary request plus fallback, got %d requests", len(requests))
	}

	if got := requests[0]; got.authorization != "Bearer copilot-token" ||
		got.integrationID != "vscode-chat" || got.intent != "model-access" ||
		got.interaction != "model-access" || got.contentType != "" {
		t.Errorf("primary model-access headers = %+v", got)
	}
	if got := requests[1]; got.authorization != "Bearer github-token" ||
		got.integrationID != "copilot-developer-cli" || got.intent != "model-access" ||
		got.interaction != "model-access" || got.contentType != "" {
		t.Errorf("developer CLI fallback headers = %+v", got)
	}

	var response modelsListResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		t.Fatalf("cached response is not JSON: %v", err)
	}
	if len(response.Data) != 1 || response.Data[0].ID != "gpt-5" {
		t.Fatalf("cached models = %+v, want gpt-5", response.Data)
	}

	if _, err := cache.GetRaw(context.Background()); err != nil {
		t.Fatalf("cached GetRaw() error = %v", err)
	}
	if len(requests) != 2 {
		t.Fatalf("cached catalog triggered another upstream request: %d", len(requests))
	}
}

func TestCacheDoesNotFallbackForUnrelatedModelsFailure(t *testing.T) {
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"message":"temporary catalog failure"}`))
	}))
	defer server.Close()

	provider := &modelCatalogTokenProvider{baseURL: server.URL}
	client := upstream.NewClient(provider, &http.Transport{}, false)
	cache := NewCache(client, time.Minute)

	if _, err := cache.GetRaw(context.Background()); err == nil {
		t.Fatal("GetRaw() unexpectedly succeeded for unrelated catalog failure")
	}
	if requestCount != 1 {
		t.Fatalf("unrelated catalog failure triggered OAuth fallback: %d requests", requestCount)
	}
}

func TestCachePreservesDeveloperCLIFallbackError(t *testing.T) {
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		w.Header().Set("Content-Type", "application/json")
		if requestCount == 1 {
			_, _ = w.Write([]byte(billingMigrationNoticeJSON))
			return
		}
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"error":"developer CLI rejected"}`))
	}))
	defer server.Close()

	provider := &modelCatalogTokenProvider{baseURL: server.URL}
	client := upstream.NewClient(provider, &http.Transport{}, false)
	cache := NewCache(client, time.Minute)

	_, err := cache.GetRaw(context.Background())
	if err == nil {
		t.Fatal("GetRaw() unexpectedly succeeded after fallback failure")
	}
	var upstreamErr *upstream.UpstreamError
	if !errors.As(err, &upstreamErr) {
		t.Fatalf("GetRaw() error = %v, want upstream fallback error", err)
	}
	if upstreamErr.StatusCode != http.StatusForbidden {
		t.Fatalf("fallback status = %d, want %d", upstreamErr.StatusCode, http.StatusForbidden)
	}
}

const billingMigrationNoticeJSON = `"Your billing plan has changed to usage-based billing and model multipliers no longer apply. Please update your client to the latest version to see the new billing information."`

// modelCatalogTokenProvider models the two credentials available on auth.Client
// without exposing real tokens in tests.
type modelCatalogTokenProvider struct {
	baseURL string
}

func (p *modelCatalogTokenProvider) GetToken(context.Context) (string, error) {
	return "copilot-token", nil
}

func (p *modelCatalogTokenProvider) GetGitHubToken(context.Context) (string, error) {
	return "github-token", nil
}

func (p *modelCatalogTokenProvider) GetBaseURL() string {
	return p.baseURL
}
