package models

import "testing"

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
