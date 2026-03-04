package auth

import (
	"testing"
	"time"
)

func TestExtractBaseURLFromToken(t *testing.T) {
	tests := []struct {
		name     string
		token    string
		expected string
	}{
		{
			name:     "valid token with proxy-ep",
			token:    "tid=123;exp=456;sku=copilot;proxy-ep=proxy.individual.githubcopilot.com;other=value",
			expected: "https://api.individual.githubcopilot.com",
		},
		{
			name:     "token with different proxy-ep",
			token:    "tid=abc;proxy-ep=proxy.enterprise.githubcopilot.com;exp=789",
			expected: "https://api.enterprise.githubcopilot.com",
		},
		{
			name:     "token without proxy-ep",
			token:    "tid=123;exp=456;sku=copilot;other=value",
			expected: DefaultBaseURL,
		},
		{
			name:     "empty token",
			token:    "",
			expected: DefaultBaseURL,
		},
		{
			name:     "malformed token",
			token:    "invalid-token-format",
			expected: DefaultBaseURL,
		},
		{
			name:     "proxy-ep without proxy prefix",
			token:    "tid=123;proxy-ep=individual.githubcopilot.com;exp=456",
			expected: DefaultBaseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ExtractBaseURLFromToken(tt.token)
			if result != tt.expected {
				t.Errorf("ExtractBaseURLFromToken(%q) = %q, want %q", tt.token, result, tt.expected)
			}
		})
	}
}

func TestCopilotToken_IsTokenUsable(t *testing.T) {
	tests := []struct {
		name     string
		token    *CopilotToken
		expected bool
	}{
		{
			name:     "nil token",
			token:    nil,
			expected: false,
		},
		{
			name: "empty token string",
			token: &CopilotToken{
				Token:     "",
				ExpiresAt: time.Now().Add(10 * time.Minute),
			},
			expected: false,
		},
		{
			name: "token expiring in 10 minutes (usable)",
			token: &CopilotToken{
				Token:     "valid-token",
				ExpiresAt: time.Now().Add(10 * time.Minute),
			},
			expected: true,
		},
		{
			name: "token expiring in 3 minutes (not usable)",
			token: &CopilotToken{
				Token:     "valid-token",
				ExpiresAt: time.Now().Add(3 * time.Minute),
			},
			expected: false,
		},
		{
			name: "expired token",
			token: &CopilotToken{
				Token:     "valid-token",
				ExpiresAt: time.Now().Add(-1 * time.Minute),
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.token.IsTokenUsable()
			if result != tt.expected {
				t.Errorf("IsTokenUsable() = %v, want %v", result, tt.expected)
			}
		})
	}
}