package copilot

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// Exported constants for User-Agent and version headers.
const (
	CopilotUserAgent    = "GitHubCopilotChat/0.58.0"
	EditorVersion       = "vscode/1.120.0"
	EditorPluginVersion = "copilot-chat/0.58.0"
	CopilotAPIVersion   = "2026-06-01"
)

// AddHeaders adds required Copilot headers to the request
func AddHeaders(req *http.Request, token string) {
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("User-Agent", CopilotUserAgent)
	req.Header.Set("Editor-Version", EditorVersion)
	req.Header.Set("Editor-Plugin-Version", EditorPluginVersion)
	req.Header.Set("Copilot-Integration-Id", "vscode-chat")
	req.Header.Set("Openai-Intent", "conversation-agent")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Github-Api-Version", CopilotAPIVersion)

	// Generate request ID if not present
	if req.Header.Get("X-Request-Id") == "" {
		req.Header.Set("X-Request-Id", GenerateRequestID())
	}
}

// AddModelAccessHeaders adds the headers used by Copilot's model catalog
// endpoint. The catalog is a metadata request, not a conversation request.
func AddModelAccessHeaders(req *http.Request, token string, integrationID string) {
	AddHeaders(req, token)
	req.Header.Set("Openai-Intent", "model-access")
	req.Header.Set("X-Interaction-Type", "model-access")
	req.Header.Del("Content-Type")
	if integrationID != "" {
		req.Header.Set("Copilot-Integration-Id", integrationID)
	}
}

// GenerateRequestID generates a unique request ID using crypto/rand
func GenerateRequestID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		slog.Error("crypto/rand.Read failed", "error", err)
		return fmt.Sprintf("req_fallback_%d", time.Now().UnixNano())
	}
	return "req_" + hex.EncodeToString(b)
}
