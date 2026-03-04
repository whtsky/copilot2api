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
	CopilotUserAgent    = "GitHubCopilotChat/0.26.7"
	EditorVersion       = "vscode/1.96.2"
	EditorPluginVersion = "copilot-chat/0.26.7"
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
	req.Header.Set("X-Github-Api-Version", "2025-04-01")

	// Generate request ID if not present
	if req.Header.Get("X-Request-Id") == "" {
		req.Header.Set("X-Request-Id", GenerateRequestID())
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
