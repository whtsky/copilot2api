package upstream

import (
	"fmt"
	"net/http"
)

// UpstreamError represents an error from the upstream API.
type UpstreamError struct {
	StatusCode int
	Body       []byte
}

func (e *UpstreamError) Error() string {
	body := e.Body
	if len(body) > 512 {
		body = body[:512]
	}
	return fmt.Sprintf("upstream error: status %d, body: %s", e.StatusCode, string(body))
}

// WriteRawError writes the upstream error's status code and body directly to
// w as a JSON response. This is the common "forward upstream error as-is"
// pattern used by both the proxy and anthropic handlers.
func (e *UpstreamError) WriteRawError(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.StatusCode)
	_, _ = w.Write(e.Body)
}
