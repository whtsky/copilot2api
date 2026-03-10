package upstream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
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

// LogRequestError logs an upstream request failure at the appropriate level.
// Client disconnects (context.Canceled) and timeouts (context.DeadlineExceeded)
// are logged at Warn since they are normal operational events, not server errors.
func LogRequestError(msg string, err error, extraArgs ...any) {
	args := append([]any{"error", err}, extraArgs...)
	if isContextError(err) {
		slog.Warn(msg, args...)
		return
	}
	slog.Error(msg, args...)
}

func isContextError(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}
