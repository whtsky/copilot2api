package proxy

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/whtsky/copilot2api/internal/sse"
	"github.com/whtsky/copilot2api/internal/upstream"
)

// headersSentError wraps an error that occurred after response headers were
// already written to the client. The caller must not attempt to write an HTTP
// error response because the status line has already been flushed.
type headersSentError struct{ err error }

func (e *headersSentError) Error() string { return e.err.Error() }
func (e *headersSentError) Unwrap() error { return e.err }

// HandleStreamingRequest handles streaming requests to Copilot API
func (h *Handler) HandleStreamingRequest(w http.ResponseWriter, r *http.Request, endpoint string) error {
	var body interface{}
	if r.Body != nil {
		body = r.Body
	}

	resp, _, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     endpoint,
		Body:         body,
		QueryString:  r.URL.RawQuery,
		Stream:       true,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return nil
		}
		return fmt.Errorf("upstream request failed: %w", err)
	}
	defer resp.Body.Close()

	// Set up streaming response headers
	sse.BeginSSE(w)

	// Flush headers
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	// Stream the response — headers are already sent at this point.
	if err := h.streamResponse(w, resp.Body, endpoint); err != nil {
		return &headersSentError{err: err}
	}
	return nil
}

// streamResponse streams the SSE response line by line.
// endpoint is used to select the correct termination strategy.
func (h *Handler) streamResponse(w http.ResponseWriter, body io.ReadCloser, endpoint string) error {
	scanner := bufio.NewScanner(body)

	// Increase buffer size to handle large SSE lines (default is 64KB, increase to 1MB)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	flusher, canFlush := w.(http.Flusher)
	isResponses := endpoint == "/responses"

	for scanner.Scan() {
		line := scanner.Text()

		// Write the line
		if _, err := io.WriteString(w, line); err != nil {
			slog.Error("failed to write response line", "error", err)
			return err
		}
		if _, err := io.WriteString(w, "\n"); err != nil {
			slog.Error("failed to write response line", "error", err)
			return err
		}

		// SSE events are delimited by blank lines; flush at event boundaries
		if canFlush && line == "" {
			flusher.Flush()
		}

		// Chat Completions: terminate on data: [DONE]
		// Skip this check for the Responses API — it uses event-based
		// termination and may send data: [DONE] before response.completed.
		if !isResponses && strings.TrimSpace(line) == "data: [DONE]" {
			slog.Debug("chat completions stream done")
			if canFlush {
				flusher.Flush()
			}
			break
		}

		// Responses API termination events
		if isResponses && isResponsesTerminationEvent(line) {
			slog.Debug("responses stream termination", "event", line)
			// Write remaining data lines for this event, then stop
			for scanner.Scan() {
				remaining := scanner.Text()
				if _, err := io.WriteString(w, remaining); err != nil {
					slog.Error("failed to write response line", "error", err)
					return err
				}
				if _, err := io.WriteString(w, "\n"); err != nil {
					slog.Error("failed to write response line", "error", err)
					return err
				}
				if remaining == "" {
					break
				}
			}
			if canFlush {
				flusher.Flush()
			}
			break
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("error reading upstream response", "error", err)
		return err
	}

	return nil
}

// isResponsesTerminationEvent checks if an SSE event line indicates
// stream termination for the Responses API.
func isResponsesTerminationEvent(line string) bool {
	line = strings.TrimSpace(line)
	return line == "event: response.completed" ||
		line == "event: response.incomplete" ||
		line == "event: response.failed" ||
		line == "event: error"
}

// isStreamingRequest checks if the request body wants streaming.
func isStreamingRequest(body []byte) bool {
	var top struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &top); err != nil {
		return false
	}
	return top.Stream
}
