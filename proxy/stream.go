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
	"github.com/whtsky/copilot2api/internal/types"
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
			return err
		}
		if _, err := io.WriteString(w, "\n"); err != nil {
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
			slog.Debug("stream done", "endpoint", endpoint)
			if canFlush {
				flusher.Flush()
			}
			break
		}

		// Responses API termination events
		if isResponses && isResponsesTerminationEvent(line) {
			slog.Debug("stream termination event", "endpoint", endpoint, "event", strings.TrimSpace(line))
			// Write remaining data lines for this event, then stop
			for scanner.Scan() {
				remaining := scanner.Text()
				if _, err := io.WriteString(w, remaining); err != nil {
					return err
				}
				if _, err := io.WriteString(w, "\n"); err != nil {
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

// --- Streaming conversion functions ---

// streamResponsesAsChatChunks reads Responses API SSE events from body,
// converts each to Chat Completions chunks, and writes them to w.
// Used when the client requested /chat/completions but the model only
// supports /responses.
func streamResponsesAsChatChunks(w http.ResponseWriter, body io.ReadCloser) error {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	flusher, canFlush := w.(http.Flusher)
	state := NewResponsesStreamConvertState()

	var currentEventType string

	for scanner.Scan() {
		line := scanner.Text()

		// Parse SSE event type lines
		if strings.HasPrefix(line, "event: ") {
			currentEventType = strings.TrimPrefix(line, "event: ")
			continue
		}

		// Parse SSE data lines
		if strings.HasPrefix(line, "data: ") {
			dataStr := strings.TrimPrefix(line, "data: ")

			// The Responses API may send data: [DONE] before response.completed.
			// Ignore it — we rely on the converter's Finished state to know when
			// to emit the Chat Completions [DONE] sentinel.
			if dataStr == "[DONE]" {
				continue
			}

			var event types.ResponseStreamEvent
			if err := json.Unmarshal([]byte(dataStr), &event); err != nil {
				slog.Debug("skipping unparseable SSE data in conversion", "error", err)
				continue
			}

			// Use the event type from the "event:" line if the JSON doesn't
			// include it (some implementations put it only on the SSE line).
			if event.Type == "" && currentEventType != "" {
				event.Type = currentEventType
			}

			chunks := ConvertResponsesStreamEventToChatChunk(event, state)
			for _, chunk := range chunks {
				chunkData, err := json.Marshal(chunk)
				if err != nil {
					continue
				}
				if _, err := fmt.Fprintf(w, "data: %s\n\n", chunkData); err != nil {
					return err
				}
				if canFlush {
					flusher.Flush()
				}
			}

			// Check if the converter signaled stream completion
			if state.Finished {
				// Send [DONE] sentinel for Chat Completions format
				if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
					return err
				}
				if canFlush {
					flusher.Flush()
				}
				return nil
			}

			currentEventType = ""
			continue
		}

		// Blank lines are SSE event delimiters — reset event type
		if line == "" {
			currentEventType = ""
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// If we didn't see a termination event, return an error instead of
	// silently completing — the upstream stream ended unexpectedly.
	if !state.Finished {
		return fmt.Errorf("responses stream ended without terminal event")
	}

	return nil
}

// streamChatChunksAsResponsesEvents reads Chat Completions SSE chunks from body,
// converts each to Responses API events, and writes them to w.
// Used when the client requested /responses but the model only
// supports /chat/completions.
func streamChatChunksAsResponsesEvents(w http.ResponseWriter, body io.ReadCloser) error {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	flusher, canFlush := w.(http.Flusher)
	state := NewChatStreamConvertState()

	for scanner.Scan() {
		line := scanner.Text()

		// Only process data lines
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		dataStr := strings.TrimPrefix(line, "data: ")

		// [DONE] sentinel — stream is finished
		if dataStr == "[DONE]" {
			// If finish_reason was seen but termination event wasn't emitted
			// (no usage-only chunk arrived), emit it now without usage.
			if state.FinishSeen && !state.Finished {
				terminationEvent := state.buildTerminationEvent()
				if err := writeResponsesSSEEvent(w, terminationEvent); err != nil {
					return err
				}
				if canFlush {
					flusher.Flush()
				}
			}
			// If we haven't sent any termination event (edge case: no finish_reason seen)
			if !state.Finished {
				completedEvent := types.ResponseStreamEvent{
					Type: "response.completed",
					Response: &types.ResponsesResult{
						ID:     state.ID,
						Model:  state.Model,
						Status: "completed",
					},
				}
				if err := writeResponsesSSEEvent(w, completedEvent); err != nil {
					return err
				}
				if canFlush {
					flusher.Flush()
				}
			}
			return nil
		}

		var chunk types.OpenAIChatCompletionChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			slog.Debug("skipping unparseable SSE data in conversion", "error", err)
			continue
		}

		events := ConvertChatChunkToResponsesStreamEvents(chunk, state)
		for _, event := range events {
			if err := writeResponsesSSEEvent(w, event); err != nil {
				return err
			}
			if canFlush {
				flusher.Flush()
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	// If we reach EOF without seeing [DONE], the stream ended unexpectedly.
	if !state.Finished {
		return fmt.Errorf("chat completions stream ended without [DONE] sentinel")
	}

	return nil
}

// writeResponsesSSEEvent writes a single Responses API SSE event to w.
func writeResponsesSSEEvent(w io.Writer, event types.ResponseStreamEvent) error {
	data, err := json.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, data); err != nil {
		return err
	}
	return nil
}
