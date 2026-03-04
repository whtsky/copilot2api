package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/whtsky/copilot2api/auth"
	"github.com/whtsky/copilot2api/internal/cache"
	"github.com/whtsky/copilot2api/internal/upstream"
)

type Handler struct {
	upstream    *upstream.Client
	authClient  *auth.Client
	modelsCache *cache.Cache[[]byte]
}

// NewHandler creates a new proxy handler.
// The transport is used for upstream HTTP requests (pass nil to create a new one).
func NewHandler(authClient *auth.Client, transport *http.Transport) *Handler {
	return &Handler{
		upstream:    upstream.NewClient(authClient, transport),
		authClient:  authClient,
		modelsCache: cache.New[[]byte](5 * time.Minute),
	}
}

// WarmModels pre-populates the models cache to avoid cold-cache latency.
func (h *Handler) WarmModels(ctx context.Context) {
	_, err := h.modelsCache.Get(ctx, func(ctx context.Context) ([]byte, error) {
		return h.doModelsRequest(ctx)
	})
	if err != nil {
		slog.Warn("failed to warm models cache", "error", err)
	}
}

// doModelsRequest fetches /models from upstream.
func (h *Handler) doModelsRequest(ctx context.Context) ([]byte, error) {
	_, respData, err := h.upstream.Do(ctx, upstream.Request{
		Method:   "GET",
		Endpoint: "/models",
	})
	return respData, err
}

// ServeHTTP handles all proxy requests
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Extract endpoint from path
	endpoint := strings.TrimPrefix(r.URL.Path, "/v1")

	slog.Debug("proxy request", "method", r.Method, "path", r.URL.Path, "endpoint", endpoint)
	switch endpoint {
	case "/models":
		h.handleModels(w, r)
	case "/embeddings":
		h.handleEmbeddings(w, r)
	case "/chat/completions":
		h.handlePassthrough(w, r, endpoint)
	case "/responses":
		h.handlePassthrough(w, r, endpoint)
	default:
		WriteOpenAIError(w, http.StatusNotFound, OpenAIErrorTypeInvalidRequest, "Endpoint not found")
	}
}

// handleModels handles /v1/models with caching
func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		WriteOpenAIError(w, http.StatusMethodNotAllowed, OpenAIErrorTypeInvalidRequest, "Method not allowed")
		return
	}

	respData, err := h.modelsCache.Get(r.Context(), func(ctx context.Context) ([]byte, error) {
		return h.doModelsRequest(ctx)
	})

	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		slog.Error("failed to fetch models", "error", err)
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Internal server error")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(respData)
}

// handlePassthrough handles direct passthrough requests
func (h *Handler) handlePassthrough(w http.ResponseWriter, r *http.Request, endpoint string) {
	// Check body size before processing — reject oversized payloads with 413
	var bodyBytes []byte
	if r.Body != nil {
		const maxBodySize = 10 << 20 // 10MB
		var err error
		bodyBytes, err = io.ReadAll(io.LimitReader(r.Body, maxBodySize+1))
		if err != nil {
			WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Failed to read request body")
			return
		}
		if len(bodyBytes) > maxBodySize {
			WriteOpenAIError(w, http.StatusRequestEntityTooLarge, OpenAIErrorTypeInvalidRequest, "Request body too large")
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
	}

	h.handlePassthroughBody(w, r, endpoint, bodyBytes)
}

// handlePassthroughBody processes the passthrough request after the body has been read and validated.
// It takes pre-read body bytes to avoid redundant body reading.
func (h *Handler) handlePassthroughBody(w http.ResponseWriter, r *http.Request, endpoint string, bodyBytes []byte) {
	// Check if this is a streaming request
	if isStreamingRequest(bodyBytes) {
		if err := h.HandleStreamingRequest(w, r, endpoint); err != nil {
			slog.Error("streaming request failed", "endpoint", endpoint, "error", err)
			// If headers were already sent we can't write an HTTP error.
			// Otherwise, send a proper 502 so the client doesn't get an empty 200.
			var hse *headersSentError
			if !errors.As(err, &hse) {
				WriteOpenAIError(w, http.StatusBadGateway, OpenAIErrorTypeServerError, "upstream request failed")
			}
		}
		return
	}

	// Handle non-streaming request
	respData, err := h.doNonStreamingRequest(r, endpoint)
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			// Forward upstream status code and body
			upstreamErr.WriteRawError(w)
			return
		}
		slog.Error("passthrough request failed", "endpoint", endpoint, "error", err)
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Internal server error")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(respData)
}

// doNonStreamingRequest makes a non-streaming request to the Copilot API via the shared upstream client.
func (h *Handler) doNonStreamingRequest(r *http.Request, endpoint string) ([]byte, error) {
	var body interface{}
	if r.Body != nil {
		body = r.Body
	}

	_, respData, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     endpoint,
		Body:         body,
		QueryString:  r.URL.RawQuery,
		ExtraHeaders: collectForwardHeaders(r),
	})
	return respData, err
}

// collectForwardHeaders returns headers from the original request that should be
// forwarded to the upstream API.
func collectForwardHeaders(r *http.Request) map[string]string {
	headers := make(map[string]string)
	for _, name := range []string{"Content-Type", "Accept", "Cache-Control"} {
		if v := r.Header.Get(name); v != "" {
			headers[name] = v
		}
	}
	return headers
}

// handleUsage returns usage/quota info from the Copilot token response
func (h *Handler) HandleUsage(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		WriteOpenAIError(w, http.StatusMethodNotAllowed, OpenAIErrorTypeInvalidRequest, "Method not allowed")
		return
	}

	usage, err := h.authClient.GetUsageInfo(r.Context())
	if err != nil {
		slog.Error("failed to get usage info", "error", err)
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to get usage info")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(usage)
}

// handleEmbeddings normalizes input to array format before proxying
func (h *Handler) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 10<<20) // 10MB limit
	body, err := io.ReadAll(r.Body)
	if err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Failed to read request body")
		return
	}

	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Invalid JSON")
		return
	}

	// If input is a string, wrap it in an array
	if input, ok := req["input"]; ok {
		var s string
		if json.Unmarshal(input, &s) == nil {
			wrapped, _ := json.Marshal([]string{s})
			req["input"] = wrapped
			body, _ = json.Marshal(req)
		}
	}

	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	h.handlePassthroughBody(w, r, "/embeddings", body)
}

// --- OpenAI error response helpers ---

// OpenAIErrorResponse represents an error response in OpenAI API format
type OpenAIErrorResponse struct {
	Error OpenAIError `json:"error"`
}

// OpenAIError represents the error object in OpenAI API responses
type OpenAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// Error type constants for OpenAI API
const (
	OpenAIErrorTypeServerError    = "server_error"
	OpenAIErrorTypeInvalidRequest = "invalid_request_error"
)

// WriteOpenAIError writes an error response in OpenAI API format
func WriteOpenAIError(w http.ResponseWriter, statusCode int, errorType, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := OpenAIErrorResponse{
		Error: OpenAIError{
			Message: message,
			Type:    errorType,
		},
	}

	json.NewEncoder(w).Encode(errorResp)
}
