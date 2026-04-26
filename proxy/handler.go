package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/whtsky/copilot2api/auth"
	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/sse"
	"github.com/whtsky/copilot2api/internal/types"
	"github.com/whtsky/copilot2api/internal/upstream"
)

type Handler struct {
	upstream    *upstream.Client
	authClient  *auth.Client
	modelsCache *models.Cache
}

// NewHandler creates a new proxy handler.
// The transport is used for upstream HTTP requests (pass nil to create a new one).
func NewHandler(authClient *auth.Client, transport *http.Transport, mc *models.Cache, debug bool) *Handler {
	return &Handler{
		upstream:    upstream.NewClient(authClient, transport, debug),
		authClient:  authClient,
		modelsCache: mc,
	}
}

// ServeHTTP handles all proxy requests
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Extract endpoint from path
	endpoint := strings.TrimPrefix(r.URL.Path, "/v1")
	defer func() {
		slog.Info("proxy request", "method", r.Method, "endpoint", endpoint, "duration_ms", time.Since(start).Milliseconds())
	}()

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

	respData, err := h.modelsCache.GetRaw(r.Context())
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("failed to fetch models", err)
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
		var err error
		bodyBytes, err = io.ReadAll(io.LimitReader(r.Body, upstream.MaxRequestBody+1))
		if err != nil {
			WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Failed to read request body")
			return
		}
		if len(bodyBytes) > upstream.MaxRequestBody {
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
	// Determine smart routing: should we convert to a different endpoint?
	targetEndpoint := h.resolveTargetEndpoint(r, endpoint, bodyBytes)

	if targetEndpoint != endpoint {
		slog.Info("smart routing: converting request", "from", endpoint, "to", targetEndpoint)
	}

	streaming := isStreamingRequest(bodyBytes)

	// If no conversion needed, passthrough as before
	if targetEndpoint == endpoint {
		if streaming {
			if err := h.HandleStreamingRequest(w, r, endpoint); err != nil {
				upstream.LogRequestError("streaming request failed", err, "endpoint", endpoint)
				var hse *headersSentError
				if !errors.As(err, &hse) {
					WriteOpenAIError(w, http.StatusBadGateway, OpenAIErrorTypeServerError, "upstream request failed")
				}
			}
			return
		}

		respData, err := h.doNonStreamingRequest(r, endpoint)
		if err != nil {
			var upstreamErr *upstream.UpstreamError
			if errors.As(err, &upstreamErr) {
				upstreamErr.WriteRawError(w)
				return
			}
			upstream.LogRequestError("passthrough request failed", err, "endpoint", endpoint)
			WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Internal server error")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Write(respData)
		return
	}

	// Conversion needed: route based on direction
	switch {
	case endpoint == "/chat/completions" && targetEndpoint == "/responses":
		if streaming {
			h.handleChatToResponsesStreaming(w, r, bodyBytes)
		} else {
			h.handleChatToResponsesNonStreaming(w, r, bodyBytes)
		}
	case endpoint == "/responses" && targetEndpoint == "/chat/completions":
		if streaming {
			h.handleResponsesToChatStreaming(w, r, bodyBytes)
		} else {
			h.handleResponsesToChatNonStreaming(w, r, bodyBytes)
		}
	}
}

// resolveTargetEndpoint determines which upstream endpoint to use based on
// model capabilities. Returns the original endpoint when no conversion is
// needed (model supports it, model is unknown, or capabilities are unavailable).
func (h *Handler) resolveTargetEndpoint(r *http.Request, endpoint string, bodyBytes []byte) string {
	if h.modelsCache == nil {
		return endpoint
	}

	modelID := extractModelField(bodyBytes)
	if modelID == "" {
		return endpoint // no model field → passthrough
	}

	modelMap, err := h.modelsCache.GetInfo(r.Context())
	if err != nil {
		slog.Debug("smart routing: failed to get model info, falling back to passthrough", "error", err)
		return endpoint
	}

	info := modelMap[modelID]
	// Unknown model → passthrough (best effort)
	if info == nil {
		return endpoint
	}

	// Build preference order: requested endpoint first, then the alternative
	var preferred []string
	switch endpoint {
	case "/chat/completions":
		preferred = []string{"/chat/completions", "/responses"}
	case "/responses":
		preferred = []string{"/responses", "/chat/completions"}
	default:
		return endpoint
	}

	target := models.PickEndpoint(info, preferred)
	if target == "" {
		// Model supports neither → passthrough, let upstream return the error
		return endpoint
	}
	return target
}

// extractModelField extracts the "model" field from a JSON request body.
func extractModelField(body []byte) string {
	var top struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &top); err != nil {
		return ""
	}
	return top.Model
}

// --- Non-streaming conversion handlers ---

// handleChatToResponsesNonStreaming converts a Chat Completions request to
// Responses API, sends it upstream, and converts the response back.
func (h *Handler) handleChatToResponsesNonStreaming(w http.ResponseWriter, r *http.Request, bodyBytes []byte) {
	var chatReq types.OpenAIChatCompletionsRequest
	if err := json.Unmarshal(bodyBytes, &chatReq); err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Invalid JSON in request body")
		return
	}

	responsesReq := ConvertChatToResponsesRequest(chatReq)
	reqBody, err := json.Marshal(responsesReq)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal converted request")
		return
	}

	_, respData, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     "/responses",
		Body:         reqBody,
		QueryString:  r.URL.RawQuery,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("converted request failed", err, "from", "/chat/completions", "to", "/responses")
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Internal server error")
		return
	}

	var responsesResult types.ResponsesResult
	if err := json.Unmarshal(respData, &responsesResult); err != nil {
		slog.Error("failed to parse responses result for conversion", "error", err)
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to parse upstream response")
		return
	}

	// Check for failed response
	if responsesResult.Status == "failed" || responsesResult.Error != nil {
		msg := "Upstream request failed"
		if responsesResult.Error != nil && responsesResult.Error.Message != "" {
			msg = responsesResult.Error.Message
		}
		WriteOpenAIError(w, http.StatusBadGateway, OpenAIErrorTypeServerError, msg)
		return
	}

	chatResp := ConvertResponsesResultToChatResponse(responsesResult, chatReq.Model)
	result, err := json.Marshal(chatResp)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal response")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(result)
}

// handleResponsesToChatNonStreaming converts a Responses API request to
// Chat Completions, sends it upstream, and converts the response back.
func (h *Handler) handleResponsesToChatNonStreaming(w http.ResponseWriter, r *http.Request, bodyBytes []byte) {
	var responsesReq types.ResponsesRequest
	if err := json.Unmarshal(bodyBytes, &responsesReq); err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Invalid JSON in request body")
		return
	}

	chatReq := ConvertResponsesToChatRequest(responsesReq)
	reqBody, err := json.Marshal(chatReq)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal converted request")
		return
	}

	_, respData, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     "/chat/completions",
		Body:         reqBody,
		QueryString:  r.URL.RawQuery,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("converted request failed", err, "from", "/responses", "to", "/chat/completions")
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Internal server error")
		return
	}

	var chatResp types.OpenAIChatCompletionsResponse
	if err := json.Unmarshal(respData, &chatResp); err != nil {
		slog.Error("failed to parse chat response for conversion", "error", err)
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to parse upstream response")
		return
	}

	responsesResult := ConvertChatResponseToResponsesResult(chatResp)
	result, err := json.Marshal(responsesResult)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal response")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(result)
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
	r.Body = http.MaxBytesReader(w, r.Body, upstream.MaxRequestBody) // 10MB limit
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

// --- Streaming conversion handlers ---

// handleChatToResponsesStreaming sends a converted Chat→Responses request upstream
// and converts the Responses stream events back to Chat Completions chunks.
func (h *Handler) handleChatToResponsesStreaming(w http.ResponseWriter, r *http.Request, bodyBytes []byte) {
	var chatReq types.OpenAIChatCompletionsRequest
	if err := json.Unmarshal(bodyBytes, &chatReq); err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Invalid JSON in request body")
		return
	}

	responsesReq := ConvertChatToResponsesRequest(chatReq)
	reqBody, err := json.Marshal(responsesReq)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal converted request")
		return
	}

	resp, _, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     "/responses",
		Body:         reqBody,
		QueryString:  r.URL.RawQuery,
		Stream:       true,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("converted streaming request failed", err, "from", "/chat/completions", "to", "/responses")
		WriteOpenAIError(w, http.StatusBadGateway, OpenAIErrorTypeServerError, "upstream request failed")
		return
	}
	defer resp.Body.Close()

	sse.BeginSSE(w)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	if err := streamResponsesAsChatChunks(w, resp.Body); err != nil {
		slog.Error("streaming conversion failed (responses→chat)", "error", err)
		// Headers already sent, can't write HTTP error
	}
}

// handleResponsesToChatStreaming sends a converted Responses→Chat request upstream
// and converts the Chat Completions stream chunks back to Responses API events.
func (h *Handler) handleResponsesToChatStreaming(w http.ResponseWriter, r *http.Request, bodyBytes []byte) {
	var responsesReq types.ResponsesRequest
	if err := json.Unmarshal(bodyBytes, &responsesReq); err != nil {
		WriteOpenAIError(w, http.StatusBadRequest, OpenAIErrorTypeInvalidRequest, "Invalid JSON in request body")
		return
	}

	chatReq := ConvertResponsesToChatRequest(responsesReq)
	reqBody, err := json.Marshal(chatReq)
	if err != nil {
		WriteOpenAIError(w, http.StatusInternalServerError, OpenAIErrorTypeServerError, "Failed to marshal converted request")
		return
	}

	resp, _, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       r.Method,
		Endpoint:     "/chat/completions",
		Body:         reqBody,
		QueryString:  r.URL.RawQuery,
		Stream:       true,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("converted streaming request failed", err, "from", "/responses", "to", "/chat/completions")
		WriteOpenAIError(w, http.StatusBadGateway, OpenAIErrorTypeServerError, "upstream request failed")
		return
	}
	defer resp.Body.Close()

	sse.BeginSSE(w)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	if err := streamChatChunksAsResponsesEvents(w, resp.Body); err != nil {
		slog.Error("streaming conversion failed (chat→responses)", "error", err)
		// Headers already sent, can't write HTTP error
	}
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
