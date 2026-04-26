package gemini

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/sse"
	"github.com/whtsky/copilot2api/internal/types"
	"github.com/whtsky/copilot2api/internal/upstream"
	"github.com/whtsky/copilot2api/proxy"
)

type Handler struct {
	upstream *upstream.Client
	models   *models.Cache
}

func NewHandler(authClient upstream.TokenProvider, transport *http.Transport, mc *models.Cache, debug bool) *Handler {
	return &Handler{upstream: upstream.NewClient(authClient, transport, debug), models: mc}
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	defer func() {
		slog.Info("gemini request", "method", r.Method, "path", r.URL.Path, "duration_ms", time.Since(start).Milliseconds())
	}()

	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/v1beta/models":
		h.handleModels(w, r)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1beta/models/") && strings.HasSuffix(r.URL.Path, ":generateContent"):
		h.handleGenerateContent(w, r, false)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1beta/models/") && strings.HasSuffix(r.URL.Path, ":streamGenerateContent"):
		h.handleGenerateContent(w, r, true)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/v1beta/models/") && strings.HasSuffix(r.URL.Path, ":countTokens"):
		h.handleCountTokens(w, r)
	default:
		writeError(w, http.StatusNotFound, "NOT_FOUND", "Endpoint not found")
	}
}

func (h *Handler) handleModels(w http.ResponseWriter, r *http.Request) {
	info, err := h.models.GetInfo(r.Context())
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("failed to fetch models for gemini list", err)
		writeError(w, http.StatusInternalServerError, "INTERNAL", "Failed to load models")
		return
	}

	resp := ModelListResponse{Models: make([]Model, 0, len(info))}
	for _, modelInfo := range info {
		resp.Models = append(resp.Models, Model{
			Name:                       "models/" + modelInfo.ID,
			Version:                    modelInfo.ID,
			DisplayName:                modelInfo.ID,
			Description:                "copilot2api compatible model",
			SupportedGenerationMethods: supportedGenerationMethods(modelInfo),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) handleGenerateContent(w http.ResponseWriter, r *http.Request, stream bool) {
	modelID, ok := modelIDFromPath(r.URL.Path)
	if !ok {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", "Invalid model path")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, upstream.MaxRequestBody)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", fmt.Sprintf("Invalid request body: %v", err))
		return
	}

	var req GenerateContentRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", fmt.Sprintf("Invalid JSON: %v", err))
		return
	}

	openAIReq, err := toOpenAIChatRequest(modelID, req, stream)
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", err.Error())
		return
	}

	targetEndpoint := "/chat/completions"
	if h.models != nil {
		if modelMap, err := h.models.GetInfo(r.Context()); err == nil {
			if info := modelMap[modelID]; info != nil {
				if endpoint := models.PickEndpoint(info, []string{"/chat/completions", "/responses"}); endpoint != "" {
					targetEndpoint = endpoint
				}
			}
		}
	}

	if stream {
		h.handleStreamingGenerateContent(w, r, openAIReq, targetEndpoint)
		return
	}

	resp, err := h.doNonStreamingGenerateContent(r, openAIReq, targetEndpoint)
	if err != nil {
		var upstreamErr *upstream.UpstreamError
		if errors.As(err, &upstreamErr) {
			upstreamErr.WriteRawError(w)
			return
		}
		upstream.LogRequestError("gemini generateContent failed", err, "endpoint", targetEndpoint)
		writeError(w, http.StatusBadGateway, "INTERNAL", "Upstream request failed")
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (h *Handler) doNonStreamingGenerateContent(r *http.Request, openAIReq types.OpenAIChatCompletionsRequest, targetEndpoint string) (*GenerateContentResponse, error) {
	if targetEndpoint == "/responses" {
		responsesReq := proxy.ConvertChatToResponsesRequest(openAIReq)
		_, data, err := h.upstream.Do(r.Context(), upstream.Request{
			Method:       http.MethodPost,
			Endpoint:     "/responses",
			Body:         responsesReq,
			QueryString:  r.URL.RawQuery,
			ExtraHeaders: collectForwardHeaders(r),
		})
		if err != nil {
			return nil, err
		}
		var result types.ResponsesResult
		if err := json.Unmarshal(data, &result); err != nil {
			return nil, fmt.Errorf("failed to parse responses result: %w", err)
		}
		chatResp := proxy.ConvertResponsesResultToChatResponse(result, openAIReq.Model)
		resp := fromOpenAIChatResponse(chatResp)
		return &resp, nil
	}

	_, data, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       http.MethodPost,
		Endpoint:     "/chat/completions",
		Body:         openAIReq,
		QueryString:  r.URL.RawQuery,
		ExtraHeaders: collectForwardHeaders(r),
	})
	if err != nil {
		return nil, err
	}

	var chatResp types.OpenAIChatCompletionsResponse
	if err := json.Unmarshal(data, &chatResp); err != nil {
		return nil, fmt.Errorf("failed to parse chat completion: %w", err)
	}
	resp := fromOpenAIChatResponse(chatResp)
	return &resp, nil
}

func (h *Handler) handleStreamingGenerateContent(w http.ResponseWriter, r *http.Request, openAIReq types.OpenAIChatCompletionsRequest, targetEndpoint string) {
	resp, _, err := h.upstream.Do(r.Context(), upstream.Request{
		Method:       http.MethodPost,
		Endpoint:     targetEndpoint,
		Body:         openAIReq,
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
		writeError(w, http.StatusBadGateway, "INTERNAL", "Upstream request failed")
		return
	}
	defer resp.Body.Close()

	sse.BeginSSE(w)
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}

	if err := streamChatAsGemini(w, resp.Body); err != nil {
		upstream.LogRequestError("gemini chat stream failed", err)
	}
}

func (h *Handler) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, upstream.MaxRequestBody)
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", fmt.Sprintf("Invalid request body: %v", err))
		return
	}

	var req GenerateContentRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_ARGUMENT", fmt.Sprintf("Invalid JSON: %v", err))
		return
	}

	total := estimateTokens(req)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(CountTokensResponse{TotalTokens: total, PromptTokens: total})
}

func modelIDFromPath(p string) (string, bool) {
	trimmed := strings.TrimPrefix(p, "/v1beta/models/")
	if trimmed == p || trimmed == "" {
		return "", false
	}
	base := trimmed
	if i := strings.Index(base, ":"); i >= 0 {
		base = base[:i]
	}
	base = path.Clean("/" + base)
	base = strings.TrimPrefix(base, "/")
	base = strings.TrimPrefix(base, "models/")
	if base == "" || base == "." {
		return "", false
	}
	return base, true
}

func supportedGenerationMethods(info *models.Info) []string {
	if info == nil {
		return nil
	}
	return []string{"generateContent", "streamGenerateContent", "countTokens"}
}

func toOpenAIChatRequest(model string, req GenerateContentRequest, stream bool) (types.OpenAIChatCompletionsRequest, error) {
	openAIReq := types.OpenAIChatCompletionsRequest{Model: model, Stream: stream}
	if stream {
		openAIReq.StreamOptions = &types.OpenAIStreamOptions{IncludeUsage: true}
	}

	config := req.GenerationConfig
	if config == nil {
		config = req.Config
	}
	if config != nil {
		openAIReq.Temperature = config.Temperature
		openAIReq.TopP = config.TopP
		openAIReq.MaxTokens = config.MaxOutputTokens
	}

	if req.SystemInstruction != nil {
		text := flattenParts(req.SystemInstruction.Parts)
		if text != "" {
			openAIReq.Messages = append(openAIReq.Messages, types.OpenAIMessage{Role: "system", Content: &types.OpenAIContent{Text: &text}})
		}
	}

	for _, content := range req.Contents {
		msgs, err := contentToMessages(content)
		if err != nil {
			return openAIReq, err
		}
		openAIReq.Messages = append(openAIReq.Messages, msgs...)
	}

	for _, tool := range req.Tools {
		for _, decl := range tool.FunctionDeclarations {
			openAIReq.Tools = append(openAIReq.Tools, types.OpenAIToolDefinition{
				Type:     "function",
				Function: types.OpenAIFunctionSpec{Name: decl.Name, Description: decl.Description, Parameters: decl.Parameters},
			})
		}
	}

	if req.ToolConfig != nil && req.ToolConfig.FunctionCallingConfig != nil {
		switch strings.ToUpper(req.ToolConfig.FunctionCallingConfig.Mode) {
		case "NONE":
			openAIReq.ToolChoice = "none"
		case "ANY":
			if len(req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames) == 1 {
				openAIReq.ToolChoice = map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": req.ToolConfig.FunctionCallingConfig.AllowedFunctionNames[0]}}
			} else {
				openAIReq.ToolChoice = "required"
			}
		default:
			openAIReq.ToolChoice = "auto"
		}
	}

	if len(openAIReq.Messages) == 0 {
		return openAIReq, fmt.Errorf("contents or systemInstruction is required")
	}
	return openAIReq, nil
}

func contentToMessages(content Content) ([]types.OpenAIMessage, error) {
	role := content.Role
	if role == "" {
		role = "user"
	}
	switch role {
	case "user":
		return userContentToMessages(content)
	case "model":
		return modelContentToMessages(content)
	default:
		return nil, fmt.Errorf("unsupported Gemini role: %s", role)
	}
}

func userContentToMessages(content Content) ([]types.OpenAIMessage, error) {
	var messages []types.OpenAIMessage
	var textParts []string
	flush := func() {
		if len(textParts) == 0 {
			return
		}
		text := strings.Join(textParts, "\n\n")
		messages = append(messages, types.OpenAIMessage{Role: "user", Content: &types.OpenAIContent{Text: &text}})
		textParts = nil
	}
	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
		if part.FunctionResponse != nil {
			flush()
			payload := part.FunctionResponse.Response
			if payload == nil {
				payload = map[string]interface{}{}
			}
			data, err := json.Marshal(payload)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			text := string(data)
			messages = append(messages, types.OpenAIMessage{Role: "tool", ToolCallID: firstNonEmpty(part.FunctionResponse.ID, part.FunctionResponse.Name), Content: &types.OpenAIContent{Text: &text}})
		}
	}
	flush()
	return messages, nil
}

func modelContentToMessages(content Content) ([]types.OpenAIMessage, error) {
	msg := types.OpenAIMessage{Role: "assistant"}
	var textParts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
		if part.FunctionCall != nil {
			args := "{}"
			if part.FunctionCall.Args != nil {
				data, err := json.Marshal(part.FunctionCall.Args)
				if err != nil {
					return nil, fmt.Errorf("failed to marshal function call args: %w", err)
				}
				args = string(data)
			}
			msg.ToolCalls = append(msg.ToolCalls, types.OpenAIToolCall{ID: firstNonEmpty(part.FunctionCall.ID, part.FunctionCall.Name), Type: "function", Function: types.OpenAIToolCallFunction{Name: part.FunctionCall.Name, Arguments: args}})
		}
	}
	if len(textParts) > 0 {
		text := strings.Join(textParts, "\n\n")
		msg.Content = &types.OpenAIContent{Text: &text}
	}
	if msg.Content == nil && len(msg.ToolCalls) == 0 {
		empty := ""
		msg.Content = &types.OpenAIContent{Text: &empty}
	}
	return []types.OpenAIMessage{msg}, nil
}

func flattenParts(parts []Part) string {
	segments := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Text != "" {
			segments = append(segments, part.Text)
		}
	}
	return strings.Join(segments, "\n\n")
}

func fromOpenAIChatResponse(resp types.OpenAIChatCompletionsResponse) GenerateContentResponse {
	out := GenerateContentResponse{ModelVersion: resp.Model}
	for _, choice := range resp.Choices {
		candidate := Candidate{Index: choice.Index, FinishReason: mapFinishReason(choice.FinishReason), Content: Content{Role: "model"}}
		if choice.Message.Content != nil && choice.Message.Content.Text != nil && *choice.Message.Content.Text != "" {
			candidate.Content.Parts = append(candidate.Content.Parts, Part{Text: *choice.Message.Content.Text})
		}
		for _, tc := range choice.Message.ToolCalls {
			var args map[string]interface{}
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			}
			candidate.Content.Parts = append(candidate.Content.Parts, Part{FunctionCall: &FunctionCall{ID: tc.ID, Name: tc.Function.Name, Args: args}})
		}
		out.Candidates = append(out.Candidates, candidate)
	}
	if resp.Usage != nil {
		out.UsageMetadata = &UsageMetadata{PromptTokenCount: resp.Usage.PromptTokens, CandidatesTokenCount: resp.Usage.CompletionTokens, TotalTokenCount: resp.Usage.TotalTokens}
	}
	return out
}

func mapFinishReason(reason string) string {
	switch reason {
	case "stop", "tool_calls", "":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	default:
		return strings.ToUpper(reason)
	}
}

func streamChatAsGemini(w http.ResponseWriter, body io.Reader) error {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	state := &geminiStreamState{}

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			if state.hasContent() {
				return writeGeminiSSE(w, state.response())
			}
			return nil
		}
		var chunk types.OpenAIChatCompletionChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if chunk.Model != "" {
			state.model = chunk.Model
		}
		if chunk.Usage != nil {
			state.usage = &UsageMetadata{PromptTokenCount: chunk.Usage.PromptTokens, CandidatesTokenCount: chunk.Usage.CompletionTokens, TotalTokenCount: chunk.Usage.TotalTokens}
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != nil && choice.Delta.Content.Text != nil {
				state.text.WriteString(*choice.Delta.Content.Text)
			}
			for _, tc := range choice.Delta.ToolCalls {
				state.addToolCall(tc)
			}
			if choice.FinishReason != "" {
				state.finishReason = mapFinishReason(choice.FinishReason)
			}
		}
	}
	return scanner.Err()
}

type geminiStreamState struct {
	text         strings.Builder
	toolCalls    []types.OpenAIToolCall
	toolCallPos  map[int]int
	usage        *UsageMetadata
	finishReason string
	model        string
}

func (s *geminiStreamState) addToolCall(tc types.OpenAIToolCall) {
	if s.toolCallPos == nil {
		s.toolCallPos = make(map[int]int)
	}

	if tc.Index != nil {
		if pos, ok := s.toolCallPos[*tc.Index]; ok {
			mergeToolCall(&s.toolCalls[pos], tc)
			return
		}
	}

	if tc.ID != "" {
		for i := range s.toolCalls {
			if s.toolCalls[i].ID == tc.ID {
				mergeToolCall(&s.toolCalls[i], tc)
				if tc.Index != nil {
					s.toolCallPos[*tc.Index] = i
				}
				return
			}
		}
	}

	if tc.Type == "" {
		tc.Type = "function"
	}
	if tc.Function.Arguments == "" {
		tc.Function.Arguments = "{}"
	}

	pos := len(s.toolCalls)
	s.toolCalls = append(s.toolCalls, tc)
	if tc.Index != nil {
		s.toolCallPos[*tc.Index] = pos
	}
}

func mergeToolCall(dst *types.OpenAIToolCall, src types.OpenAIToolCall) {
	if src.ID != "" {
		dst.ID = src.ID
	}
	if src.Type != "" {
		dst.Type = src.Type
	}
	if src.Index != nil {
		dst.Index = src.Index
	}
	if src.Function.Name != "" {
		dst.Function.Name = src.Function.Name
	}
	if src.Function.Arguments != "" {
		if dst.Function.Arguments == "{}" {
			dst.Function.Arguments = src.Function.Arguments
		} else {
			dst.Function.Arguments += src.Function.Arguments
		}
	}
}

func (s *geminiStreamState) hasContent() bool {
	return s.text.Len() > 0 || len(s.toolCalls) > 0 || s.usage != nil
}

func (s *geminiStreamState) response() GenerateContentResponse {
	resp := GenerateContentResponse{ModelVersion: s.model, UsageMetadata: s.usage}
	candidate := Candidate{Index: 0, FinishReason: firstNonEmpty(s.finishReason, "STOP"), Content: Content{Role: "model"}}
	if s.text.Len() > 0 {
		candidate.Content.Parts = append(candidate.Content.Parts, Part{Text: s.text.String()})
	}
	for _, tc := range s.toolCalls {
		var args map[string]interface{}
		if tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		candidate.Content.Parts = append(candidate.Content.Parts, Part{FunctionCall: &FunctionCall{ID: tc.ID, Name: tc.Function.Name, Args: args}})
	}
	resp.Candidates = []Candidate{candidate}
	return resp
}

func writeGeminiSSE(w io.Writer, payload GenerateContentResponse) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

func writeError(w http.ResponseWriter, code int, status, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(ErrorResponse{Error: APIError{Code: code, Message: message, Status: status}})
}

func collectForwardHeaders(r *http.Request) map[string]string {
	headers := map[string]string{}
	for _, key := range []string{"Content-Type", "Accept", "Cache-Control"} {
		if value := r.Header.Get(key); value != "" {
			headers[key] = value
		}
	}
	return headers
}

func estimateTokens(req GenerateContentRequest) int {
	chars := 0
	if req.SystemInstruction != nil {
		chars += len(flattenParts(req.SystemInstruction.Parts))
	}
	for _, content := range req.Contents {
		for _, part := range content.Parts {
			chars += len(part.Text)
			if part.FunctionCall != nil {
				data, _ := json.Marshal(part.FunctionCall.Args)
				chars += len(part.FunctionCall.Name) + len(data)
			}
			if part.FunctionResponse != nil {
				data, _ := json.Marshal(part.FunctionResponse.Response)
				chars += len(part.FunctionResponse.Name) + len(data)
			}
		}
	}
	for _, tool := range req.Tools {
		for _, decl := range tool.FunctionDeclarations {
			data, _ := json.Marshal(decl.Parameters)
			chars += len(decl.Name) + len(decl.Description) + len(data)
		}
	}
	if chars == 0 {
		return 0
	}
	return int(math.Ceil(float64(chars) / 4.0))
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
