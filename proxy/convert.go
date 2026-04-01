package proxy

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/whtsky/copilot2api/internal/types"
)

// --- Chat Completions → Responses ---

// ConvertChatToResponsesRequest converts an OpenAI Chat Completions request
// to a Responses API request.
func ConvertChatToResponsesRequest(req types.OpenAIChatCompletionsRequest) types.ResponsesRequest {
	result := types.ResponsesRequest{
		Model:    req.Model,
		Stream:   req.Stream,
		Metadata: req.Metadata,
		Store:    false,
	}

	// Extract system messages into instructions field
	var systemTexts []string
	for _, msg := range req.Messages {
		if msg.Role == "system" && msg.Content != nil && msg.Content.Text != nil {
			systemTexts = append(systemTexts, *msg.Content.Text)
		}
	}
	if len(systemTexts) > 0 {
		combined := strings.Join(systemTexts, " ")
		result.Instructions = &combined
	}

	// Convert messages to input items (skips system messages)
	result.Input = convertMessagesToInput(req.Messages)

	// Convert tools
	if len(req.Tools) > 0 {
		result.Tools = convertChatToolsToResponsesTools(req.Tools)
	}

	// Convert tool_choice (normalize non-standard formats)
	if req.ToolChoice != nil {
		result.ToolChoice = normalizeToolChoice(req.ToolChoice)
	}

	// Convert parallel_tool_calls
	if req.ParallelToolCalls != nil {
		result.ParallelToolCalls = *req.ParallelToolCalls
	}

	// Temperature — pass through as-is (pointer, nil = omit from JSON)
	result.Temperature = req.Temperature

	// TopP
	result.TopP = req.TopP

	// MaxOutputTokens
	if req.MaxTokens != nil {
		v := *req.MaxTokens
		result.MaxOutputTokens = &v
	}

	// Thinking budget → reasoning
	if req.ThinkingBudget != nil {
		result.Reasoning = &types.ResponseReasoning{
			Effort:  thinkingBudgetToEffort(*req.ThinkingBudget),
			Summary: "detailed",
		}
	}

	// response_format → text.format
	if req.ResponseFormat != nil {
		result.Text = &types.ResponseText{
			Format: types.ResponseTextFormat{
				Type:       req.ResponseFormat.Type,
				JSONSchema: req.ResponseFormat.JSONSchema,
			},
		}
	}

	return result
}

func convertMessagesToInput(messages []types.OpenAIMessage) []types.ResponseInputItem {
	var input []types.ResponseInputItem

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// System messages are handled by the caller (extracted into instructions)
			continue
		case "user":
			input = append(input, convertUserMessageToInput(msg))
		case "assistant":
			input = append(input, convertAssistantMessageToInput(msg)...)
		case "tool":
			input = append(input, types.ResponseInputItem{
				Type:   "function_call_output",
				CallID: msg.ToolCallID,
				Output: contentToString(msg.Content),
			})
		}
	}

	return input
}

func convertUserMessageToInput(msg types.OpenAIMessage) types.ResponseInputItem {
	if msg.Content != nil {
		if msg.Content.Text != nil {
			return types.ResponseInputItem{
				Type:    "message",
				Role:    "user",
				Content: *msg.Content.Text,
			}
		}
		if len(msg.Content.Parts) > 0 {
			var contentItems []types.ResponseInputContent
			for _, part := range msg.Content.Parts {
				switch part.Type {
				case "text":
					contentItems = append(contentItems, types.ResponseInputContent{
						Type: "input_text",
						Text: part.Text,
					})
				case "image_url":
					if part.ImageURL != nil {
						contentItems = append(contentItems, types.ResponseInputContent{
							Type:     "input_image",
							ImageURL: part.ImageURL.URL,
							Detail:   part.ImageURL.Detail,
						})
					}
				}
			}
			return types.ResponseInputItem{
				Type:    "message",
				Role:    "user",
				Content: contentItems,
			}
		}
	}
	return types.ResponseInputItem{
		Type:    "message",
		Role:    "user",
		Content: "",
	}
}

func convertAssistantMessageToInput(msg types.OpenAIMessage) []types.ResponseInputItem {
	var items []types.ResponseInputItem

	// Emit reasoning items (preserving encrypted_content for multi-turn)
	if len(msg.ReasoningItems) > 0 {
		for _, ri := range msg.ReasoningItems {
			item := types.ResponseInputItem{
				Type:             "reasoning",
				ID:               ri.ID,
				EncryptedContent: ri.EncryptedContent,
			}
			if len(ri.Summary) > 0 {
				summaryBlocks := make([]types.ResponseSummaryBlock, len(ri.Summary))
				for i, s := range ri.Summary {
					summaryBlocks[i] = types.ResponseSummaryBlock{
						Type: s.Type,
						Text: s.Text,
					}
				}
				item.Summary = &summaryBlocks
			}
			items = append(items, item)
		}
	}

	// Add the assistant message itself
	item := types.ResponseInputItem{
		Type: "message",
		Role: "assistant",
	}
	if msg.Content != nil && msg.Content.Text != nil {
		item.Content = *msg.Content.Text
	} else {
		item.Content = ""
	}
	items = append(items, item)

	// Convert tool calls to function_call input items
	for _, tc := range msg.ToolCalls {
		items = append(items, types.ResponseInputItem{
			Type:      "function_call",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
			Status:    "completed",
		})
	}

	return items
}

func convertChatToolsToResponsesTools(tools []types.OpenAIToolDefinition) []types.ResponseTool {
	result := make([]types.ResponseTool, 0, len(tools))
	for _, t := range tools {
		if t.Type == "function" {
			result = append(result, types.ResponseTool{
				Type:        "function",
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  t.Function.Parameters,
				Strict:      false,
			})
		}
	}
	return result
}

func thinkingBudgetToEffort(budget int) string {
	if budget >= 16000 {
		return "high"
	}
	if budget >= 8000 {
		return "medium"
	}
	return "low"
}

func contentToString(content *types.OpenAIContent) string {
	if content == nil {
		return ""
	}
	if content.Text != nil {
		return *content.Text
	}
	if len(content.Parts) > 0 {
		var parts []string
		for _, p := range content.Parts {
			if p.Type == "text" {
				parts = append(parts, p.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return ""
}

// normalizeToolChoice normalizes non-standard tool_choice formats.
// Some clients (like Cursor IDE) send {"type": "auto"} instead of "auto".
func normalizeToolChoice(toolChoice interface{}) interface{} {
	// If it's already a string, pass through
	if _, ok := toolChoice.(string); ok {
		return toolChoice
	}

	// If it's a map, check the type field
	m, ok := toolChoice.(map[string]interface{})
	if !ok {
		return toolChoice
	}

	typeVal, ok := m["type"].(string)
	if !ok {
		return toolChoice
	}

	switch typeVal {
	case "auto":
		return "auto"
	case "none":
		return "none"
	case "required", "tool":
		return "required"
	case "function":
		// Pass through as-is (has function name specifics)
		return toolChoice
	default:
		return toolChoice
	}
}

// ConvertResponsesResultToChatResponse converts a Responses API result
// back to an OpenAI Chat Completions response.
func ConvertResponsesResultToChatResponse(result types.ResponsesResult, model string) types.OpenAIChatCompletionsResponse {
	resp := types.OpenAIChatCompletionsResponse{
		ID:      result.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
	}

	// Build the message from output items
	msg := types.OpenAIMessage{
		Role: "assistant",
	}

	var textParts []string
	var toolCalls []types.OpenAIToolCall

	for _, item := range result.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text":
					textParts = append(textParts, content.Text)
				case "refusal":
					textParts = append(textParts, content.Refusal)
				}
			}
		case "function_call":
			toolCalls = append(toolCalls, types.OpenAIToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: types.OpenAIToolCallFunction{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		case "reasoning":
			// Map reasoning summary to reasoning_text (backward compat)
			if len(item.Summary) > 0 {
				var summaryParts []string
				for _, s := range item.Summary {
					if s.Text != "" {
						summaryParts = append(summaryParts, s.Text)
					}
				}
				if len(summaryParts) > 0 {
					combined := strings.Join(summaryParts, "")
					msg.ReasoningText = &combined
				}
			}
			// Preserve full reasoning item (including encrypted_content)
			ri := types.ReasoningItem{
				ID:               item.ID,
				Type:             "reasoning",
				EncryptedContent: item.EncryptedContent,
			}
			for _, s := range item.Summary {
				ri.Summary = append(ri.Summary, types.ReasoningSummary{
					Type: s.Type,
					Text: s.Text,
				})
			}
			msg.ReasoningItems = append(msg.ReasoningItems, ri)
		}
	}

	if len(textParts) > 0 {
		combined := strings.Join(textParts, "")
		msg.Content = &types.OpenAIContent{Text: &combined}
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
	}

	// Determine finish reason
	finishReason := mapResponsesStatusToFinishReason(&result)

	resp.Choices = []types.OpenAIChoice{{
		Index:        0,
		Message:      msg,
		FinishReason: finishReason,
	}}

	// Map usage
	if result.Usage != nil {
		resp.Usage = &types.OpenAIUsage{
			PromptTokens:     result.Usage.InputTokens,
			CompletionTokens: result.Usage.OutputTokens,
			TotalTokens:      result.Usage.InputTokens + result.Usage.OutputTokens,
		}
		if result.Usage.InputTokensDetails != nil {
			resp.Usage.PromptTokensDetails = &types.OpenAIPromptTokensDetails{
				CachedTokens: result.Usage.InputTokensDetails.CachedTokens,
			}
		}
	}

	return resp
}

func mapResponsesStatusToFinishReason(result *types.ResponsesResult) string {
	if result == nil {
		return "stop"
	}
	switch result.Status {
	case "completed":
		for _, item := range result.Output {
			if item.Type == "function_call" {
				return "tool_calls"
			}
		}
		return "stop"
	case "incomplete":
		if result.IncompleteDetails != nil && result.IncompleteDetails.Reason == "max_output_tokens" {
			return "length"
		}
		return "stop"
	default:
		return "stop"
	}
}

// --- Responses → Chat Completions ---

// ConvertResponsesToChatRequest converts a Responses API request
// to an OpenAI Chat Completions request.
func ConvertResponsesToChatRequest(req types.ResponsesRequest) types.OpenAIChatCompletionsRequest {
	result := types.OpenAIChatCompletionsRequest{
		Model:    req.Model,
		Stream:   req.Stream,
		Metadata: req.Metadata,
	}

	// Convert input items to messages
	result.Messages = convertInputToMessages(req.Input, req.Instructions)

	// Convert tools
	if len(req.Tools) > 0 {
		result.Tools = convertResponsesToolsToChatTools(req.Tools)
	}

	// Convert tool_choice (normalize non-standard formats)
	if req.ToolChoice != nil {
		result.ToolChoice = normalizeToolChoice(req.ToolChoice)
	}

	// ParallelToolCalls
	parallelCalls := req.ParallelToolCalls
	result.ParallelToolCalls = &parallelCalls

	// Temperature — pass through as-is (pointer, nil = omit)
	result.Temperature = req.Temperature

	// TopP
	result.TopP = req.TopP

	// MaxTokens
	if req.MaxOutputTokens != nil {
		v := *req.MaxOutputTokens
		result.MaxTokens = &v
	}

	// Reasoning → thinking budget
	if req.Reasoning != nil {
		budget := effortToThinkingBudget(req.Reasoning.Effort)
		result.ThinkingBudget = &budget
	}

	// text.format → response_format
	if req.Text != nil {
		result.ResponseFormat = &types.ResponseFormat{
			Type:       req.Text.Format.Type,
			JSONSchema: req.Text.Format.JSONSchema,
		}
	}

	// previous_response_id: stateless proxy, log and ignore during conversion
	if req.PreviousResponseID != "" {
		slog.Debug("previous_response_id set during responses→chat conversion, ignoring (stateless proxy)", "previous_response_id", req.PreviousResponseID)
	}

	// Stream options
	if req.Stream {
		result.StreamOptions = &types.OpenAIStreamOptions{IncludeUsage: true}
	}

	return result
}

func convertInputToMessages(input []types.ResponseInputItem, instructions *string) []types.OpenAIMessage {
	var messages []types.OpenAIMessage

	// Add instructions as system message
	if instructions != nil && *instructions != "" {
		messages = append(messages, types.OpenAIMessage{
			Role:    "system",
			Content: &types.OpenAIContent{Text: instructions},
		})
	}

	for _, item := range input {
		switch item.Type {
		case "message":
			messages = append(messages, convertInputItemToMessage(item))
		case "function_call":
			// Becomes a tool_call on the previous assistant message
			// or creates a new assistant message with the tool call
			messages = appendFunctionCallAsToolCall(messages, item)
		case "function_call_output":
			outputStr := ""
			if s, ok := item.Output.(string); ok {
				outputStr = s
			} else if item.Output != nil {
				data, _ := json.Marshal(item.Output)
				outputStr = string(data)
			}
			messages = append(messages, types.OpenAIMessage{
				Role:       "tool",
				ToolCallID: item.CallID,
				Content:    &types.OpenAIContent{Text: &outputStr},
			})
		}
	}

	return messages
}

func convertInputItemToMessage(item types.ResponseInputItem) types.OpenAIMessage {
	msg := types.OpenAIMessage{
		Role: item.Role,
	}

	switch v := item.Content.(type) {
	case string:
		msg.Content = &types.OpenAIContent{Text: &v}
	case []interface{}:
		// Array of content items
		var parts []types.OpenAIContentPart
		for _, raw := range v {
			if m, ok := raw.(map[string]interface{}); ok {
				part := types.OpenAIContentPart{}
				if t, ok := m["type"].(string); ok {
					part.Type = mapResponsesContentTypeToChat(t)
				}
				if text, ok := m["text"].(string); ok {
					part.Text = text
				}
				if imgURL, ok := m["image_url"].(string); ok {
					part.Type = "image_url"
					detail := ""
					if d, ok := m["detail"].(string); ok {
						detail = d
					}
					part.ImageURL = &types.OpenAIImageURLPayload{
						URL:    imgURL,
						Detail: detail,
					}
				}
				parts = append(parts, part)
			}
		}
		if len(parts) > 0 {
			msg.Content = &types.OpenAIContent{Parts: parts}
		}
	case []types.ResponseInputContent:
		// Typed content items
		var parts []types.OpenAIContentPart
		for _, c := range v {
			switch c.Type {
			case "input_text", "output_text":
				parts = append(parts, types.OpenAIContentPart{
					Type: "text",
					Text: c.Text,
				})
			case "input_image":
				parts = append(parts, types.OpenAIContentPart{
					Type: "image_url",
					ImageURL: &types.OpenAIImageURLPayload{
						URL:    c.ImageURL,
						Detail: c.Detail,
					},
				})
			}
		}
		if len(parts) > 0 {
			msg.Content = &types.OpenAIContent{Parts: parts}
		}
	}

	return msg
}

func appendFunctionCallAsToolCall(messages []types.OpenAIMessage, item types.ResponseInputItem) []types.OpenAIMessage {
	tc := types.OpenAIToolCall{
		ID:   item.CallID,
		Type: "function",
		Function: types.OpenAIToolCallFunction{
			Name:      item.Name,
			Arguments: item.Arguments,
		},
	}

	// Try to attach to the last assistant message
	if len(messages) > 0 && messages[len(messages)-1].Role == "assistant" {
		messages[len(messages)-1].ToolCalls = append(messages[len(messages)-1].ToolCalls, tc)
		return messages
	}

	// Create a new assistant message with the tool call
	return append(messages, types.OpenAIMessage{
		Role:      "assistant",
		ToolCalls: []types.OpenAIToolCall{tc},
	})
}

func convertResponsesToolsToChatTools(tools []types.ResponseTool) []types.OpenAIToolDefinition {
	result := make([]types.OpenAIToolDefinition, 0, len(tools))
	for _, t := range tools {
		if t.Type == "function" {
			result = append(result, types.OpenAIToolDefinition{
				Type: "function",
				Function: types.OpenAIFunctionSpec{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.Parameters,
				},
			})
		}
	}
	return result
}

func mapResponsesContentTypeToChat(contentType string) string {
	switch contentType {
	case "input_text", "output_text":
		return "text"
	case "input_image":
		return "image_url"
	default:
		return contentType
	}
}

func effortToThinkingBudget(effort string) int {
	switch effort {
	case "high":
		return 32000
	case "medium":
		return 12000
	case "low":
		return 4000
	default:
		return 32000
	}
}

// ConvertChatResponseToResponsesResult converts an OpenAI Chat Completions
// response to a Responses API result.
func ConvertChatResponseToResponsesResult(resp types.OpenAIChatCompletionsResponse) types.ResponsesResult {
	result := types.ResponsesResult{
		ID:    resp.ID,
		Model: resp.Model,
	}

	var outputItems []types.ResponseOutputItem

	for _, choice := range resp.Choices {
		// Handle reasoning text
		if choice.Message.ReasoningText != nil && *choice.Message.ReasoningText != "" {
			reasoningItem := types.ResponseOutputItem{
				Type: "reasoning",
				ID:   fmt.Sprintf("reasoning_%s", resp.ID),
			}
			if *choice.Message.ReasoningText != "" {
				reasoningItem.Summary = []types.ResponseSummaryBlock{{
					Type: "summary_text",
					Text: *choice.Message.ReasoningText,
				}}
			}
			outputItems = append(outputItems, reasoningItem)
		}

		// Handle text content
		if choice.Message.Content != nil {
			var text string
			if choice.Message.Content.Text != nil {
				text = *choice.Message.Content.Text
			} else if len(choice.Message.Content.Parts) > 0 {
				var parts []string
				for _, p := range choice.Message.Content.Parts {
					if p.Type == "text" {
						parts = append(parts, p.Text)
					}
				}
				text = strings.Join(parts, "")
			}
			if text != "" {
				outputItems = append(outputItems, types.ResponseOutputItem{
					Type: "message",
					Content: []types.ResponseOutputContent{{
						Type: "output_text",
						Text: text,
					}},
				})
			}
		}

		// Handle tool calls
		for _, tc := range choice.Message.ToolCalls {
			outputItems = append(outputItems, types.ResponseOutputItem{
				Type:      "function_call",
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	result.Output = outputItems

	// Build output_text from message content
	var textParts []string
	for _, item := range outputItems {
		if item.Type == "message" {
			for _, c := range item.Content {
				if c.Type == "output_text" {
					textParts = append(textParts, c.Text)
				}
			}
		}
	}
	result.OutputText = strings.Join(textParts, "")

	// Map status from finish reason
	result.Status = mapChatFinishReasonToResponsesStatus(resp)

	// Map usage
	if resp.Usage != nil {
		result.Usage = &types.ResponsesUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		}
		if resp.Usage.PromptTokensDetails != nil {
			result.Usage.InputTokensDetails = &types.InputTokenDetails{
				CachedTokens: resp.Usage.PromptTokensDetails.CachedTokens,
			}
		}
	}

	// Map incomplete details
	if result.Status == "incomplete" {
		result.IncompleteDetails = &types.IncompleteDetails{
			Reason: "max_output_tokens",
		}
	}

	return result
}

func mapChatFinishReasonToResponsesStatus(resp types.OpenAIChatCompletionsResponse) string {
	if len(resp.Choices) == 0 {
		return "completed"
	}
	switch resp.Choices[0].FinishReason {
	case "length":
		return "incomplete"
	default:
		return "completed"
	}
}

// --- Streaming conversion helpers ---

// ConvertResponsesStreamEventToChatChunk translates a Responses API stream event
// into zero or more Chat Completions chunks.
func ConvertResponsesStreamEventToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	switch event.Type {
	case "response.created":
		return responsesCreatedToChatChunk(event, state)
	case "response.output_text.delta":
		return responsesTextDeltaToChatChunk(event, state)
	case "response.function_call_arguments.delta":
		return responsesFuncArgsDeltaToChatChunk(event, state)
	case "response.output_item.added":
		return responsesOutputItemAddedToChatChunk(event, state)
	case "response.reasoning_summary_text.delta":
		return responsesReasoningDeltaToChatChunk(event, state)
	case "response.completed", "response.incomplete":
		return responsesCompletedToChatChunk(event, state)
	case "response.failed", "error":
		return responsesFailedToChatChunk(event, state)
	default:
		return nil
	}
}

// ResponsesStreamConvertState tracks state during Responses→ChatCompletions stream conversion.
type ResponsesStreamConvertState struct {
	ID      string
	Model   string
	Created int64
	// ToolCallIndexByOutputIndex maps responses output_index to chat completions tool call index
	ToolCallIndexByOutputIndex map[int]int
	NextToolCallIndex          int
	Finished                   bool
}

// NewResponsesStreamConvertState creates a new conversion state.
func NewResponsesStreamConvertState() *ResponsesStreamConvertState {
	return &ResponsesStreamConvertState{
		ToolCallIndexByOutputIndex: make(map[int]int),
		Created:                    time.Now().Unix(),
	}
}

func responsesCreatedToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	if event.Response != nil {
		state.ID = event.Response.ID
		state.Model = event.Response.Model
	}
	// Emit an initial empty chunk (role indicator)
	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index: 0,
			Delta: types.OpenAIMessage{Role: "assistant"},
		}},
	}}
}

func responsesTextDeltaToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	if event.Delta == "" {
		return nil
	}
	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index: 0,
			Delta: types.OpenAIMessage{
				Content: &types.OpenAIContent{Text: &event.Delta},
			},
		}},
	}}
}

func responsesReasoningDeltaToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	if event.Delta == "" {
		return nil
	}
	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index: 0,
			Delta: types.OpenAIMessage{
				ReasoningText: &event.Delta,
			},
		}},
	}}
}

func responsesFuncArgsDeltaToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	if event.Delta == "" {
		return nil
	}
	tcIndex, ok := state.ToolCallIndexByOutputIndex[event.OutputIndex]
	if !ok {
		return nil
	}
	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index: 0,
			Delta: types.OpenAIMessage{
				ToolCalls: []types.OpenAIToolCall{{
					Index: &tcIndex,
					Function: types.OpenAIToolCallFunction{
						Arguments: event.Delta,
					},
				}},
			},
		}},
	}}
}

func responsesOutputItemAddedToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	if event.Item == nil || event.Item.Type != "function_call" {
		return nil
	}

	tcIndex := state.NextToolCallIndex
	state.ToolCallIndexByOutputIndex[event.OutputIndex] = tcIndex
	state.NextToolCallIndex++

	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index: 0,
			Delta: types.OpenAIMessage{
				ToolCalls: []types.OpenAIToolCall{{
					Index: &tcIndex,
					ID:    event.Item.CallID,
					Type:  "function",
					Function: types.OpenAIToolCallFunction{
						Name: event.Item.Name,
					},
				}},
			},
		}},
	}}
}

func responsesCompletedToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	state.Finished = true

	finishReason := "stop"
	var usage *types.OpenAIUsage
	if event.Response != nil {
		finishReason = mapResponsesStatusToFinishReason(event.Response)
		if event.Response.Usage != nil {
			usage = &types.OpenAIUsage{
				PromptTokens:     event.Response.Usage.InputTokens,
				CompletionTokens: event.Response.Usage.OutputTokens,
				TotalTokens:      event.Response.Usage.InputTokens + event.Response.Usage.OutputTokens,
			}
			if event.Response.Usage.InputTokensDetails != nil {
				usage.PromptTokensDetails = &types.OpenAIPromptTokensDetails{
					CachedTokens: event.Response.Usage.InputTokensDetails.CachedTokens,
				}
			}
		}
	}

	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index:        0,
			Delta:        types.OpenAIMessage{},
			FinishReason: finishReason,
		}},
		Usage: usage,
	}}
}

func responsesFailedToChatChunk(event types.ResponseStreamEvent, state *ResponsesStreamConvertState) []types.OpenAIChatCompletionChunk {
	state.Finished = true

	// Map response.failed / error to a "stop" finish with the content we have.
	// Chat Completions doesn't have a "failed" finish reason, so we use "stop"
	// and let the lack of content signal the failure to the client.
	return []types.OpenAIChatCompletionChunk{{
		ID:      state.ID,
		Object:  "chat.completion.chunk",
		Created: state.Created,
		Model:   state.Model,
		Choices: []types.OpenAIChunkChoice{{
			Index:        0,
			Delta:        types.OpenAIMessage{},
			FinishReason: "stop",
		}},
	}}
}

// ConvertChatChunkToResponsesStreamEvents translates a Chat Completions chunk
// into zero or more Responses API stream events.
func ConvertChatChunkToResponsesStreamEvents(chunk types.OpenAIChatCompletionChunk, state *ChatStreamConvertState) []types.ResponseStreamEvent {
	var events []types.ResponseStreamEvent

	// Usage-only chunk (choices is empty, usage is populated).
	// This arrives after the finish_reason chunk when stream_options.include_usage
	// is set. Capture usage and emit the deferred termination event.
	if len(chunk.Choices) == 0 {
		if chunk.Usage != nil {
			state.PendingUsage = chunk.Usage
		}
		// If finish was already seen, emit the termination event now with usage
		if state.FinishSeen && !state.Finished {
			events = append(events, state.buildTerminationEvent())
		}
		return events
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// First chunk: emit response.created
	if !state.CreatedSent {
		state.CreatedSent = true
		state.ID = chunk.ID
		state.Model = chunk.Model
		events = append(events, types.ResponseStreamEvent{
			Type: "response.created",
			Response: &types.ResponsesResult{
				ID:     chunk.ID,
				Model:  chunk.Model,
				Status: "in_progress",
			},
		})
	}

	// Handle text content delta
	if delta.Content != nil && delta.Content.Text != nil && *delta.Content.Text != "" {
		if !state.OutputItemStarted {
			state.OutputItemStarted = true
			state.CurrentOutputIndex = state.NextOutputIndex
			state.NextOutputIndex++
			events = append(events, types.ResponseStreamEvent{
				Type:        "response.output_item.added",
				OutputIndex: state.CurrentOutputIndex,
				Item: &types.ResponseOutputItem{
					Type: "message",
					Content: []types.ResponseOutputContent{{
						Type: "output_text",
						Text: "",
					}},
				},
			})
		}
		events = append(events, types.ResponseStreamEvent{
			Type:        "response.output_text.delta",
			OutputIndex: state.CurrentOutputIndex,
			Delta:       *delta.Content.Text,
		})
	}

	// Handle reasoning text delta
	if delta.ReasoningText != nil && *delta.ReasoningText != "" {
		if !state.ReasoningStarted {
			state.ReasoningStarted = true
			state.ReasoningOutputIndex = state.NextOutputIndex
			state.NextOutputIndex++
			events = append(events, types.ResponseStreamEvent{
				Type:        "response.output_item.added",
				OutputIndex: state.ReasoningOutputIndex,
				Item: &types.ResponseOutputItem{
					Type: "reasoning",
					Summary: []types.ResponseSummaryBlock{{
						Type: "summary_text",
						Text: "",
					}},
				},
			})
		}
		events = append(events, types.ResponseStreamEvent{
			Type:         "response.reasoning_summary_text.delta",
			OutputIndex:  state.ReasoningOutputIndex,
			Delta:        *delta.ReasoningText,
		})
	}

	// Handle tool calls
	for _, tc := range delta.ToolCalls {
		if tc.Index == nil {
			continue
		}
		idx := *tc.Index

		// New tool call
		if tc.ID != "" && tc.Function.Name != "" {
			outputIdx := state.NextOutputIndex
			state.NextOutputIndex++
			state.ToolCallOutputIndex[idx] = outputIdx
			events = append(events, types.ResponseStreamEvent{
				Type:        "response.output_item.added",
				OutputIndex: outputIdx,
				Item: &types.ResponseOutputItem{
					Type:   "function_call",
					CallID: tc.ID,
					Name:   tc.Function.Name,
				},
			})
		}

		// Tool call arguments delta
		if tc.Function.Arguments != "" {
			if outputIdx, ok := state.ToolCallOutputIndex[idx]; ok {
				events = append(events, types.ResponseStreamEvent{
					Type:        "response.function_call_arguments.delta",
					OutputIndex: outputIdx,
					Delta:       tc.Function.Arguments,
				})
			}
		}
	}

	// Handle finish reason — defer the termination event to capture usage
	// from a subsequent usage-only chunk (if stream_options.include_usage is set).
	if choice.FinishReason != "" {
		state.FinishSeen = true
		state.FinishStatus = "completed"
		if choice.FinishReason == "length" {
			state.FinishStatus = "incomplete"
		}

		// If usage is already on this chunk, emit termination immediately
		if chunk.Usage != nil {
			state.PendingUsage = chunk.Usage
			events = append(events, state.buildTerminationEvent())
		}
		// Otherwise, defer — termination event will be emitted when usage
		// chunk arrives or when [DONE] is seen.
	}

	return events
}

// ChatStreamConvertState tracks state during ChatCompletions→Responses stream conversion.
type ChatStreamConvertState struct {
	CreatedSent        bool
	ID                 string
	Model              string
	OutputItemStarted  bool
	CurrentOutputIndex int
	NextOutputIndex    int
	ToolCallOutputIndex map[int]int
	Finished           bool
	// Reasoning tracking
	ReasoningStarted     bool
	ReasoningOutputIndex int
	// Deferred termination fields
	FinishSeen   bool             // finish_reason was seen
	FinishStatus string           // "completed" or "incomplete"
	PendingUsage *types.OpenAIUsage // usage captured from usage-only chunk
}

// buildTerminationEvent creates the response.completed/incomplete event and
// marks the state as finished. Should only be called after FinishSeen is true.
func (s *ChatStreamConvertState) buildTerminationEvent() types.ResponseStreamEvent {
	s.Finished = true

	result := &types.ResponsesResult{
		ID:     s.ID,
		Model:  s.Model,
		Status: s.FinishStatus,
	}

	if s.PendingUsage != nil {
		result.Usage = &types.ResponsesUsage{
			InputTokens:  s.PendingUsage.PromptTokens,
			OutputTokens: s.PendingUsage.CompletionTokens,
		}
		if s.PendingUsage.PromptTokensDetails != nil {
			result.Usage.InputTokensDetails = &types.InputTokenDetails{
				CachedTokens: s.PendingUsage.PromptTokensDetails.CachedTokens,
			}
		}
	}

	eventType := "response.completed"
	if s.FinishStatus == "incomplete" {
		eventType = "response.incomplete"
	}

	return types.ResponseStreamEvent{
		Type:     eventType,
		Response: result,
	}
}

// NewChatStreamConvertState creates a new conversion state.
func NewChatStreamConvertState() *ChatStreamConvertState {
	return &ChatStreamConvertState{
		ToolCallOutputIndex: make(map[int]int),
	}
}
