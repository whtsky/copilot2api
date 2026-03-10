package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Convert Anthropic request to OpenAI request
func ConvertAnthropicToOpenAI(req AnthropicMessagesRequest) (OpenAIChatCompletionsRequest, error) {
	openAIReq := OpenAIChatCompletionsRequest{
		Model:       req.Model,
		MaxTokens:   &req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Metadata:    req.Metadata,
	}

	// Request usage in streaming chunks so message_delta gets real output_tokens
	if req.Stream {
		openAIReq.StreamOptions = &OpenAIStreamOptions{IncludeUsage: true}
	}

	// Map thinking configuration
	if req.Thinking != nil && req.Thinking.BudgetTokens != nil {
		openAIReq.ThinkingBudget = req.Thinking.BudgetTokens
	}

	// Convert stop sequences
	if len(req.StopSequences) > 0 {
		if len(req.StopSequences) == 1 {
			openAIReq.Stop = req.StopSequences[0]
		} else {
			openAIReq.Stop = req.StopSequences
		}
	}

	// Convert messages
	messages, err := convertAnthropicMessagesToOpenAI(req.System, req.Messages)
	if err != nil {
		return openAIReq, fmt.Errorf("failed to convert messages: %w", err)
	}
	openAIReq.Messages = messages

	// Convert tools
	if len(req.Tools) > 0 {
		openAIReq.Tools = convertAnthropicToolsToOpenAI(req.Tools)
	}

	// Convert tool choice
	if req.ToolChoice != nil {
		toolChoice, parallelToolCalls := convertAnthropicToolChoiceToOpenAI(req.ToolChoice)
		openAIReq.ToolChoice = toolChoice
		if parallelToolCalls != nil {
			openAIReq.ParallelToolCalls = parallelToolCalls
		}
	}

	// Add user ID if present in metadata
	if req.Metadata != nil && req.Metadata["user_id"] != "" {
		openAIReq.User = req.Metadata["user_id"]
	}

	return openAIReq, nil
}

func convertAnthropicMessagesToOpenAI(system *AnthropicSystem, messages []AnthropicMessage) ([]OpenAIMessage, error) {
	var openAIMessages []OpenAIMessage

	// Convert system message
	if system != nil {
		systemMsg, err := convertSystemToOpenAI(system)
		if err != nil {
			return nil, fmt.Errorf("failed to convert system message: %w", err)
		}
		if systemMsg != nil {
			openAIMessages = append(openAIMessages, *systemMsg)
		}
	}

	// Convert messages
	for _, msg := range messages {
		converted, err := convertAnthropicMessageToOpenAI(msg)
		if err != nil {
			return nil, fmt.Errorf("failed to convert message: %w", err)
		}
		openAIMessages = append(openAIMessages, converted...)
	}

	return openAIMessages, nil
}

func convertSystemToOpenAI(system *AnthropicSystem) (*OpenAIMessage, error) {
	text := extractSystemText(system)
	if text == nil {
		return nil, nil
	}
	return &OpenAIMessage{
		Role:    "system",
		Content: &OpenAIContent{Text: text},
	}, nil
}

func convertAnthropicMessageToOpenAI(msg AnthropicMessage) ([]OpenAIMessage, error) {
	if msg.Role == "user" {
		return convertUserMessageToOpenAI(msg)
	} else if msg.Role == "assistant" {
		return convertAssistantMessageToOpenAI(msg)
	}

	return nil, fmt.Errorf("unsupported message role: %s", msg.Role)
}

func convertUserMessageToOpenAI(msg AnthropicMessage) ([]OpenAIMessage, error) {
	var openAIMessages []OpenAIMessage

	// If content is just text
	if msg.Content.Text != nil {
		return []OpenAIMessage{{
			Role: "user",
			Content: &OpenAIContent{
				Text: msg.Content.Text,
			},
		}}, nil
	}

	// Process content blocks in order, preserving the original sequence.
	// Consecutive non-tool-result blocks are batched into a single user message;
	// each tool_result block becomes its own tool message.
	if len(msg.Content.Blocks) > 0 {
		var pendingBlocks []AnthropicContentBlock

		flushPending := func() error {
			if len(pendingBlocks) == 0 {
				return nil
			}
			content, err := convertContentBlocksToOpenAI(pendingBlocks)
			if err != nil {
				return fmt.Errorf("failed to convert content blocks: %w", err)
			}
			openAIMessages = append(openAIMessages, OpenAIMessage{
				Role:    "user",
				Content: content,
			})
			pendingBlocks = nil
			return nil
		}

		for _, block := range msg.Content.Blocks {
			if block.Type == "tool_result" {
				// Flush any pending non-tool blocks first
				if err := flushPending(); err != nil {
					return nil, err
				}

				content, err := convertToolResultContent(block)
				if err != nil {
					return nil, fmt.Errorf("failed to convert tool result: %w", err)
				}

				openAIMessages = append(openAIMessages, OpenAIMessage{
					Role:       "tool",
					ToolCallID: block.ToolUseID,
					Content: &OpenAIContent{
						Text: &content,
					},
				})
			} else {
				pendingBlocks = append(pendingBlocks, block)
			}
		}

		// Flush any remaining non-tool blocks
		if err := flushPending(); err != nil {
			return nil, err
		}
	}

	return openAIMessages, nil
}

func convertAssistantMessageToOpenAI(msg AnthropicMessage) ([]OpenAIMessage, error) {
	if msg.Content.Text != nil {
		return []OpenAIMessage{{
			Role: "assistant",
			Content: &OpenAIContent{
				Text: msg.Content.Text,
			},
		}}, nil
	}

	if len(msg.Content.Blocks) == 0 {
		return []OpenAIMessage{{
			Role: "assistant",
			Content: &OpenAIContent{
				Text: stringPtr(""),
			},
		}}, nil
	}

	// Extract different block types
	var textBlocks []AnthropicContentBlock
	var toolUseBlocks []AnthropicContentBlock
	var thinkingBlocks []AnthropicContentBlock

	for _, block := range msg.Content.Blocks {
		switch block.Type {
		case "text":
			textBlocks = append(textBlocks, block)
		case "tool_use":
			toolUseBlocks = append(toolUseBlocks, block)
		case "thinking":
			thinkingBlocks = append(thinkingBlocks, block)
		}
	}

	// Build the OpenAI message
	openAIMsg := OpenAIMessage{
		Role: "assistant",
	}

	// Handle text content
	if len(textBlocks) > 0 {
		var textParts []string
		for _, block := range textBlocks {
			if block.Text != "" {
				textParts = append(textParts, block.Text)
			}
		}
		if len(textParts) > 0 {
			combined := strings.Join(textParts, "\n\n")
			openAIMsg.Content = &OpenAIContent{
				Text: &combined,
			}
		}
	}

	// Handle tool calls
	if len(toolUseBlocks) > 0 {
		var toolCalls []OpenAIToolCall
		for _, block := range toolUseBlocks {
			args, err := json.Marshal(block.Input)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool input: %w", err)
			}

			toolCalls = append(toolCalls, OpenAIToolCall{
				ID:   block.ID,
				Type: "function",
				Function: OpenAIToolCallFunction{
					Name:      block.Name,
					Arguments: string(args),
				},
			})
		}
		openAIMsg.ToolCalls = toolCalls
	}

	// Handle thinking blocks (reasoning)
	if len(thinkingBlocks) > 0 {
		var thinkingTexts []string
		var signature string

		for _, block := range thinkingBlocks {
			if block.Thinking != "" {
				thinkingTexts = append(thinkingTexts, block.Thinking)
			}
			if block.Signature != "" {
				signature = block.Signature
			}
		}

		if len(thinkingTexts) > 0 {
			combined := strings.Join(thinkingTexts, "\n\n")
			openAIMsg.ReasoningText = &combined
		}

		if signature != "" {
			openAIMsg.ReasoningOpaque = &signature
		}
	}

	return []OpenAIMessage{openAIMsg}, nil
}

func convertToolResultContent(toolResult AnthropicContentBlock) (string, error) {
	if toolResult.Content == nil {
		return "", nil
	}

	if toolResult.Content.Text != nil {
		return *toolResult.Content.Text, nil
	}

	if len(toolResult.Content.Blocks) > 0 {
		var textParts []string
		for _, block := range toolResult.Content.Blocks {
			if block.Type == "text" {
				textParts = append(textParts, block.Text)
			}
		}
		return strings.Join(textParts, "\n"), nil
	}

	return "", nil
}

func convertContentBlocksToOpenAI(blocks []AnthropicContentBlock) (*OpenAIContent, error) {
	hasImages := false
	for _, block := range blocks {
		if block.Type == "image" {
			hasImages = true
			break
		}
	}

	if !hasImages {
		// Text only - combine all text blocks
		var textParts []string
		for _, block := range blocks {
			if block.Type == "text" {
				textParts = append(textParts, block.Text)
			}
		}
		if len(textParts) > 0 {
			combined := strings.Join(textParts, "\n\n")
			return &OpenAIContent{Text: &combined}, nil
		}
		return &OpenAIContent{Text: stringPtr("")}, nil
	}

	// Has images - create parts array
	var parts []OpenAIContentPart
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, OpenAIContentPart{
				Type: "text",
				Text: block.Text,
			})
		case "image":
			if block.Source != nil {
				url := imageDataURL(block.Source.MediaType, block.Source.Data)
				parts = append(parts, OpenAIContentPart{
					Type: "image_url",
					ImageURL: &OpenAIImageURLPayload{
						URL: url,
					},
				})
			}
		}
	}

	return &OpenAIContent{Parts: parts}, nil
}

func convertAnthropicToolsToOpenAI(tools []AnthropicTool) []OpenAIToolDefinition {
	defs := extractToolDefs(tools)
	openAITools := make([]OpenAIToolDefinition, len(defs))
	for i, d := range defs {
		openAITools[i] = OpenAIToolDefinition{
			Type: "function",
			Function: OpenAIFunctionSpec{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.InputSchema,
			},
		}
	}
	return openAITools
}

func convertAnthropicToolChoiceToOpenAI(choice *AnthropicToolChoice) (interface{}, *bool) {
	tc := mapToolChoice(choice)

	var parallelCalls *bool
	if tc.DisableParallelCalls != nil {
		inverted := !(*tc.DisableParallelCalls)
		parallelCalls = &inverted
	}

	switch tc.Mode {
	case "auto":
		return "auto", parallelCalls
	case "required":
		return "required", parallelCalls
	case "function":
		return map[string]interface{}{
			"type": "function",
			"function": map[string]string{
				"name": tc.FunctionName,
			},
		}, parallelCalls
	case "none":
		return "none", parallelCalls
	default:
		return nil, parallelCalls
	}
}

// Convert OpenAI response to Anthropic response.
// The Copilot API may return split choices for Claude models: text content in
// one choice and tool_calls in another. We merge all choices into a single
// Anthropic message, picking the most significant finish_reason.
func ConvertOpenAIToAnthropic(resp OpenAIChatCompletionsResponse) (AnthropicMessagesResponse, error) {
	if len(resp.Choices) == 0 {
		return AnthropicMessagesResponse{}, fmt.Errorf("no choices in OpenAI response")
	}

	var contentBlocks []AnthropicContentBlock
	bestFinishReason := ""

	for _, choice := range resp.Choices {
		// Handle thinking blocks (from reasoning fields)
		if choice.Message.ReasoningText != nil && *choice.Message.ReasoningText != "" {
			signature := ""
			if choice.Message.ReasoningOpaque != nil {
				signature = *choice.Message.ReasoningOpaque
			}

			contentBlocks = append(contentBlocks, AnthropicContentBlock{
				Type:      "thinking",
				Thinking:  *choice.Message.ReasoningText,
				Signature: signature,
			})
		}

		// Handle text content
		if choice.Message.Content != nil {
			if choice.Message.Content.Text != nil && *choice.Message.Content.Text != "" {
				contentBlocks = append(contentBlocks, AnthropicContentBlock{
					Type: "text",
					Text: *choice.Message.Content.Text,
				})
			} else if len(choice.Message.Content.Parts) > 0 {
				for _, part := range choice.Message.Content.Parts {
					if part.Type == "text" {
						contentBlocks = append(contentBlocks, AnthropicContentBlock{
							Type: "text",
							Text: part.Text,
						})
					}
				}
			}
		}

		// Handle tool calls
		for _, toolCall := range choice.Message.ToolCalls {
			var input map[string]interface{}
			if toolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &input); err != nil {
					// If parsing fails, store as raw string
					input = map[string]interface{}{
						"raw_arguments": toolCall.Function.Arguments,
					}
				}
			}

			contentBlocks = append(contentBlocks, AnthropicContentBlock{
				Type:  "tool_use",
				ID:    toolCall.ID,
				Name:  toolCall.Function.Name,
				Input: input,
			})
		}

		// Track the most significant finish_reason: tool_calls > stop > others
		bestFinishReason = pickFinishReason(bestFinishReason, choice.FinishReason)
	}

	// If no content blocks, add empty text block
	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, AnthropicContentBlock{
			Type: "text",
			Text: "",
		})
	}

	// Map finish reason
	stopReason := mapOpenAIFinishReasonToAnthropic(bestFinishReason)

	// Calculate usage
	usage := AnthropicUsage{}
	if resp.Usage != nil {
		usage.InputTokens = resp.Usage.PromptTokens
		usage.OutputTokens = resp.Usage.CompletionTokens

		// Handle cached tokens
		if resp.Usage.PromptTokensDetails != nil {
			usage.InputTokens, usage.CacheReadInputTokens = adjustCachedUsage(usage.InputTokens, resp.Usage.PromptTokensDetails.CachedTokens)
		}
	}

	return AnthropicMessagesResponse{
		ID:         resp.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      resp.Model,
		Content:    contentBlocks,
		StopReason: stopReason,
		Usage:      usage,
	}, nil
}

// pickFinishReason returns the more significant of two finish reasons.
// Priority: "tool_calls" > "stop" > "length" > any other non-empty > empty.
func pickFinishReason(current, candidate string) string {
	if current == "" {
		return candidate
	}
	if candidate == "" {
		return current
	}
	priority := map[string]int{
		"tool_calls":     3,
		"stop":           2,
		"length":         1,
		"content_filter": 0,
	}
	if priority[candidate] > priority[current] {
		return candidate
	}
	return current
}

func mapOpenAIFinishReasonToAnthropic(reason string) string {
	switch reason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "content_filter":
		return "end_turn"
	default:
		return "end_turn"
	}
}

// Helper function
func stringPtr(s string) *string {
	return &s
}
