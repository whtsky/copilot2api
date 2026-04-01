package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
)

// resolveReasoningEffort determines the reasoning effort level for the Responses API.
// OutputConfig.Effort takes priority when set; otherwise falls back to thinking budget.
func resolveReasoningEffort(thinking *AnthropicThinking, outputConfig *AnthropicOutputConfig) string {
	if outputConfig != nil && outputConfig.Effort != "" {
		effort := outputConfig.Effort
		if effort == "max" {
			effort = "high" // Responses API doesn't support "max"
		}
		return effort
	}
	return thinkingEffort(thinking)
}

// thinkingEffort maps Anthropic thinking budget to Responses API effort level.
func thinkingEffort(thinking *AnthropicThinking) string {
	if thinking == nil || thinking.BudgetTokens == nil {
		return "high"
	}
	budget := *thinking.BudgetTokens
	if budget >= 16000 {
		return "high"
	}
	if budget >= 8000 {
		return "medium"
	}
	return "low"
}

func ConvertAnthropicToResponses(req AnthropicMessagesRequest) (ResponsesRequest, error) {
	input, err := convertMessagesToResponsesInput(req.Messages, req.Model)
	if err != nil {
		return ResponsesRequest{}, fmt.Errorf("failed to convert messages: %w", err)
	}

	instructions := extractSystemText(req.System)
	tools := convertToolsToResponsesFormat(req.Tools)
	toolChoice := convertToolChoiceToResponsesFormat(req.ToolChoice)

	maxOutputTokens := req.MaxTokens
	if maxOutputTokens < 12800 {
		maxOutputTokens = 12800
	}

	ptc := req.ToolChoice == nil || req.ToolChoice.DisableParallelCalls == nil || !*req.ToolChoice.DisableParallelCalls

	result := ResponsesRequest{
		Model:             req.Model,
		Input:             input,
		Instructions:      instructions,
		Temperature:       ptrFloat64(1), // Responses API requires temperature=1 for reasoning models
		TopP:              req.TopP,
		MaxOutputTokens:   &maxOutputTokens,
		Tools:             tools,
		ToolChoice:        toolChoice,
		Metadata:          req.Metadata,
		Stream:            req.Stream,
		Store:             false,
		ParallelToolCalls: &ptc,
		Reasoning:         &ResponseReasoning{Effort: resolveReasoningEffort(req.Thinking, req.OutputConfig), Summary: "detailed"},
		Include:           []string{"reasoning.encrypted_content"},
	}

	return result, nil
}

// --- Message → Input Item conversion ---

func convertMessagesToResponsesInput(messages []AnthropicMessage, model string) ([]ResponseInputItem, error) {
	var input []ResponseInputItem
	for _, msg := range messages {
		items, err := convertMessageToResponsesInputItems(msg, model)
		if err != nil {
			return nil, err
		}
		input = append(input, items...)
	}
	return input, nil
}

func convertMessageToResponsesInputItems(msg AnthropicMessage, model string) ([]ResponseInputItem, error) {
	switch msg.Role {
	case "user":
		return convertUserMessageToResponsesInput(msg)
	case "assistant":
		return convertAssistantMessageToResponsesInput(msg, model)
	default:
		return nil, fmt.Errorf("unsupported message role: %q (expected \"user\" or \"assistant\")", msg.Role)
	}
}

func convertUserMessageToResponsesInput(msg AnthropicMessage) ([]ResponseInputItem, error) {
	if msg.Content.Text != nil {
		return []ResponseInputItem{{
			Type:    "message",
			Role:    "user",
			Content: *msg.Content.Text,
		}}, nil
	}

	var items []ResponseInputItem
	var pendingContent []ResponseInputContent

	for _, block := range msg.Content.Blocks {
		if block.Type == "tool_result" {
			if len(pendingContent) > 0 {
				items = append(items, createResponsesInputMessage("user", pendingContent, ""))
				pendingContent = nil
			}
			items = append(items, createResponsesFunctionCallOutput(block))
			continue
		}

		if converted := convertUserBlockToResponsesContent(block); converted != nil {
			pendingContent = append(pendingContent, *converted)
		}
	}

	if len(pendingContent) > 0 {
		items = append(items, createResponsesInputMessage("user", pendingContent, ""))
	}

	return items, nil
}

func convertAssistantMessageToResponsesInput(msg AnthropicMessage, model string) ([]ResponseInputItem, error) {
	phase := resolveAssistantPhase(model, msg.Content)

	if msg.Content.Text != nil {
		return []ResponseInputItem{{
			Type:    "message",
			Role:    "assistant",
			Content: *msg.Content.Text,
			Phase:   phase,
		}}, nil
	}

	var items []ResponseInputItem
	var pendingContent []ResponseInputContent

	for _, block := range msg.Content.Blocks {
		if block.Type == "tool_use" {
			if len(pendingContent) > 0 {
				items = append(items, createResponsesInputMessage("assistant", pendingContent, phase))
				pendingContent = nil
			}
			items = append(items, createResponsesFunctionToolCall(block))
			continue
		}

		if block.Type == "thinking" && block.Signature != "" && strings.Contains(block.Signature, "@") {
			if len(pendingContent) > 0 {
				items = append(items, createResponsesInputMessage("assistant", pendingContent, phase))
				pendingContent = nil
			}
			items = append(items, createResponsesReasoningItem(block))
			continue
		}

		if converted := convertAssistantBlockToResponsesContent(block); converted != nil {
			pendingContent = append(pendingContent, *converted)
		}
	}

	if len(pendingContent) > 0 {
		items = append(items, createResponsesInputMessage("assistant", pendingContent, phase))
	}

	return items, nil
}

// --- Helper functions ---

func createResponsesInputMessage(role string, content []ResponseInputContent, phase string) ResponseInputItem {
	item := ResponseInputItem{
		Type:    "message",
		Role:    role,
		Content: content,
	}
	if role == "assistant" && phase != "" {
		item.Phase = phase
	}
	return item
}

func createResponsesFunctionCallOutput(block AnthropicContentBlock) ResponseInputItem {
	output := convertToolResultContentToResponses(block)
	status := "completed"
	if block.IsError != nil && *block.IsError {
		status = "incomplete"
	}
	return ResponseInputItem{
		Type:   "function_call_output",
		CallID: block.ToolUseID,
		Output: output,
		Status: status,
	}
}

func createResponsesFunctionToolCall(block AnthropicContentBlock) ResponseInputItem {
	args := "{}"
	if block.Input != nil {
		if data, err := json.Marshal(block.Input); err == nil {
			args = string(data)
		}
	}
	return ResponseInputItem{
		Type:      "function_call",
		CallID:    block.ID,
		Name:      block.Name,
		Arguments: args,
		Status:    "completed",
	}
}

func createResponsesReasoningItem(block AnthropicContentBlock) ResponseInputItem {
	parts := strings.SplitN(block.Signature, "@", 2)
	encryptedContent := parts[0]
	id := ""
	if len(parts) > 1 {
		id = parts[1]
	}

	thinking := block.Thinking

	var summary *[]ResponseSummaryBlock
	if thinking != "" {
		blocks := []ResponseSummaryBlock{{Type: "summary_text", Text: thinking}}
		summary = &blocks
	} else {
		empty := []ResponseSummaryBlock{}
		summary = &empty
	}

	return ResponseInputItem{
		Type:             "reasoning",
		ID:               id,
		Summary:          summary,
		EncryptedContent: encryptedContent,
	}
}

func convertUserBlockToResponsesContent(block AnthropicContentBlock) *ResponseInputContent {
	switch block.Type {
	case "text":
		return &ResponseInputContent{Type: "input_text", Text: block.Text}
	case "image":
		if block.Source != nil {
			url := imageDataURL(block.Source.MediaType, block.Source.Data)
			return &ResponseInputContent{Type: "input_image", ImageURL: url, Detail: "auto"}
		}
	}
	return nil
}

func convertAssistantBlockToResponsesContent(block AnthropicContentBlock) *ResponseInputContent {
	if block.Type == "text" {
		return &ResponseInputContent{Type: "output_text", Text: block.Text}
	}
	return nil
}

func convertToolResultContentToResponses(block AnthropicContentBlock) interface{} {
	if block.Content == nil {
		return ""
	}
	if block.Content.Text != nil {
		return *block.Content.Text
	}
	if len(block.Content.Blocks) > 0 {
		var result []ResponseInputContent
		for _, b := range block.Content.Blocks {
			switch b.Type {
			case "text":
				result = append(result, ResponseInputContent{Type: "input_text", Text: b.Text})
			case "image":
				if b.Source != nil {
					url := imageDataURL(b.Source.MediaType, b.Source.Data)
					result = append(result, ResponseInputContent{Type: "input_image", ImageURL: url, Detail: "auto"})
				}
			}
		}
		return result
	}
	return ""
}

func resolveAssistantPhase(model string, content AnthropicContent) string {
	if model != codexPhaseModel {
		return ""
	}
	if content.Text != nil {
		return "final_answer"
	}
	if len(content.Blocks) == 0 {
		return ""
	}
	hasText := false
	hasToolUse := false
	for _, b := range content.Blocks {
		if b.Type == "text" {
			hasText = true
		}
		if b.Type == "tool_use" {
			hasToolUse = true
		}
	}
	if !hasText {
		return ""
	}
	if hasToolUse {
		return "commentary"
	}
	return "final_answer"
}

// --- Tools conversion ---

func convertToolsToResponsesFormat(tools []AnthropicTool) []ResponseTool {
	if len(tools) == 0 {
		return nil
	}
	defs := extractToolDefs(tools)
	result := make([]ResponseTool, len(defs))
	for i, d := range defs {
		result[i] = ResponseTool{
			Type:        "function",
			Name:        d.Name,
			Description: d.Description,
			Parameters:  d.InputSchema,
			Strict:      false,
		}
	}
	return result
}

func convertToolChoiceToResponsesFormat(choice *AnthropicToolChoice) interface{} {
	tc := mapToolChoice(choice)
	switch tc.Mode {
	case "auto":
		return "auto"
	case "required":
		return "required"
	case "function":
		return map[string]string{"type": "function", "name": tc.FunctionName}
	case "none":
		return "none"
	default:
		return "auto"
	}
}

// --- Responses → Anthropic non-streaming translation ---

// ConvertResponsesToAnthropic converts a Responses API result to an Anthropic Messages response.
func ConvertResponsesToAnthropic(result ResponsesResult) AnthropicMessagesResponse {
	contentBlocks := mapResponsesOutputToAnthropicContent(result.Output)

	if len(contentBlocks) == 0 && result.OutputText != "" {
		contentBlocks = []AnthropicContentBlock{{Type: "text", Text: result.OutputText}}
	}

	if len(contentBlocks) == 0 {
		contentBlocks = []AnthropicContentBlock{{Type: "text", Text: ""}}
	}

	stopReason := mapResponsesStopReason(result)
	usage := mapResponsesUsage(result)

	return AnthropicMessagesResponse{
		ID:         result.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      result.Model,
		Content:    contentBlocks,
		StopReason: stopReason,
		Usage:      usage,
	}
}

func mapResponsesOutputToAnthropicContent(output []ResponseOutputItem) []AnthropicContentBlock {
	var blocks []AnthropicContentBlock

	for _, item := range output {
		switch item.Type {
		case "reasoning":
			thinkingText := extractResponsesReasoningText(item)
			signature := item.EncryptedContent + "@" + item.ID
			blocks = append(blocks, AnthropicContentBlock{
				Type:      "thinking",
				Thinking:  thinkingText,
				Signature: signature,
			})
		case "function_call":
			if item.Name != "" && item.CallID != "" {
				input := parseResponsesFunctionCallArguments(item.Arguments)
				blocks = append(blocks, AnthropicContentBlock{
					Type:  "tool_use",
					ID:    item.CallID,
					Name:  item.Name,
					Input: input,
				})
			}
		case "message":
			combinedText := combineResponsesMessageTextContent(item.Content)
			if combinedText != "" {
				blocks = append(blocks, AnthropicContentBlock{
					Type: "text",
					Text: combinedText,
				})
			}
		default:
			combinedText := combineResponsesMessageTextContent(item.Content)
			if combinedText != "" {
				blocks = append(blocks, AnthropicContentBlock{
					Type: "text",
					Text: combinedText,
				})
			}
		}
	}

	return blocks
}

func extractResponsesReasoningText(item ResponseOutputItem) string {
	if len(item.Summary) == 0 {
		return ""
	}
	var parts []string
	for _, block := range item.Summary {
		if block.Text != "" {
			parts = append(parts, block.Text)
		}
	}
	return strings.TrimSpace(strings.Join(parts, ""))
}

func combineResponsesMessageTextContent(content []ResponseOutputContent) string {
	var sb strings.Builder
	for _, block := range content {
		switch block.Type {
		case "output_text":
			sb.WriteString(block.Text)
		case "refusal":
			sb.WriteString(block.Refusal)
		default:
			if block.Text != "" {
				sb.WriteString(block.Text)
			}
		}
	}
	return sb.String()
}

func parseResponsesFunctionCallArguments(rawArgs string) map[string]interface{} {
	rawArgs = strings.TrimSpace(rawArgs)
	if rawArgs == "" {
		return map[string]interface{}{}
	}

	var parsed interface{}
	if err := json.Unmarshal([]byte(rawArgs), &parsed); err != nil {
		return map[string]interface{}{"raw_arguments": rawArgs}
	}

	switch v := parsed.(type) {
	case map[string]interface{}:
		return v
	case []interface{}:
		return map[string]interface{}{"arguments": v}
	default:
		return map[string]interface{}{"raw_arguments": rawArgs}
	}
}

func mapResponsesStopReason(result ResponsesResult) string {
	switch result.Status {
	case "completed":
		for _, item := range result.Output {
			if item.Type == "function_call" {
				return "tool_use"
			}
		}
		return "end_turn"
	case "incomplete":
		if result.IncompleteDetails != nil {
			if result.IncompleteDetails.Reason == "max_output_tokens" {
				return "max_tokens"
			}
		}
		return "end_turn"
	default:
		return ""
	}
}

func mapResponsesUsage(result ResponsesResult) AnthropicUsage {
	usage := AnthropicUsage{}
	if result.Usage != nil {
		usage.InputTokens = result.Usage.InputTokens
		usage.OutputTokens = result.Usage.OutputTokens
		if result.Usage.InputTokensDetails != nil {
			usage.InputTokens, usage.CacheReadInputTokens = adjustCachedUsage(usage.InputTokens, result.Usage.InputTokensDetails.CachedTokens)
		}
	}
	return usage
}

func ptrFloat64(v float64) *float64 { return &v }
