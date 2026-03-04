package anthropic

import (
	"encoding/json"
	"testing"
)

func TestConvertAnthropicToResponses_SimpleText(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:     "claude-3.5-sonnet",
		MaxTokens: 4096,
		System:    &AnthropicSystem{Text: stringPtr("You are helpful")},
		Messages: []AnthropicMessage{
			{Role: "user", Content: AnthropicContent{Text: stringPtr("Hello!")}},
		},
		Stream: true,
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Model != "claude-3.5-sonnet" {
		t.Errorf("model = %q, want %q", result.Model, "claude-3.5-sonnet")
	}
	if result.MaxOutputTokens != 12800 {
		t.Errorf("max_output_tokens = %d, want %d", result.MaxOutputTokens, 12800)
	}
	if result.Instructions == nil || *result.Instructions != "You are helpful" {
		t.Errorf("instructions not set correctly")
	}
	if result.Temperature != 1 {
		t.Errorf("temperature = %f, want 1", result.Temperature)
	}
	if !result.Stream {
		t.Error("expected stream = true")
	}
	if result.Store {
		t.Error("expected store = false")
	}
	if result.Reasoning == nil || result.Reasoning.Effort != "high" || result.Reasoning.Summary != "detailed" {
		t.Errorf("reasoning not set correctly: %+v", result.Reasoning)
	}
	if len(result.Include) != 1 || result.Include[0] != "reasoning.encrypted_content" {
		t.Errorf("include not set correctly: %v", result.Include)
	}
	if len(result.Input) != 1 {
		t.Fatalf("expected 1 input item, got %d", len(result.Input))
	}
	if result.Input[0].Type != "message" || result.Input[0].Role != "user" {
		t.Errorf("input[0] = %+v, want message/user", result.Input[0])
	}
}

func TestConvertAnthropicToResponses_MaxTokensFloor(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:     "test-model",
		MaxTokens: 100,
		Messages:  []AnthropicMessage{{Role: "user", Content: AnthropicContent{Text: stringPtr("Hi")}}},
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.MaxOutputTokens != 12800 {
		t.Errorf("max_output_tokens = %d, want 12800 (floor)", result.MaxOutputTokens)
	}
}

func TestResolveReasoningEffort(t *testing.T) {
	budget := func(n int) *AnthropicThinking {
		return &AnthropicThinking{Type: "enabled", BudgetTokens: &n}
	}
	effort := func(s string) *AnthropicOutputConfig {
		return &AnthropicOutputConfig{Effort: s}
	}

	tests := []struct {
		name     string
		thinking *AnthropicThinking
		output   *AnthropicOutputConfig
		want     string
	}{
		{"nil both", nil, nil, "high"},
		{"thinking low budget", budget(4000), nil, "low"},
		{"thinking medium budget", budget(10000), nil, "medium"},
		{"thinking high budget", budget(20000), nil, "high"},
		{"output_config low", nil, effort("low"), "low"},
		{"output_config medium", nil, effort("medium"), "medium"},
		{"output_config high", nil, effort("high"), "high"},
		{"output_config max maps to high", nil, effort("max"), "high"},
		{"output_config overrides thinking", budget(4000), effort("high"), "high"},
		{"output_config empty falls back to thinking", budget(4000), &AnthropicOutputConfig{}, "low"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveReasoningEffort(tt.thinking, tt.output)
			if got != tt.want {
				t.Errorf("resolveReasoningEffort() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestConvertAnthropicToResponses_OutputConfigEffort(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:        "test-model",
		MaxTokens:    4096,
		Messages:     []AnthropicMessage{{Role: "user", Content: AnthropicContent{Text: stringPtr("Hi")}}},
		OutputConfig: &AnthropicOutputConfig{Effort: "low"},
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Reasoning == nil || result.Reasoning.Effort != "low" {
		t.Errorf("reasoning effort = %v, want low", result.Reasoning)
	}
}

func TestConvertAnthropicToResponses_LargeMaxTokens(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:     "test-model",
		MaxTokens: 50000,
		Messages:  []AnthropicMessage{{Role: "user", Content: AnthropicContent{Text: stringPtr("Hi")}}},
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.MaxOutputTokens != 50000 {
		t.Errorf("max_output_tokens = %d, want 50000", result.MaxOutputTokens)
	}
}

func TestConvertAnthropicToResponses_ToolUse(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:     "claude-3.5-sonnet",
		MaxTokens: 4096,
		Messages: []AnthropicMessage{
			{Role: "user", Content: AnthropicContent{Text: stringPtr("Search for go")}},
			{
				Role: "assistant",
				Content: AnthropicContent{
					Blocks: []AnthropicContentBlock{
						{Type: "text", Text: "I'll search for that."},
						{Type: "tool_use", ID: "call_1", Name: "search", Input: map[string]interface{}{"query": "go"}},
					},
				},
			},
			{
				Role: "user",
				Content: AnthropicContent{
					Blocks: []AnthropicContentBlock{
						{Type: "tool_result", ToolUseID: "call_1", Content: &AnthropicContent{Text: stringPtr("Go is a programming language")}},
					},
				},
			},
		},
		Tools: []AnthropicTool{
			{Name: "search", Description: "Search", InputSchema: map[string]interface{}{"type": "object"}},
		},
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check tools
	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}
	if result.Tools[0].Name != "search" || result.Tools[0].Type != "function" {
		t.Errorf("tool[0] = %+v", result.Tools[0])
	}

	// Check input items
	if len(result.Input) != 4 {
		t.Fatalf("expected 4 input items, got %d", len(result.Input))
	}
	// user message
	if result.Input[0].Type != "message" || result.Input[0].Role != "user" {
		t.Errorf("input[0] type=%q role=%q", result.Input[0].Type, result.Input[0].Role)
	}
	// assistant message with text
	if result.Input[1].Type != "message" || result.Input[1].Role != "assistant" {
		t.Errorf("input[1] type=%q role=%q", result.Input[1].Type, result.Input[1].Role)
	}
	// function_call
	if result.Input[2].Type != "function_call" || result.Input[2].Name != "search" || result.Input[2].CallID != "call_1" {
		t.Errorf("input[2] = %+v", result.Input[2])
	}
	// function_call_output
	if result.Input[3].Type != "function_call_output" || result.Input[3].CallID != "call_1" {
		t.Errorf("input[3] = %+v", result.Input[3])
	}
}

func TestConvertAnthropicToResponses_ThinkingBlock(t *testing.T) {
	req := AnthropicMessagesRequest{
		Model:     "claude-3.5-sonnet",
		MaxTokens: 4096,
		Messages: []AnthropicMessage{
			{
				Role: "assistant",
				Content: AnthropicContent{
					Blocks: []AnthropicContentBlock{
						{Type: "thinking", Thinking: "Let me analyze...", Signature: "enc_data@reason_123"},
						{Type: "text", Text: "Here's my answer."},
					},
				},
			},
		},
	}

	result, err := ConvertAnthropicToResponses(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Input) != 2 {
		t.Fatalf("expected 2 input items, got %d", len(result.Input))
	}

	// reasoning item
	if result.Input[0].Type != "reasoning" {
		t.Errorf("input[0] type=%q, want reasoning", result.Input[0].Type)
	}
	if result.Input[0].ID != "reason_123" {
		t.Errorf("input[0] id=%q, want reason_123", result.Input[0].ID)
	}
	if result.Input[0].EncryptedContent != "enc_data" {
		t.Errorf("input[0] encrypted_content=%q, want enc_data", result.Input[0].EncryptedContent)
	}

	// assistant message with text
	if result.Input[1].Type != "message" || result.Input[1].Role != "assistant" {
		t.Errorf("input[1] type=%q role=%q", result.Input[1].Type, result.Input[1].Role)
	}
}

func TestConvertAnthropicToResponses_ToolChoice(t *testing.T) {
	tests := []struct {
		name     string
		choice   *AnthropicToolChoice
		expected string
	}{
		{"nil", nil, `"auto"`},
		{"auto", &AnthropicToolChoice{Type: "auto"}, `"auto"`},
		{"any", &AnthropicToolChoice{Type: "any"}, `"required"`},
		{"none", &AnthropicToolChoice{Type: "none"}, `"none"`},
		{"tool", &AnthropicToolChoice{Type: "tool", Name: "search"}, `{"name":"search","type":"function"}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToolChoiceToResponsesFormat(tt.choice)
			data, _ := json.Marshal(result)
			if string(data) != tt.expected {
				t.Errorf("got %s, want %s", string(data), tt.expected)
			}
		})
	}
}

func TestConvertAnthropicToResponses_SystemBlocks(t *testing.T) {
	system := &AnthropicSystem{
		Blocks: []AnthropicSystemBlock{
			{Type: "text", Text: "Part one."},
			{Type: "text", Text: "Part two."},
		},
	}

	result := extractSystemText(system)
	if result == nil || *result != "Part one.\n\nPart two." {
		t.Errorf("instructions = %v, want 'Part one.\\n\\nPart two.'", result)
	}
}

func TestConvertAnthropicToResponses_CodexPhase(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		content  AnthropicContent
		expected string
	}{
		{"non-codex model", "claude-3.5-sonnet", AnthropicContent{Text: stringPtr("Hello")}, ""},
		{"codex string", codexPhaseModel, AnthropicContent{Text: stringPtr("Hello")}, "final_answer"},
		{"codex text only", codexPhaseModel, AnthropicContent{Blocks: []AnthropicContentBlock{{Type: "text", Text: "Hi"}}}, "final_answer"},
		{"codex text+tool", codexPhaseModel, AnthropicContent{Blocks: []AnthropicContentBlock{{Type: "text", Text: "Hi"}, {Type: "tool_use"}}}, "commentary"},
		{"codex no text", codexPhaseModel, AnthropicContent{Blocks: []AnthropicContentBlock{{Type: "tool_use"}}}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := resolveAssistantPhase(tt.model, tt.content)
			if result != tt.expected {
				t.Errorf("resolveAssistantPhase() = %q, want %q", result, tt.expected)
			}
		})
	}
}

// --- Responses → Anthropic non-streaming tests ---

func TestConvertResponsesToAnthropic_SimpleText(t *testing.T) {
	result := ResponsesResult{
		ID:     "resp_123",
		Model:  "claude-3.5-sonnet",
		Status: "completed",
		Output: []ResponseOutputItem{
			{
				Type: "message",
				Content: []ResponseOutputContent{
					{Type: "output_text", Text: "Hello! How can I help?"},
				},
			},
		},
		Usage: &ResponsesUsage{
			InputTokens:  10,
			OutputTokens: 8,
		},
	}

	resp := ConvertResponsesToAnthropic(result)

	if resp.ID != "resp_123" {
		t.Errorf("ID = %q, want resp_123", resp.ID)
	}
	if resp.Model != "claude-3.5-sonnet" {
		t.Errorf("Model = %q", resp.Model)
	}
	if resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want end_turn", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "Hello! How can I help?" {
		t.Errorf("Content = %+v", resp.Content)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 8 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
}

func TestConvertResponsesToAnthropic_WithToolUse(t *testing.T) {
	result := ResponsesResult{
		ID:     "resp_456",
		Model:  "claude-3.5-sonnet",
		Status: "completed",
		Output: []ResponseOutputItem{
			{
				Type: "message",
				Content: []ResponseOutputContent{
					{Type: "output_text", Text: "I'll search."},
				},
			},
			{
				Type:      "function_call",
				CallID:    "call_1",
				Name:      "search",
				Arguments: `{"query":"go"}`,
			},
		},
		Usage: &ResponsesUsage{InputTokens: 5, OutputTokens: 10},
	}

	resp := ConvertResponsesToAnthropic(result)

	if resp.StopReason != "tool_use" {
		t.Errorf("StopReason = %q, want tool_use", resp.StopReason)
	}
	if len(resp.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != "text" || resp.Content[0].Text != "I'll search." {
		t.Errorf("Content[0] = %+v", resp.Content[0])
	}
	if resp.Content[1].Type != "tool_use" || resp.Content[1].Name != "search" || resp.Content[1].ID != "call_1" {
		t.Errorf("Content[1] = %+v", resp.Content[1])
	}
}

func TestConvertResponsesToAnthropic_WithReasoning(t *testing.T) {
	result := ResponsesResult{
		ID:     "resp_789",
		Model:  "claude-3.5-sonnet",
		Status: "completed",
		Output: []ResponseOutputItem{
			{
				Type:             "reasoning",
				ID:               "reason_1",
				EncryptedContent: "encrypted_data",
				Summary:          []ResponseSummaryBlock{{Type: "summary_text", Text: "I analyzed the problem."}},
			},
			{
				Type:    "message",
				Content: []ResponseOutputContent{{Type: "output_text", Text: "Here's my answer."}},
			},
		},
		Usage: &ResponsesUsage{InputTokens: 10, OutputTokens: 20},
	}

	resp := ConvertResponsesToAnthropic(result)

	if len(resp.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(resp.Content))
	}
	if resp.Content[0].Type != "thinking" {
		t.Errorf("Content[0].Type = %q, want thinking", resp.Content[0].Type)
	}
	if resp.Content[0].Thinking != "I analyzed the problem." {
		t.Errorf("Content[0].Thinking = %q", resp.Content[0].Thinking)
	}
	if resp.Content[0].Signature != "encrypted_data@reason_1" {
		t.Errorf("Content[0].Signature = %q, want encrypted_data@reason_1", resp.Content[0].Signature)
	}
	if resp.Content[1].Type != "text" || resp.Content[1].Text != "Here's my answer." {
		t.Errorf("Content[1] = %+v", resp.Content[1])
	}
}

func TestConvertResponsesToAnthropic_Incomplete(t *testing.T) {
	result := ResponsesResult{
		ID:                "resp_inc",
		Model:             "test",
		Status:            "incomplete",
		IncompleteDetails: &IncompleteDetails{Reason: "max_output_tokens"},
		Output:            []ResponseOutputItem{{Type: "message", Content: []ResponseOutputContent{{Type: "output_text", Text: "Partial"}}}},
	}

	resp := ConvertResponsesToAnthropic(result)
	if resp.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want max_tokens", resp.StopReason)
	}
}

func TestConvertResponsesToAnthropic_FallbackToOutputText(t *testing.T) {
	result := ResponsesResult{
		ID:         "resp_fb",
		Model:      "test",
		Status:     "completed",
		Output:     []ResponseOutputItem{},
		OutputText: "Fallback text",
	}

	resp := ConvertResponsesToAnthropic(result)
	if len(resp.Content) != 1 || resp.Content[0].Text != "Fallback text" {
		t.Errorf("Content = %+v", resp.Content)
	}
}

func TestConvertResponsesToAnthropic_CachedTokens(t *testing.T) {
	result := ResponsesResult{
		ID:     "resp_cached",
		Model:  "test",
		Status: "completed",
		Output: []ResponseOutputItem{{Type: "message", Content: []ResponseOutputContent{{Type: "output_text", Text: "Hi"}}}},
		Usage: &ResponsesUsage{
			InputTokens:  100,
			OutputTokens: 10,
			InputTokensDetails: &InputTokenDetails{CachedTokens: 30},
		},
	}

	resp := ConvertResponsesToAnthropic(result)
	if resp.Usage.InputTokens != 70 {
		t.Errorf("InputTokens = %d, want 70 (100-30)", resp.Usage.InputTokens)
	}
	if resp.Usage.CacheReadInputTokens != 30 {
		t.Errorf("CacheReadInputTokens = %d, want 30", resp.Usage.CacheReadInputTokens)
	}
}
