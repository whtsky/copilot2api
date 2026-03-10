package anthropic

import (
	"encoding/json"
	"testing"
)

func TestConvertAnthropicToOpenAI_SimpleText(t *testing.T) {
	anthropicReq := AnthropicMessagesRequest{
		Model:     "claude-3-sonnet-20240229",
		MaxTokens: 1000,
		Messages: []AnthropicMessage{
			{
				Role:    "user",
				Content: AnthropicContent{Text: stringPtr("Hello!")},
			},
		},
		Temperature: floatPtr(0.7),
	}

	openAIReq, err := ConvertAnthropicToOpenAI(anthropicReq)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if openAIReq.Model != "claude-3-sonnet-20240229" {
		t.Errorf("Expected model 'claude-3-sonnet-20240229', got %q", openAIReq.Model)
	}

	if *openAIReq.MaxTokens != 1000 {
		t.Errorf("Expected max_tokens 1000, got %d", *openAIReq.MaxTokens)
	}

	if len(openAIReq.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(openAIReq.Messages))
	}

	msg := openAIReq.Messages[0]
	if msg.Role != "user" {
		t.Errorf("Expected role 'user', got %q", msg.Role)
	}

	if msg.Content == nil || msg.Content.Text == nil || *msg.Content.Text != "Hello!" {
		t.Errorf("Message content not converted correctly")
	}

	if *openAIReq.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", *openAIReq.Temperature)
	}
}

func TestConvertAnthropicToOpenAI_WithSystem(t *testing.T) {
	anthropicReq := AnthropicMessagesRequest{
		Model:     "claude-3-sonnet-20240229",
		MaxTokens: 1000,
		System: &AnthropicSystem{
			Text: stringPtr("You are a helpful assistant"),
		},
		Messages: []AnthropicMessage{
			{
				Role:    "user",
				Content: AnthropicContent{Text: stringPtr("Hello!")},
			},
		},
	}

	openAIReq, err := ConvertAnthropicToOpenAI(anthropicReq)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if len(openAIReq.Messages) != 2 {
		t.Fatalf("Expected 2 messages (system + user), got %d", len(openAIReq.Messages))
	}

	systemMsg := openAIReq.Messages[0]
	if systemMsg.Role != "system" {
		t.Errorf("Expected first message role 'system', got %q", systemMsg.Role)
	}

	if systemMsg.Content == nil || systemMsg.Content.Text == nil || *systemMsg.Content.Text != "You are a helpful assistant" {
		t.Errorf("System message not converted correctly")
	}

	userMsg := openAIReq.Messages[1]
	if userMsg.Role != "user" {
		t.Errorf("Expected second message role 'user', got %q", userMsg.Role)
	}
}

func TestConvertAnthropicToOpenAI_ToolUse(t *testing.T) {
	anthropicReq := AnthropicMessagesRequest{
		Model:     "claude-3-sonnet-20240229",
		MaxTokens: 1000,
		Messages: []AnthropicMessage{
			{
				Role: "assistant",
				Content: AnthropicContent{
					Blocks: []AnthropicContentBlock{
						{
							Type: "text",
							Text: "I'll search for that information.",
						},
						{
							Type: "tool_use",
							ID:   "tool_1",
							Name: "search",
							Input: map[string]interface{}{
								"query": "weather today",
							},
						},
					},
				},
			},
		},
		Tools: []AnthropicTool{
			{
				Name:        "search",
				Description: "Search for information",
				InputSchema: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type": "string",
						},
					},
				},
			},
		},
		ToolChoice: &AnthropicToolChoice{
			Type: "auto",
		},
	}

	openAIReq, err := ConvertAnthropicToOpenAI(anthropicReq)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	// Check tools
	if len(openAIReq.Tools) != 1 {
		t.Fatalf("Expected 1 tool, got %d", len(openAIReq.Tools))
	}

	tool := openAIReq.Tools[0]
	if tool.Type != "function" {
		t.Errorf("Expected tool type 'function', got %q", tool.Type)
	}

	if tool.Function.Name != "search" {
		t.Errorf("Expected function name 'search', got %q", tool.Function.Name)
	}

	// Check tool choice
	if openAIReq.ToolChoice != "auto" {
		t.Errorf("Expected tool choice 'auto', got %v", openAIReq.ToolChoice)
	}

	// Check message
	if len(openAIReq.Messages) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(openAIReq.Messages))
	}

	msg := openAIReq.Messages[0]
	if msg.Role != "assistant" {
		t.Errorf("Expected role 'assistant', got %q", msg.Role)
	}

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("Expected 1 tool call, got %d", len(msg.ToolCalls))
	}

	toolCall := msg.ToolCalls[0]
	if toolCall.ID != "tool_1" {
		t.Errorf("Expected tool call ID 'tool_1', got %q", toolCall.ID)
	}

	if toolCall.Function.Name != "search" {
		t.Errorf("Expected function name 'search', got %q", toolCall.Function.Name)
	}

	var args map[string]interface{}
	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
		t.Fatalf("Failed to parse tool call arguments: %v", err)
	}

	if args["query"] != "weather today" {
		t.Errorf("Expected query 'weather today', got %v", args["query"])
	}
}

func TestConvertAnthropicToOpenAI_ToolResult(t *testing.T) {
	anthropicReq := AnthropicMessagesRequest{
		Model:     "claude-3-sonnet-20240229",
		MaxTokens: 1000,
		Messages: []AnthropicMessage{
			{
				Role: "user",
				Content: AnthropicContent{
					Blocks: []AnthropicContentBlock{
						{
							Type:      "tool_result",
							ToolUseID: "tool_1",
							Content: &AnthropicContent{
								Text: stringPtr("The weather is sunny."),
							},
						},
						{
							Type: "text",
							Text: "What should I wear?",
						},
					},
				},
			},
		},
	}

	openAIReq, err := ConvertAnthropicToOpenAI(anthropicReq)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	// Should create 2 messages: tool message + user message
	if len(openAIReq.Messages) != 2 {
		t.Fatalf("Expected 2 messages, got %d", len(openAIReq.Messages))
	}

	// Tool message should come first
	toolMsg := openAIReq.Messages[0]
	if toolMsg.Role != "tool" {
		t.Errorf("Expected first message role 'tool', got %q", toolMsg.Role)
	}

	if toolMsg.ToolCallID != "tool_1" {
		t.Errorf("Expected tool_call_id 'tool_1', got %q", toolMsg.ToolCallID)
	}

	if toolMsg.Content == nil || toolMsg.Content.Text == nil || *toolMsg.Content.Text != "The weather is sunny." {
		t.Errorf("Tool result content not converted correctly")
	}

	// User message should come second
	userMsg := openAIReq.Messages[1]
	if userMsg.Role != "user" {
		t.Errorf("Expected second message role 'user', got %q", userMsg.Role)
	}

	if userMsg.Content == nil || userMsg.Content.Text == nil || *userMsg.Content.Text != "What should I wear?" {
		t.Errorf("User message content not converted correctly")
	}
}

func TestConvertAnthropicToOpenAI_ToolChoice(t *testing.T) {
	tests := []struct {
		name             string
		toolChoice       *AnthropicToolChoice
		expectedChoice   interface{}
		expectedParallel *bool
	}{
		{
			name:           "auto",
			toolChoice:     &AnthropicToolChoice{Type: "auto"},
			expectedChoice: "auto",
		},
		{
			name:           "any (required)",
			toolChoice:     &AnthropicToolChoice{Type: "any"},
			expectedChoice: "required",
		},
		{
			name: "specific tool",
			toolChoice: &AnthropicToolChoice{
				Type: "tool",
				Name: "search",
			},
			expectedChoice: map[string]interface{}{
				"type": "function",
				"function": map[string]string{
					"name": "search",
				},
			},
		},
		{
			name:           "none",
			toolChoice:     &AnthropicToolChoice{Type: "none"},
			expectedChoice: "none",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			anthropicReq := AnthropicMessagesRequest{
				Model:      "claude-3-sonnet-20240229",
				MaxTokens:  1000,
				Messages:   []AnthropicMessage{{Role: "user", Content: AnthropicContent{Text: stringPtr("test")}}},
				ToolChoice: tt.toolChoice,
			}

			openAIReq, err := ConvertAnthropicToOpenAI(anthropicReq)
			if err != nil {
				t.Fatalf("Conversion failed: %v", err)
			}

			// Compare tool choice values
			expectedJSON, _ := json.Marshal(tt.expectedChoice)
			actualJSON, _ := json.Marshal(openAIReq.ToolChoice)

			if string(expectedJSON) != string(actualJSON) {
				t.Errorf("Expected tool choice %s, got %s", string(expectedJSON), string(actualJSON))
			}

			// Check parallel tool calls
			if tt.expectedParallel != nil {
				if openAIReq.ParallelToolCalls == nil || *openAIReq.ParallelToolCalls != *tt.expectedParallel {
					t.Errorf("Expected parallel tool calls %v, got %v", *tt.expectedParallel, openAIReq.ParallelToolCalls)
				}
			}
		})
	}
}

func TestConvertOpenAIToAnthropic_SimpleResponse(t *testing.T) {
	openAIResp := OpenAIChatCompletionsResponse{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIMessage{
					Role: "assistant",
					Content: &OpenAIContent{
						Text: stringPtr("Hello! How can I help you today?"),
					},
				},
				FinishReason: "stop",
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 12,
			TotalTokens:      22,
		},
	}

	anthropicResp, err := ConvertOpenAIToAnthropic(openAIResp)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if anthropicResp.ID != "msg_123" {
		t.Errorf("Expected ID 'msg_123', got %q", anthropicResp.ID)
	}

	if anthropicResp.Model != "claude-3-sonnet-20240229" {
		t.Errorf("Expected model 'claude-3-sonnet-20240229', got %q", anthropicResp.Model)
	}

	if anthropicResp.StopReason != "end_turn" {
		t.Errorf("Expected stop_reason 'end_turn', got %q", anthropicResp.StopReason)
	}

	if len(anthropicResp.Content) != 1 {
		t.Fatalf("Expected 1 content block, got %d", len(anthropicResp.Content))
	}

	block := anthropicResp.Content[0]
	if block.Type != "text" {
		t.Errorf("Expected content type 'text', got %q", block.Type)
	}

	if block.Text != "Hello! How can I help you today?" {
		t.Errorf("Expected text 'Hello! How can I help you today?', got %q", block.Text)
	}

	if anthropicResp.Usage.InputTokens != 10 {
		t.Errorf("Expected input tokens 10, got %d", anthropicResp.Usage.InputTokens)
	}

	if anthropicResp.Usage.OutputTokens != 12 {
		t.Errorf("Expected output tokens 12, got %d", anthropicResp.Usage.OutputTokens)
	}
}

func TestConvertOpenAIToAnthropic_WithToolCalls(t *testing.T) {
	openAIResp := OpenAIChatCompletionsResponse{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIMessage{
					Role: "assistant",
					Content: &OpenAIContent{
						Text: stringPtr("I'll search for that information."),
					},
					ToolCalls: []OpenAIToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: OpenAIToolCallFunction{
								Name:      "search",
								Arguments: `{"query": "weather today"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     20,
			CompletionTokens: 15,
			TotalTokens:      35,
		},
	}

	anthropicResp, err := ConvertOpenAIToAnthropic(openAIResp)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if anthropicResp.StopReason != "tool_use" {
		t.Errorf("Expected stop_reason 'tool_use', got %q", anthropicResp.StopReason)
	}

	if len(anthropicResp.Content) != 2 {
		t.Fatalf("Expected 2 content blocks, got %d", len(anthropicResp.Content))
	}

	// First block should be text
	textBlock := anthropicResp.Content[0]
	if textBlock.Type != "text" {
		t.Errorf("Expected first block type 'text', got %q", textBlock.Type)
	}

	if textBlock.Text != "I'll search for that information." {
		t.Errorf("Expected text 'I'll search for that information.', got %q", textBlock.Text)
	}

	// Second block should be tool_use
	toolBlock := anthropicResp.Content[1]
	if toolBlock.Type != "tool_use" {
		t.Errorf("Expected second block type 'tool_use', got %q", toolBlock.Type)
	}

	if toolBlock.ID != "call_1" {
		t.Errorf("Expected tool use ID 'call_1', got %q", toolBlock.ID)
	}

	if toolBlock.Name != "search" {
		t.Errorf("Expected tool use name 'search', got %q", toolBlock.Name)
	}

	if toolBlock.Input["query"] != "weather today" {
		t.Errorf("Expected query 'weather today', got %v", toolBlock.Input["query"])
	}
}

func TestMapOpenAIFinishReasonToAnthropic(t *testing.T) {
	tests := []struct {
		openAI    string
		anthropic string
	}{
		{"stop", "end_turn"},
		{"length", "max_tokens"},
		{"tool_calls", "tool_use"},
		{"content_filter", "end_turn"},
		{"unknown", "end_turn"},
		{"", "end_turn"},
	}

	for _, tt := range tests {
		t.Run(tt.openAI, func(t *testing.T) {
			result := mapOpenAIFinishReasonToAnthropic(tt.openAI)
			if result != tt.anthropic {
				t.Errorf("Expected %q, got %q", tt.anthropic, result)
			}
		})
	}
}

// Helper functions for tests
func floatPtr(f float64) *float64 {
	return &f
}

func TestConvertOpenAIToAnthropic_SplitChoices(t *testing.T) {
	// Copilot API returns split choices for Claude models:
	// text in choice 0, tool_calls in choice 1
	openAIResp := OpenAIChatCompletionsResponse{
		ID:    "msg_split",
		Model: "claude-opus-4.6",
		Choices: []OpenAIChoice{
			{
				Index: 0,
				Message: OpenAIMessage{
					Role: "assistant",
					Content: &OpenAIContent{
						Text: stringPtr("Let me search for that."),
					},
				},
				FinishReason: "stop",
			},
			{
				Index: 1,
				Message: OpenAIMessage{
					Role: "assistant",
					ToolCalls: []OpenAIToolCall{
						{
							ID:   "call_abc",
							Type: "function",
							Function: OpenAIToolCallFunction{
								Name:      "web_search",
								Arguments: `{"query": "latest news"}`,
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     100,
			CompletionTokens: 50,
			TotalTokens:      150,
		},
	}

	anthropicResp, err := ConvertOpenAIToAnthropic(openAIResp)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	// Should merge both choices: text + tool_use
	if len(anthropicResp.Content) != 2 {
		t.Fatalf("Expected 2 content blocks (text + tool_use), got %d", len(anthropicResp.Content))
	}

	if anthropicResp.Content[0].Type != "text" {
		t.Errorf("Expected first block type 'text', got %q", anthropicResp.Content[0].Type)
	}
	if anthropicResp.Content[0].Text != "Let me search for that." {
		t.Errorf("Expected text content, got %q", anthropicResp.Content[0].Text)
	}

	if anthropicResp.Content[1].Type != "tool_use" {
		t.Errorf("Expected second block type 'tool_use', got %q", anthropicResp.Content[1].Type)
	}
	if anthropicResp.Content[1].ID != "call_abc" {
		t.Errorf("Expected tool call ID 'call_abc', got %q", anthropicResp.Content[1].ID)
	}
	if anthropicResp.Content[1].Name != "web_search" {
		t.Errorf("Expected tool name 'web_search', got %q", anthropicResp.Content[1].Name)
	}

	// tool_calls finish_reason should win over stop
	if anthropicResp.StopReason != "tool_use" {
		t.Errorf("Expected stop_reason 'tool_use', got %q", anthropicResp.StopReason)
	}
}

