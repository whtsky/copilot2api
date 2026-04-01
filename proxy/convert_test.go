package proxy

import (
	"encoding/json"
	"testing"

	"github.com/whtsky/copilot2api/internal/types"
)

// --- Chat → Responses request conversion ---

func TestConvertChatToResponsesRequest_BasicTextMessage(t *testing.T) {
	text := "Hello"
	req := types.OpenAIChatCompletionsRequest{
		Model:  "gpt-4",
		Stream: false,
		Messages: []types.OpenAIMessage{
			{Role: "user", Content: &types.OpenAIContent{Text: &text}},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if result.Model != "gpt-4" {
		t.Errorf("Model = %q, want %q", result.Model, "gpt-4")
	}
	if result.Stream {
		t.Error("Stream should be false")
	}
	if result.Store {
		t.Error("Store should be false")
	}
	if len(result.Input) != 1 {
		t.Fatalf("Input length = %d, want 1", len(result.Input))
	}
	if result.Input[0].Type != "message" || result.Input[0].Role != "user" {
		t.Errorf("Input[0] type=%q role=%q, want type=message role=user", result.Input[0].Type, result.Input[0].Role)
	}
	if contentStr, ok := result.Input[0].Content.(string); !ok || contentStr != "Hello" {
		t.Errorf("Input[0].Content = %v, want %q", result.Input[0].Content, "Hello")
	}
}

func TestConvertChatToResponsesRequest_SystemMessage(t *testing.T) {
	sysText := "You are helpful"
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{Role: "system", Content: &types.OpenAIContent{Text: &sysText}},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if len(result.Input) != 1 {
		t.Fatalf("Input length = %d, want 1", len(result.Input))
	}
	item := result.Input[0]
	if item.Type != "message" || item.Role != "system" {
		t.Errorf("type=%q role=%q, want message/system", item.Type, item.Role)
	}
}

func TestConvertChatToResponsesRequest_AssistantWithToolCalls(t *testing.T) {
	assistText := "Let me check"
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{
				Role:    "assistant",
				Content: &types.OpenAIContent{Text: &assistText},
				ToolCalls: []types.OpenAIToolCall{
					{
						ID:   "call_1",
						Type: "function",
						Function: types.OpenAIToolCallFunction{
							Name:      "get_weather",
							Arguments: `{"city":"NYC"}`,
						},
					},
				},
			},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	// Should produce: assistant message + function_call item
	if len(result.Input) != 2 {
		t.Fatalf("Input length = %d, want 2", len(result.Input))
	}
	if result.Input[0].Type != "message" || result.Input[0].Role != "assistant" {
		t.Errorf("Input[0] type=%q role=%q, want message/assistant", result.Input[0].Type, result.Input[0].Role)
	}
	if result.Input[1].Type != "function_call" {
		t.Errorf("Input[1].Type = %q, want function_call", result.Input[1].Type)
	}
	if result.Input[1].CallID != "call_1" {
		t.Errorf("Input[1].CallID = %q, want call_1", result.Input[1].CallID)
	}
	if result.Input[1].Name != "get_weather" {
		t.Errorf("Input[1].Name = %q, want get_weather", result.Input[1].Name)
	}
}

func TestConvertChatToResponsesRequest_ToolMessage(t *testing.T) {
	toolText := "Sunny, 72F"
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{Role: "tool", ToolCallID: "call_1", Content: &types.OpenAIContent{Text: &toolText}},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if len(result.Input) != 1 {
		t.Fatalf("Input length = %d, want 1", len(result.Input))
	}
	item := result.Input[0]
	if item.Type != "function_call_output" {
		t.Errorf("Type = %q, want function_call_output", item.Type)
	}
	if item.CallID != "call_1" {
		t.Errorf("CallID = %q, want call_1", item.CallID)
	}
	if outputStr, ok := item.Output.(string); !ok || outputStr != "Sunny, 72F" {
		t.Errorf("Output = %v, want %q", item.Output, "Sunny, 72F")
	}
}

func TestConvertChatToResponsesRequest_Tools(t *testing.T) {
	params := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Tools: []types.OpenAIToolDefinition{
			{
				Type: "function",
				Function: types.OpenAIFunctionSpec{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  params,
				},
			},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if len(result.Tools) != 1 {
		t.Fatalf("Tools length = %d, want 1", len(result.Tools))
	}
	tool := result.Tools[0]
	if tool.Type != "function" || tool.Name != "get_weather" {
		t.Errorf("Tool type=%q name=%q", tool.Type, tool.Name)
	}
}

func TestConvertChatToResponsesRequest_Temperature(t *testing.T) {
	temp := 0.7
	req := types.OpenAIChatCompletionsRequest{
		Model:       "gpt-4",
		Temperature: &temp,
	}

	result := ConvertChatToResponsesRequest(req)

	if result.Temperature == nil || *result.Temperature != 0.7 {
		t.Errorf("Temperature = %v, want 0.7", result.Temperature)
	}
}

func TestConvertChatToResponsesRequest_MaxTokens(t *testing.T) {
	maxTok := 1000
	req := types.OpenAIChatCompletionsRequest{
		Model:     "gpt-4",
		MaxTokens: &maxTok,
	}

	result := ConvertChatToResponsesRequest(req)

	if result.MaxOutputTokens == nil || *result.MaxOutputTokens != 1000 {
		t.Errorf("MaxOutputTokens = %v, want 1000", result.MaxOutputTokens)
	}
}

func TestConvertChatToResponsesRequest_ThinkingBudget(t *testing.T) {
	tests := []struct {
		budget int
		effort string
	}{
		{20000, "high"},
		{10000, "medium"},
		{4000, "low"},
		{1000, "low"},
	}

	for _, tt := range tests {
		budget := tt.budget
		req := types.OpenAIChatCompletionsRequest{
			Model:          "gpt-4",
			ThinkingBudget: &budget,
		}

		result := ConvertChatToResponsesRequest(req)

		if result.Reasoning == nil {
			t.Fatalf("budget=%d: Reasoning should not be nil", tt.budget)
		}
		if result.Reasoning.Effort != tt.effort {
			t.Errorf("budget=%d: Effort = %q, want %q", tt.budget, result.Reasoning.Effort, tt.effort)
		}
		if result.Reasoning.Summary != "detailed" {
			t.Errorf("budget=%d: Summary = %q, want detailed", tt.budget, result.Reasoning.Summary)
		}
	}
}

func TestConvertChatToResponsesRequest_UserMultipartContent(t *testing.T) {
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{
				Role: "user",
				Content: &types.OpenAIContent{
					Parts: []types.OpenAIContentPart{
						{Type: "text", Text: "Describe this"},
						{Type: "image_url", ImageURL: &types.OpenAIImageURLPayload{URL: "https://example.com/img.png", Detail: "high"}},
					},
				},
			},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if len(result.Input) != 1 {
		t.Fatalf("Input length = %d, want 1", len(result.Input))
	}
	contentItems, ok := result.Input[0].Content.([]types.ResponseInputContent)
	if !ok {
		t.Fatalf("Content type = %T, want []ResponseInputContent", result.Input[0].Content)
	}
	if len(contentItems) != 2 {
		t.Fatalf("Content items = %d, want 2", len(contentItems))
	}
	if contentItems[0].Type != "input_text" || contentItems[0].Text != "Describe this" {
		t.Errorf("contentItems[0] = %+v", contentItems[0])
	}
	if contentItems[1].Type != "input_image" || contentItems[1].ImageURL != "https://example.com/img.png" {
		t.Errorf("contentItems[1] = %+v", contentItems[1])
	}
}

func TestConvertChatToResponsesRequest_ParallelToolCalls(t *testing.T) {
	parallel := true
	req := types.OpenAIChatCompletionsRequest{
		Model:             "gpt-4",
		ParallelToolCalls: &parallel,
	}

	result := ConvertChatToResponsesRequest(req)

	if !result.ParallelToolCalls {
		t.Error("ParallelToolCalls should be true")
	}
}

// --- Responses → Chat request conversion ---

func TestConvertResponsesToChatRequest_BasicMessage(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []types.ResponseInputItem{
			{Type: "message", Role: "user", Content: "Hello"},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if result.Model != "gpt-4" {
		t.Errorf("Model = %q, want gpt-4", result.Model)
	}
	// Should have 1 message (no instructions)
	if len(result.Messages) != 1 {
		t.Fatalf("Messages length = %d, want 1", len(result.Messages))
	}
	msg := result.Messages[0]
	if msg.Role != "user" {
		t.Errorf("Role = %q, want user", msg.Role)
	}
	if msg.Content == nil || msg.Content.Text == nil || *msg.Content.Text != "Hello" {
		t.Errorf("Content = %v, want Hello", msg.Content)
	}
}

func TestConvertResponsesToChatRequest_WithInstructions(t *testing.T) {
	instructions := "Be helpful"
	req := types.ResponsesRequest{
		Model:        "gpt-4",
		Instructions: &instructions,
		Input: []types.ResponseInputItem{
			{Type: "message", Role: "user", Content: "Hi"},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if len(result.Messages) != 2 {
		t.Fatalf("Messages length = %d, want 2", len(result.Messages))
	}
	// First message should be system
	if result.Messages[0].Role != "system" {
		t.Errorf("Messages[0].Role = %q, want system", result.Messages[0].Role)
	}
	if *result.Messages[0].Content.Text != "Be helpful" {
		t.Errorf("System message = %q, want 'Be helpful'", *result.Messages[0].Content.Text)
	}
}

func TestConvertResponsesToChatRequest_FunctionCallItems(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []types.ResponseInputItem{
			{Type: "message", Role: "assistant", Content: "Let me check"},
			{Type: "function_call", CallID: "call_1", Name: "get_weather", Arguments: `{"city":"NYC"}`},
			{Type: "function_call_output", CallID: "call_1", Output: "Sunny"},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if len(result.Messages) != 2 {
		t.Fatalf("Messages length = %d, want 2 (assistant with tool call + tool)", len(result.Messages))
	}
	// function_call should be attached to assistant message
	if len(result.Messages[0].ToolCalls) != 1 {
		t.Fatalf("ToolCalls length = %d, want 1", len(result.Messages[0].ToolCalls))
	}
	tc := result.Messages[0].ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "get_weather" {
		t.Errorf("ToolCall = %+v", tc)
	}
	// function_call_output should be tool message
	if result.Messages[1].Role != "tool" {
		t.Errorf("Messages[1].Role = %q, want tool", result.Messages[1].Role)
	}
}

func TestConvertResponsesToChatRequest_TemperatureZero(t *testing.T) {
	temp := 0.0
	req := types.ResponsesRequest{
		Model:       "gpt-4",
		Temperature: &temp,
	}

	result := ConvertResponsesToChatRequest(req)

	if result.Temperature == nil {
		t.Fatal("Temperature should not be nil (0 is a valid value)")
	}
	if *result.Temperature != 0 {
		t.Errorf("Temperature = %f, want 0", *result.Temperature)
	}
}

func TestConvertResponsesToChatRequest_Reasoning(t *testing.T) {
	tests := []struct {
		effort string
		budget int
	}{
		{"high", 32000},
		{"medium", 12000},
		{"low", 4000},
	}

	for _, tt := range tests {
		req := types.ResponsesRequest{
			Model: "gpt-4",
			Reasoning: &types.ResponseReasoning{
				Effort: tt.effort,
			},
		}

		result := ConvertResponsesToChatRequest(req)

		if result.ThinkingBudget == nil {
			t.Fatalf("effort=%q: ThinkingBudget should not be nil", tt.effort)
		}
		if *result.ThinkingBudget != tt.budget {
			t.Errorf("effort=%q: ThinkingBudget = %d, want %d", tt.effort, *result.ThinkingBudget, tt.budget)
		}
	}
}

func TestConvertResponsesToChatRequest_StreamOptions(t *testing.T) {
	req := types.ResponsesRequest{
		Model:  "gpt-4",
		Stream: true,
	}

	result := ConvertResponsesToChatRequest(req)

	if result.StreamOptions == nil {
		t.Fatal("StreamOptions should not be nil when streaming")
	}
	if !result.StreamOptions.IncludeUsage {
		t.Error("IncludeUsage should be true")
	}
}

func TestConvertResponsesToChatRequest_Tools(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Tools: []types.ResponseTool{
			{
				Type:        "function",
				Name:        "get_weather",
				Description: "Get weather info",
			},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if len(result.Tools) != 1 {
		t.Fatalf("Tools length = %d, want 1", len(result.Tools))
	}
	if result.Tools[0].Type != "function" || result.Tools[0].Function.Name != "get_weather" {
		t.Errorf("Tool = %+v", result.Tools[0])
	}
}

func TestConvertResponsesToChatRequest_FunctionCallWithoutAssistant(t *testing.T) {
	// function_call without a preceding assistant message should create one
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []types.ResponseInputItem{
			{Type: "function_call", CallID: "call_1", Name: "fn", Arguments: "{}"},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if len(result.Messages) != 1 {
		t.Fatalf("Messages length = %d, want 1", len(result.Messages))
	}
	if result.Messages[0].Role != "assistant" {
		t.Errorf("Role = %q, want assistant", result.Messages[0].Role)
	}
	if len(result.Messages[0].ToolCalls) != 1 {
		t.Fatalf("ToolCalls length = %d, want 1", len(result.Messages[0].ToolCalls))
	}
}

// --- Responses result → Chat response conversion ---

func TestConvertResponsesResultToChatResponse_TextOutput(t *testing.T) {
	result := types.ResponsesResult{
		ID:     "resp_123",
		Model:  "gpt-4",
		Status: "completed",
		Output: []types.ResponseOutputItem{
			{
				Type: "message",
				Content: []types.ResponseOutputContent{
					{Type: "output_text", Text: "Hello world"},
				},
			},
		},
		Usage: &types.ResponsesUsage{
			InputTokens:  10,
			OutputTokens: 5,
		},
	}

	resp := ConvertResponsesResultToChatResponse(result, "gpt-4")

	if resp.ID != "resp_123" {
		t.Errorf("ID = %q, want resp_123", resp.ID)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("Object = %q, want chat.completion", resp.Object)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("Choices length = %d, want 1", len(resp.Choices))
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", choice.FinishReason)
	}
	if choice.Message.Content == nil || choice.Message.Content.Text == nil {
		t.Fatal("Message content should not be nil")
	}
	if *choice.Message.Content.Text != "Hello world" {
		t.Errorf("Content = %q, want 'Hello world'", *choice.Message.Content.Text)
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 5 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
}

func TestConvertResponsesResultToChatResponse_ToolCalls(t *testing.T) {
	result := types.ResponsesResult{
		ID:     "resp_456",
		Status: "completed",
		Output: []types.ResponseOutputItem{
			{
				Type:      "function_call",
				CallID:    "call_1",
				Name:      "get_weather",
				Arguments: `{"city":"NYC"}`,
			},
		},
	}

	resp := ConvertResponsesResultToChatResponse(result, "gpt-4")

	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", resp.Choices[0].FinishReason)
	}
	if len(resp.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("ToolCalls length = %d, want 1", len(resp.Choices[0].Message.ToolCalls))
	}
	tc := resp.Choices[0].Message.ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "get_weather" {
		t.Errorf("ToolCall = %+v", tc)
	}
}

func TestConvertResponsesResultToChatResponse_Reasoning(t *testing.T) {
	result := types.ResponsesResult{
		ID:     "resp_789",
		Status: "completed",
		Output: []types.ResponseOutputItem{
			{
				Type: "reasoning",
				Summary: []types.ResponseSummaryBlock{
					{Type: "summary_text", Text: "Thinking about it..."},
				},
			},
			{
				Type: "message",
				Content: []types.ResponseOutputContent{
					{Type: "output_text", Text: "Answer"},
				},
			},
		},
	}

	resp := ConvertResponsesResultToChatResponse(result, "gpt-4")

	msg := resp.Choices[0].Message
	if msg.ReasoningText == nil || *msg.ReasoningText != "Thinking about it..." {
		t.Errorf("ReasoningText = %v, want 'Thinking about it...'", msg.ReasoningText)
	}
	if msg.Content == nil || *msg.Content.Text != "Answer" {
		t.Errorf("Content = %v, want 'Answer'", msg.Content)
	}
}

func TestConvertResponsesResultToChatResponse_Incomplete(t *testing.T) {
	result := types.ResponsesResult{
		ID:     "resp_inc",
		Status: "incomplete",
		IncompleteDetails: &types.IncompleteDetails{
			Reason: "max_output_tokens",
		},
	}

	resp := ConvertResponsesResultToChatResponse(result, "gpt-4")

	if resp.Choices[0].FinishReason != "length" {
		t.Errorf("FinishReason = %q, want length", resp.Choices[0].FinishReason)
	}
}

// --- Chat response → Responses result conversion ---

func TestConvertChatResponseToResponsesResult_TextResponse(t *testing.T) {
	text := "Hello"
	resp := types.OpenAIChatCompletionsResponse{
		ID:    "chatcmpl_123",
		Model: "gpt-4",
		Choices: []types.OpenAIChoice{
			{
				Index:        0,
				FinishReason: "stop",
				Message: types.OpenAIMessage{
					Role:    "assistant",
					Content: &types.OpenAIContent{Text: &text},
				},
			},
		},
		Usage: &types.OpenAIUsage{
			PromptTokens:     20,
			CompletionTokens: 10,
			TotalTokens:      30,
		},
	}

	result := ConvertChatResponseToResponsesResult(resp)

	if result.ID != "chatcmpl_123" {
		t.Errorf("ID = %q, want chatcmpl_123", result.ID)
	}
	if result.Status != "completed" {
		t.Errorf("Status = %q, want completed", result.Status)
	}
	if result.OutputText != "Hello" {
		t.Errorf("OutputText = %q, want Hello", result.OutputText)
	}
	if result.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if result.Usage.InputTokens != 20 || result.Usage.OutputTokens != 10 {
		t.Errorf("Usage = %+v", result.Usage)
	}
}

func TestConvertChatResponseToResponsesResult_ToolCalls(t *testing.T) {
	resp := types.OpenAIChatCompletionsResponse{
		ID:    "chatcmpl_tc",
		Model: "gpt-4",
		Choices: []types.OpenAIChoice{
			{
				FinishReason: "tool_calls",
				Message: types.OpenAIMessage{
					Role: "assistant",
					ToolCalls: []types.OpenAIToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: types.OpenAIToolCallFunction{
								Name:      "search",
								Arguments: `{"q":"test"}`,
							},
						},
					},
				},
			},
		},
	}

	result := ConvertChatResponseToResponsesResult(resp)

	if result.Status != "completed" {
		t.Errorf("Status = %q, want completed", result.Status)
	}
	// Should have function_call output item
	found := false
	for _, item := range result.Output {
		if item.Type == "function_call" && item.CallID == "call_1" {
			found = true
			if item.Name != "search" {
				t.Errorf("function call Name = %q, want search", item.Name)
			}
		}
	}
	if !found {
		t.Error("Expected function_call output item")
	}
}

func TestConvertChatResponseToResponsesResult_LengthFinish(t *testing.T) {
	text := "partial"
	resp := types.OpenAIChatCompletionsResponse{
		ID:    "chatcmpl_len",
		Model: "gpt-4",
		Choices: []types.OpenAIChoice{
			{
				FinishReason: "length",
				Message: types.OpenAIMessage{
					Role:    "assistant",
					Content: &types.OpenAIContent{Text: &text},
				},
			},
		},
	}

	result := ConvertChatResponseToResponsesResult(resp)

	if result.Status != "incomplete" {
		t.Errorf("Status = %q, want incomplete", result.Status)
	}
	if result.IncompleteDetails == nil {
		t.Fatal("IncompleteDetails should not be nil")
	}
	if result.IncompleteDetails.Reason != "max_output_tokens" {
		t.Errorf("Reason = %q, want max_output_tokens", result.IncompleteDetails.Reason)
	}
}

func TestConvertChatResponseToResponsesResult_Reasoning(t *testing.T) {
	text := "answer"
	reasoning := "thinking..."
	resp := types.OpenAIChatCompletionsResponse{
		ID:    "chatcmpl_reason",
		Model: "gpt-4",
		Choices: []types.OpenAIChoice{
			{
				FinishReason: "stop",
				Message: types.OpenAIMessage{
					Role:          "assistant",
					Content:       &types.OpenAIContent{Text: &text},
					ReasoningText: &reasoning,
				},
			},
		},
	}

	result := ConvertChatResponseToResponsesResult(resp)

	// Should have reasoning output item
	found := false
	for _, item := range result.Output {
		if item.Type == "reasoning" {
			found = true
			if len(item.Summary) != 1 || item.Summary[0].Text != "thinking..." {
				t.Errorf("Reasoning summary = %+v", item.Summary)
			}
		}
	}
	if !found {
		t.Error("Expected reasoning output item")
	}
}

// --- Streaming: Responses → Chat Completions ---

func TestConvertResponsesStreamEvent_Created(t *testing.T) {
	state := NewResponsesStreamConvertState()
	event := types.ResponseStreamEvent{
		Type: "response.created",
		Response: &types.ResponsesResult{
			ID:    "resp_stream",
			Model: "gpt-4",
		},
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if len(chunks) != 1 {
		t.Fatalf("chunks length = %d, want 1", len(chunks))
	}
	if chunks[0].ID != "resp_stream" {
		t.Errorf("ID = %q, want resp_stream", chunks[0].ID)
	}
	if chunks[0].Model != "gpt-4" {
		t.Errorf("Model = %q, want gpt-4", chunks[0].Model)
	}
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("Delta.Role = %q, want assistant", chunks[0].Choices[0].Delta.Role)
	}
	if state.ID != "resp_stream" || state.Model != "gpt-4" {
		t.Error("State not updated after response.created")
	}
}

func TestConvertResponsesStreamEvent_TextDelta(t *testing.T) {
	state := NewResponsesStreamConvertState()
	state.ID = "resp_1"
	state.Model = "gpt-4"

	event := types.ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Hello",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if len(chunks) != 1 {
		t.Fatalf("chunks length = %d, want 1", len(chunks))
	}
	delta := chunks[0].Choices[0].Delta
	if delta.Content == nil || delta.Content.Text == nil || *delta.Content.Text != "Hello" {
		t.Errorf("Delta content = %v, want Hello", delta.Content)
	}
}

func TestConvertResponsesStreamEvent_EmptyDelta(t *testing.T) {
	state := NewResponsesStreamConvertState()
	event := types.ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if len(chunks) != 0 {
		t.Errorf("Empty delta should produce 0 chunks, got %d", len(chunks))
	}
}

func TestConvertResponsesStreamEvent_ReasoningDelta(t *testing.T) {
	state := NewResponsesStreamConvertState()
	state.ID = "resp_1"

	event := types.ResponseStreamEvent{
		Type:  "response.reasoning_summary_text.delta",
		Delta: "thinking",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if len(chunks) != 1 {
		t.Fatalf("chunks length = %d, want 1", len(chunks))
	}
	if chunks[0].Choices[0].Delta.ReasoningText == nil {
		t.Fatal("ReasoningText should not be nil")
	}
	if *chunks[0].Choices[0].Delta.ReasoningText != "thinking" {
		t.Errorf("ReasoningText = %q, want thinking", *chunks[0].Choices[0].Delta.ReasoningText)
	}
}

func TestConvertResponsesStreamEvent_FunctionCallFlow(t *testing.T) {
	state := NewResponsesStreamConvertState()
	state.ID = "resp_1"

	// Step 1: output_item.added with function_call
	addedEvent := types.ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &types.ResponseOutputItem{
			Type:   "function_call",
			CallID: "call_1",
			Name:   "get_weather",
		},
	}
	chunks := ConvertResponsesStreamEventToChatChunk(addedEvent, state)
	if len(chunks) != 1 {
		t.Fatalf("output_item.added: chunks = %d, want 1", len(chunks))
	}
	tc := chunks[0].Choices[0].Delta.ToolCalls
	if len(tc) != 1 || tc[0].ID != "call_1" || tc[0].Function.Name != "get_weather" {
		t.Errorf("ToolCall = %+v", tc)
	}

	// Step 2: function_call_arguments.delta
	argsEvent := types.ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"city":`,
	}
	chunks = ConvertResponsesStreamEventToChatChunk(argsEvent, state)
	if len(chunks) != 1 {
		t.Fatalf("arguments.delta: chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments != `{"city":` {
		t.Errorf("Arguments = %q", chunks[0].Choices[0].Delta.ToolCalls[0].Function.Arguments)
	}
}

func TestConvertResponsesStreamEvent_Completed(t *testing.T) {
	state := NewResponsesStreamConvertState()
	state.ID = "resp_1"

	event := types.ResponseStreamEvent{
		Type: "response.completed",
		Response: &types.ResponsesResult{
			Status: "completed",
			Usage: &types.ResponsesUsage{
				InputTokens:  100,
				OutputTokens: 50,
			},
		},
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if !state.Finished {
		t.Error("State should be Finished")
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", chunks[0].Choices[0].FinishReason)
	}
	if chunks[0].Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if chunks[0].Usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", chunks[0].Usage.PromptTokens)
	}
}

func TestConvertResponsesStreamEvent_Failed(t *testing.T) {
	state := NewResponsesStreamConvertState()

	event := types.ResponseStreamEvent{
		Type: "response.failed",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)

	if !state.Finished {
		t.Error("State should be Finished")
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", chunks[0].Choices[0].FinishReason)
	}
}

func TestConvertResponsesStreamEvent_UnknownType(t *testing.T) {
	state := NewResponsesStreamConvertState()
	event := types.ResponseStreamEvent{
		Type: "response.output_item.done",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)
	if len(chunks) != 0 {
		t.Errorf("Unknown event type should produce 0 chunks, got %d", len(chunks))
	}
}

// --- Streaming: Chat Completions → Responses ---

func TestConvertChatChunkToResponsesStreamEvents_FirstChunk(t *testing.T) {
	state := NewChatStreamConvertState()
	text := "Hi"
	chunk := types.OpenAIChatCompletionChunk{
		ID:    "chatcmpl_1",
		Model: "gpt-4",
		Choices: []types.OpenAIChunkChoice{
			{
				Index: 0,
				Delta: types.OpenAIMessage{
					Role:    "assistant",
					Content: &types.OpenAIContent{Text: &text},
				},
			},
		},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	if !state.CreatedSent {
		t.Error("CreatedSent should be true")
	}
	// Should have: response.created, output_item.added, output_text.delta
	if len(events) < 3 {
		t.Fatalf("events = %d, want >= 3", len(events))
	}
	if events[0].Type != "response.created" {
		t.Errorf("events[0].Type = %q, want response.created", events[0].Type)
	}
	if events[1].Type != "response.output_item.added" {
		t.Errorf("events[1].Type = %q, want response.output_item.added", events[1].Type)
	}
	if events[2].Type != "response.output_text.delta" {
		t.Errorf("events[2].Type = %q, want response.output_text.delta", events[2].Type)
	}
	if events[2].Delta != "Hi" {
		t.Errorf("Delta = %q, want Hi", events[2].Delta)
	}
}

func TestConvertChatChunkToResponsesStreamEvents_SubsequentTextChunk(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true
	state.OutputItemStarted = true
	state.CurrentOutputIndex = 0
	state.NextOutputIndex = 1

	text := " world"
	chunk := types.OpenAIChatCompletionChunk{
		ID:    "chatcmpl_1",
		Model: "gpt-4",
		Choices: []types.OpenAIChunkChoice{
			{
				Index: 0,
				Delta: types.OpenAIMessage{
					Content: &types.OpenAIContent{Text: &text},
				},
			},
		},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	// Should only have text delta (no created, no output_item.added)
	if len(events) != 1 {
		t.Fatalf("events = %d, want 1", len(events))
	}
	if events[0].Type != "response.output_text.delta" {
		t.Errorf("Type = %q, want response.output_text.delta", events[0].Type)
	}
	if events[0].Delta != " world" {
		t.Errorf("Delta = %q, want ' world'", events[0].Delta)
	}
}

func TestConvertChatChunkToResponsesStreamEvents_ReasoningDelta(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true

	reasoning := "let me think"
	chunk := types.OpenAIChatCompletionChunk{
		ID: "chatcmpl_1",
		Choices: []types.OpenAIChunkChoice{
			{Delta: types.OpenAIMessage{ReasoningText: &reasoning}},
		},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	// Should have: output_item.added (reasoning) + reasoning_summary_text.delta
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	if events[0].Type != "response.output_item.added" {
		t.Errorf("events[0].Type = %q, want response.output_item.added", events[0].Type)
	}
	if events[0].Item.Type != "reasoning" {
		t.Errorf("Item.Type = %q, want reasoning", events[0].Item.Type)
	}
	if events[1].Type != "response.reasoning_summary_text.delta" {
		t.Errorf("events[1].Type = %q", events[1].Type)
	}
}

func TestConvertChatChunkToResponsesStreamEvents_ToolCall(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true

	idx := 0
	chunk := types.OpenAIChatCompletionChunk{
		ID: "chatcmpl_1",
		Choices: []types.OpenAIChunkChoice{
			{
				Delta: types.OpenAIMessage{
					ToolCalls: []types.OpenAIToolCall{
						{
							Index: &idx,
							ID:    "call_1",
							Type:  "function",
							Function: types.OpenAIToolCallFunction{
								Name:      "search",
								Arguments: `{"q":`,
							},
						},
					},
				},
			},
		},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	// Should have: output_item.added + function_call_arguments.delta
	if len(events) != 2 {
		t.Fatalf("events = %d, want 2", len(events))
	}
	if events[0].Type != "response.output_item.added" {
		t.Errorf("events[0].Type = %q, want response.output_item.added", events[0].Type)
	}
	if events[0].Item.Type != "function_call" {
		t.Errorf("Item.Type = %q, want function_call", events[0].Item.Type)
	}
	if events[1].Type != "response.function_call_arguments.delta" {
		t.Errorf("events[1].Type = %q, want response.function_call_arguments.delta", events[1].Type)
	}
}

func TestConvertChatChunkToResponsesStreamEvents_FinishWithUsageSameChunk(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true
	state.ID = "chatcmpl_1"
	state.Model = "gpt-4"

	chunk := types.OpenAIChatCompletionChunk{
		ID:    "chatcmpl_1",
		Model: "gpt-4",
		Choices: []types.OpenAIChunkChoice{
			{FinishReason: "stop"},
		},
		Usage: &types.OpenAIUsage{
			PromptTokens:     50,
			CompletionTokens: 20,
			TotalTokens:      70,
		},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	if !state.Finished {
		t.Error("State should be Finished")
	}
	// Should emit response.completed
	found := false
	for _, e := range events {
		if e.Type == "response.completed" {
			found = true
			if e.Response.Usage == nil {
				t.Error("Usage should be present")
			}
			if e.Response.Usage.InputTokens != 50 {
				t.Errorf("InputTokens = %d, want 50", e.Response.Usage.InputTokens)
			}
		}
	}
	if !found {
		t.Error("Expected response.completed event")
	}
}

func TestConvertChatChunkToResponsesStreamEvents_DeferredTermination(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true
	state.ID = "chatcmpl_1"
	state.Model = "gpt-4"

	// Step 1: finish_reason chunk without usage
	finishChunk := types.OpenAIChatCompletionChunk{
		ID: "chatcmpl_1",
		Choices: []types.OpenAIChunkChoice{
			{FinishReason: "stop"},
		},
	}
	events := ConvertChatChunkToResponsesStreamEvents(finishChunk, state)

	// Should NOT emit termination yet (deferred)
	if state.Finished {
		t.Error("State should NOT be Finished yet")
	}
	if !state.FinishSeen {
		t.Error("FinishSeen should be true")
	}
	for _, e := range events {
		if e.Type == "response.completed" || e.Type == "response.incomplete" {
			t.Error("Should not emit termination event yet")
		}
	}

	// Step 2: usage-only chunk (empty choices)
	usageChunk := types.OpenAIChatCompletionChunk{
		ID: "chatcmpl_1",
		Usage: &types.OpenAIUsage{
			PromptTokens:     100,
			CompletionTokens: 40,
		},
	}
	events = ConvertChatChunkToResponsesStreamEvents(usageChunk, state)

	if !state.Finished {
		t.Error("State should be Finished after usage chunk")
	}
	found := false
	for _, e := range events {
		if e.Type == "response.completed" {
			found = true
			if e.Response.Usage.InputTokens != 100 {
				t.Errorf("InputTokens = %d, want 100", e.Response.Usage.InputTokens)
			}
		}
	}
	if !found {
		t.Error("Expected response.completed event after usage chunk")
	}
}

func TestConvertChatChunkToResponsesStreamEvents_LengthFinish(t *testing.T) {
	state := NewChatStreamConvertState()
	state.CreatedSent = true
	state.ID = "chatcmpl_1"
	state.Model = "gpt-4"

	chunk := types.OpenAIChatCompletionChunk{
		ID: "chatcmpl_1",
		Choices: []types.OpenAIChunkChoice{
			{FinishReason: "length"},
		},
		Usage: &types.OpenAIUsage{PromptTokens: 10, CompletionTokens: 5},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	found := false
	for _, e := range events {
		if e.Type == "response.incomplete" {
			found = true
			if e.Response.Status != "incomplete" {
				t.Errorf("Status = %q, want incomplete", e.Response.Status)
			}
		}
	}
	if !found {
		t.Error("Expected response.incomplete event for length finish")
	}
}

func TestConvertChatChunkToResponsesStreamEvents_EmptyChoices(t *testing.T) {
	state := NewChatStreamConvertState()

	// Usage-only chunk before finish_reason (edge case)
	chunk := types.OpenAIChatCompletionChunk{
		ID:    "chatcmpl_1",
		Usage: &types.OpenAIUsage{PromptTokens: 10},
	}

	events := ConvertChatChunkToResponsesStreamEvents(chunk, state)

	// Should capture usage but not emit termination (FinishSeen is false)
	if state.PendingUsage == nil {
		t.Error("PendingUsage should be set")
	}
	if state.Finished {
		t.Error("Should not be Finished without FinishSeen")
	}
	if len(events) != 0 {
		t.Errorf("events = %d, want 0", len(events))
	}
}

// --- Helper function tests ---

func TestThinkingBudgetToEffort(t *testing.T) {
	tests := []struct {
		budget int
		want   string
	}{
		{32000, "high"},
		{16000, "high"},
		{15999, "medium"},
		{8000, "medium"},
		{7999, "low"},
		{0, "low"},
	}

	for _, tt := range tests {
		got := thinkingBudgetToEffort(tt.budget)
		if got != tt.want {
			t.Errorf("thinkingBudgetToEffort(%d) = %q, want %q", tt.budget, got, tt.want)
		}
	}
}

func TestEffortToThinkingBudget(t *testing.T) {
	tests := []struct {
		effort string
		want   int
	}{
		{"high", 32000},
		{"medium", 12000},
		{"low", 4000},
		{"unknown", 32000}, // default
	}

	for _, tt := range tests {
		got := effortToThinkingBudget(tt.effort)
		if got != tt.want {
			t.Errorf("effortToThinkingBudget(%q) = %d, want %d", tt.effort, got, tt.want)
		}
	}
}

func TestContentToString(t *testing.T) {
	tests := []struct {
		name    string
		content *types.OpenAIContent
		want    string
	}{
		{"nil content", nil, ""},
		{"text content", &types.OpenAIContent{Text: strPtr("hello")}, "hello"},
		{"parts content", &types.OpenAIContent{Parts: []types.OpenAIContentPart{
			{Type: "text", Text: "a"},
			{Type: "text", Text: "b"},
		}}, "a\nb"},
		{"empty parts", &types.OpenAIContent{Parts: []types.OpenAIContentPart{}}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := contentToString(tt.content)
			if got != tt.want {
				t.Errorf("contentToString() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMapResponsesStatusToFinishReason(t *testing.T) {
	tests := []struct {
		name   string
		result *types.ResponsesResult
		want   string
	}{
		{"nil result", nil, "stop"},
		{"completed no tool calls", &types.ResponsesResult{Status: "completed"}, "stop"},
		{"completed with tool calls", &types.ResponsesResult{
			Status: "completed",
			Output: []types.ResponseOutputItem{{Type: "function_call"}},
		}, "tool_calls"},
		{"incomplete max tokens", &types.ResponsesResult{
			Status:            "incomplete",
			IncompleteDetails: &types.IncompleteDetails{Reason: "max_output_tokens"},
		}, "length"},
		{"incomplete other", &types.ResponsesResult{Status: "incomplete"}, "stop"},
		{"failed", &types.ResponsesResult{Status: "failed"}, "stop"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapResponsesStatusToFinishReason(tt.result)
			if got != tt.want {
				t.Errorf("mapResponsesStatusToFinishReason() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMapChatFinishReasonToResponsesStatus(t *testing.T) {
	tests := []struct {
		name   string
		resp   types.OpenAIChatCompletionsResponse
		want   string
	}{
		{"empty choices", types.OpenAIChatCompletionsResponse{}, "completed"},
		{"stop", types.OpenAIChatCompletionsResponse{
			Choices: []types.OpenAIChoice{{FinishReason: "stop"}},
		}, "completed"},
		{"length", types.OpenAIChatCompletionsResponse{
			Choices: []types.OpenAIChoice{{FinishReason: "length"}},
		}, "incomplete"},
		{"tool_calls", types.OpenAIChatCompletionsResponse{
			Choices: []types.OpenAIChoice{{FinishReason: "tool_calls"}},
		}, "completed"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mapChatFinishReasonToResponsesStatus(tt.resp)
			if got != tt.want {
				t.Errorf("mapChatFinishReasonToResponsesStatus() = %q, want %q", got, tt.want)
			}
		})
	}
}

func strPtr(s string) *string {
	return &s
}
