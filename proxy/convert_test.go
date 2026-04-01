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

	// System messages should be extracted into instructions, not input items
	if len(result.Input) != 0 {
		t.Errorf("Input length = %d, want 0 (system messages go to instructions)", len(result.Input))
	}
	if result.Instructions == nil {
		t.Fatal("Instructions should not be nil")
	}
	if *result.Instructions != "You are helpful" {
		t.Errorf("Instructions = %q, want %q", *result.Instructions, "You are helpful")
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

	if result.ParallelToolCalls == nil || !*result.ParallelToolCalls {
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
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2", len(chunks))
	}
	// First chunk should contain error message as content
	if chunks[0].Choices[0].Delta.Content == nil {
		t.Error("first chunk should have error content")
	}
	// Second chunk should have finish reason
	if chunks[1].Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", chunks[1].Choices[0].FinishReason)
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

// --- Task 1: System messages → instructions ---

func TestConvertChatToResponsesRequest_MultipleSystemMessages(t *testing.T) {
	sys1 := "You are helpful"
	sys2 := "Be concise"
	userText := "Hi"
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{Role: "system", Content: &types.OpenAIContent{Text: &sys1}},
			{Role: "system", Content: &types.OpenAIContent{Text: &sys2}},
			{Role: "user", Content: &types.OpenAIContent{Text: &userText}},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if result.Instructions == nil {
		t.Fatal("Instructions should not be nil")
	}
	if *result.Instructions != "You are helpful Be concise" {
		t.Errorf("Instructions = %q, want %q", *result.Instructions, "You are helpful Be concise")
	}
	// Only user message in input
	if len(result.Input) != 1 {
		t.Fatalf("Input length = %d, want 1", len(result.Input))
	}
	if result.Input[0].Role != "user" {
		t.Errorf("Input[0].Role = %q, want user", result.Input[0].Role)
	}
}

// --- Task 2: response_format / text.format ---

func TestConvertChatToResponsesRequest_ResponseFormat(t *testing.T) {
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		ResponseFormat: &types.ResponseFormat{
			Type: "json_object",
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if result.Text == nil {
		t.Fatal("Text should not be nil")
	}
	if result.Text.Format.Type != "json_object" {
		t.Errorf("Text.Format.Type = %q, want json_object", result.Text.Format.Type)
	}
}

func TestConvertChatToResponsesRequest_ResponseFormatJSONSchema(t *testing.T) {
	schema := map[string]interface{}{"name": "my_schema", "schema": map[string]interface{}{"type": "object"}}
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		ResponseFormat: &types.ResponseFormat{
			Type:       "json_schema",
			JSONSchema: schema,
		},
	}

	result := ConvertChatToResponsesRequest(req)

	if result.Text == nil {
		t.Fatal("Text should not be nil")
	}
	if result.Text.Format.Type != "json_schema" {
		t.Errorf("Text.Format.Type = %q, want json_schema", result.Text.Format.Type)
	}
	if result.Text.Format.JSONSchema == nil {
		t.Error("Text.Format.JSONSchema should not be nil")
	}
}

func TestConvertResponsesToChatRequest_TextFormat(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Text: &types.ResponseText{
			Format: types.ResponseTextFormat{
				Type: "json_object",
			},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if result.ResponseFormat == nil {
		t.Fatal("ResponseFormat should not be nil")
	}
	if result.ResponseFormat.Type != "json_object" {
		t.Errorf("ResponseFormat.Type = %q, want json_object", result.ResponseFormat.Type)
	}
}

func TestConvertResponsesToChatRequest_TextFormatJSONSchema(t *testing.T) {
	schema := map[string]interface{}{"name": "test"}
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Text: &types.ResponseText{
			Format: types.ResponseTextFormat{
				Type:       "json_schema",
				JSONSchema: schema,
			},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	if result.ResponseFormat == nil {
		t.Fatal("ResponseFormat should not be nil")
	}
	if result.ResponseFormat.Type != "json_schema" {
		t.Errorf("ResponseFormat.Type = %q, want json_schema", result.ResponseFormat.Type)
	}
	if result.ResponseFormat.JSONSchema == nil {
		t.Error("ResponseFormat.JSONSchema should not be nil")
	}
}

// --- Task 3: tool_choice direction-aware conversion ---

func TestChatToolChoiceToResponses_StringPassthrough(t *testing.T) {
	result := chatToolChoiceToResponses("auto")
	if result != "auto" {
		t.Errorf("got %v, want auto", result)
	}
}

func TestChatToolChoiceToResponses_MapAuto(t *testing.T) {
	result := chatToolChoiceToResponses(map[string]interface{}{"type": "auto"})
	if result != "auto" {
		t.Errorf("got %v, want auto", result)
	}
}

func TestChatToolChoiceToResponses_MapNone(t *testing.T) {
	result := chatToolChoiceToResponses(map[string]interface{}{"type": "none"})
	if result != "none" {
		t.Errorf("got %v, want none", result)
	}
}

func TestChatToolChoiceToResponses_MapRequired(t *testing.T) {
	result := chatToolChoiceToResponses(map[string]interface{}{"type": "required"})
	if result != "required" {
		t.Errorf("got %v, want required", result)
	}
}

func TestChatToolChoiceToResponses_MapTool(t *testing.T) {
	result := chatToolChoiceToResponses(map[string]interface{}{"type": "tool"})
	if result != "required" {
		t.Errorf("got %v, want required", result)
	}
}

func TestChatToolChoiceToResponses_MapFunction(t *testing.T) {
	input := map[string]interface{}{
		"type":     "function",
		"function": map[string]interface{}{"name": "my_func"},
	}
	result := chatToolChoiceToResponses(input)
	// Should convert to Responses API format: {"type":"function","name":"my_func"}
	m, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map, got %T", result)
	}
	if m["type"] != "function" {
		t.Errorf("type = %v, want function", m["type"])
	}
	if m["name"] != "my_func" {
		t.Errorf("name = %v, want my_func", m["name"])
	}
	// Should NOT have nested "function" key
	if _, hasFunc := m["function"]; hasFunc {
		t.Error("should not have nested 'function' key in Responses format")
	}
}

func TestConvertChatToResponsesRequest_ToolChoiceNormalized(t *testing.T) {
	// Simulate Cursor IDE sending {"type": "auto"}
	req := types.OpenAIChatCompletionsRequest{
		Model:      "gpt-4",
		ToolChoice: map[string]interface{}{"type": "auto"},
	}

	result := ConvertChatToResponsesRequest(req)

	if result.ToolChoice != "auto" {
		t.Errorf("ToolChoice = %v, want auto", result.ToolChoice)
	}
}

func TestConvertResponsesToChatRequest_ToolChoiceNormalized(t *testing.T) {
	req := types.ResponsesRequest{
		Model:      "gpt-4",
		ToolChoice: map[string]interface{}{"type": "none"},
	}

	result := ConvertResponsesToChatRequest(req)

	if result.ToolChoice != "none" {
		t.Errorf("ToolChoice = %v, want none", result.ToolChoice)
	}
}

// --- Task 4: previous_response_id ---

func TestConvertResponsesToChatRequest_PreviousResponseID(t *testing.T) {
	// Should not panic or error, just logs debug and ignores
	req := types.ResponsesRequest{
		Model:              "gpt-4",
		PreviousResponseID: "resp_abc123",
		Input: []types.ResponseInputItem{
			{Type: "message", Role: "user", Content: "Hi"},
		},
	}

	result := ConvertResponsesToChatRequest(req)

	// Should still produce valid output
	if len(result.Messages) != 1 {
		t.Fatalf("Messages length = %d, want 1", len(result.Messages))
	}
}

func TestPreviousResponseID_JSONRoundtrip(t *testing.T) {
	req := types.ResponsesRequest{
		Model:              "gpt-4",
		PreviousResponseID: "resp_abc123",
		Stream:             true,
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var decoded types.ResponsesRequest
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if decoded.PreviousResponseID != "resp_abc123" {
		t.Errorf("PreviousResponseID = %q, want resp_abc123", decoded.PreviousResponseID)
	}
}

// --- Task 5: Reasoning encrypted_content roundtrip ---

func TestConvertResponsesResultToChatResponse_ReasoningWithEncryptedContent(t *testing.T) {
	result := types.ResponsesResult{
		ID:     "resp_enc",
		Status: "completed",
		Output: []types.ResponseOutputItem{
			{
				Type:             "reasoning",
				ID:               "rs_abc",
				EncryptedContent: "encrypted_data_here",
				Summary: []types.ResponseSummaryBlock{
					{Type: "summary_text", Text: "Thinking..."},
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
	// Should have reasoning_text (backward compat)
	if msg.ReasoningText == nil || *msg.ReasoningText != "Thinking..." {
		t.Errorf("ReasoningText = %v, want 'Thinking...'", msg.ReasoningText)
	}
	// Should have reasoning_items with encrypted_content
	if len(msg.ReasoningItems) != 1 {
		t.Fatalf("ReasoningItems length = %d, want 1", len(msg.ReasoningItems))
	}
	ri := msg.ReasoningItems[0]
	if ri.ID != "rs_abc" {
		t.Errorf("ReasoningItems[0].ID = %q, want rs_abc", ri.ID)
	}
	if ri.EncryptedContent != "encrypted_data_here" {
		t.Errorf("EncryptedContent = %q, want encrypted_data_here", ri.EncryptedContent)
	}
	if ri.Type != "reasoning" {
		t.Errorf("Type = %q, want reasoning", ri.Type)
	}
}

func TestConvertChatToResponsesRequest_ReasoningItemsRoundtrip(t *testing.T) {
	text := "Answer"
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			{
				Role:    "assistant",
				Content: &types.OpenAIContent{Text: &text},
				ReasoningItems: []types.ReasoningItem{
					{
						ID:               "rs_abc",
						Type:             "reasoning",
						EncryptedContent: "encrypted_data_here",
						Summary: []types.ReasoningSummary{
							{Type: "summary_text", Text: "Thinking..."},
						},
					},
				},
			},
		},
	}

	result := ConvertChatToResponsesRequest(req)

	// Should have reasoning input item + assistant message
	if len(result.Input) < 2 {
		t.Fatalf("Input length = %d, want >= 2", len(result.Input))
	}
	// First should be reasoning item
	ri := result.Input[0]
	if ri.Type != "reasoning" {
		t.Errorf("Input[0].Type = %q, want reasoning", ri.Type)
	}
	if ri.ID != "rs_abc" {
		t.Errorf("Input[0].ID = %q, want rs_abc", ri.ID)
	}
	if ri.EncryptedContent != "encrypted_data_here" {
		t.Errorf("Input[0].EncryptedContent = %q, want encrypted_data_here", ri.EncryptedContent)
	}
	// Second should be assistant message
	if result.Input[1].Type != "message" || result.Input[1].Role != "assistant" {
		t.Errorf("Input[1] type=%q role=%q, want message/assistant", result.Input[1].Type, result.Input[1].Role)
	}
}

func TestReasoningEncryptedContent_FullRoundtrip(t *testing.T) {
	// Simulate: Responses API result → Chat response → back to Responses request
	// The encrypted_content should survive the roundtrip.

	// Step 1: Responses result with encrypted_content
	responsesResult := types.ResponsesResult{
		ID:     "resp_rt",
		Status: "completed",
		Output: []types.ResponseOutputItem{
			{
				Type:             "reasoning",
				ID:               "rs_123",
				EncryptedContent: "secret_encrypted_blob",
				Summary: []types.ResponseSummaryBlock{
					{Type: "summary_text", Text: "I thought about it"},
				},
			},
			{
				Type: "message",
				Content: []types.ResponseOutputContent{
					{Type: "output_text", Text: "Here is my answer"},
				},
			},
		},
	}

	// Step 2: Convert to Chat response
	chatResp := ConvertResponsesResultToChatResponse(responsesResult, "gpt-4")

	// Step 3: Build a new Chat request using the assistant message from the response
	userText := "Follow up question"
	chatReq := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		Messages: []types.OpenAIMessage{
			chatResp.Choices[0].Message, // assistant with reasoning_items
			{Role: "user", Content: &types.OpenAIContent{Text: &userText}},
		},
	}

	// Step 4: Convert to Responses request
	responsesReq := ConvertChatToResponsesRequest(chatReq)

	// Verify encrypted_content survived
	foundReasoning := false
	for _, item := range responsesReq.Input {
		if item.Type == "reasoning" {
			foundReasoning = true
			if item.EncryptedContent != "secret_encrypted_blob" {
				t.Errorf("EncryptedContent = %q, want secret_encrypted_blob", item.EncryptedContent)
			}
			if item.ID != "rs_123" {
				t.Errorf("ID = %q, want rs_123", item.ID)
			}
		}
	}
	if !foundReasoning {
		t.Error("Expected reasoning input item with encrypted_content")
	}
}

// ============================================================
// Tests for code review fixes (Issues 1-7)
// ============================================================

// --- Issue 1: tool_choice direction-aware conversion ---

func TestChatToolChoiceToResponses_FunctionWithName(t *testing.T) {
	// Chat format: {"type":"function","function":{"name":"foo"}}
	input := map[string]interface{}{
		"type":     "function",
		"function": map[string]interface{}{"name": "get_weather"},
	}
	result := chatToolChoiceToResponses(input)
	m, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map, got %T", result)
	}
	if m["type"] != "function" {
		t.Errorf("type = %v, want function", m["type"])
	}
	if m["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", m["name"])
	}
	if _, has := m["function"]; has {
		t.Error("Responses format should not have nested 'function' key")
	}
}

func TestChatToolChoiceToResponses_FunctionWithoutName(t *testing.T) {
	input := map[string]interface{}{
		"type":     "function",
		"function": map[string]interface{}{},
	}
	result := chatToolChoiceToResponses(input)
	if result != "required" {
		t.Errorf("function without name should fallback to 'required', got %v", result)
	}
}

func TestResponsesToolChoiceToChat_FunctionWithName(t *testing.T) {
	// Responses format: {"type":"function","name":"foo"}
	input := map[string]interface{}{
		"type": "function",
		"name": "get_weather",
	}
	result := responsesToolChoiceToChat(input)
	m, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map, got %T", result)
	}
	if m["type"] != "function" {
		t.Errorf("type = %v, want function", m["type"])
	}
	fn, ok := m["function"].(map[string]interface{})
	if !ok {
		t.Fatal("expected nested 'function' map in Chat format")
	}
	if fn["name"] != "get_weather" {
		t.Errorf("function.name = %v, want get_weather", fn["name"])
	}
}

func TestResponsesToolChoiceToChat_FunctionWithoutName(t *testing.T) {
	input := map[string]interface{}{
		"type": "function",
	}
	result := responsesToolChoiceToChat(input)
	if result != "required" {
		t.Errorf("function without name should fallback to 'required', got %v", result)
	}
}

func TestToolChoiceRoundTrip_ChatToResponsesAndBack(t *testing.T) {
	// Start with Chat format
	chatChoice := map[string]interface{}{
		"type":     "function",
		"function": map[string]interface{}{"name": "my_tool"},
	}
	responsesChoice := chatToolChoiceToResponses(chatChoice)
	chatBack := responsesToolChoiceToChat(responsesChoice)

	m, ok := chatBack.(map[string]interface{})
	if !ok {
		t.Fatalf("expected map, got %T", chatBack)
	}
	fn, ok := m["function"].(map[string]interface{})
	if !ok {
		t.Fatal("expected nested function key")
	}
	if fn["name"] != "my_tool" {
		t.Errorf("round-trip lost function name: got %v", fn["name"])
	}
}

// --- Issue 2: reasoning items in Responses → Chat conversion ---

func TestConvertInputToMessages_ReasoningAttachedToAssistant(t *testing.T) {
	summaryBlocks := []types.ResponseSummaryBlock{
		{Type: "summary_text", Text: "I thought about it"},
	}
	input := []types.ResponseInputItem{
		{
			Type:             "reasoning",
			ID:               "r_1",
			EncryptedContent: "enc123",
			Summary:          &summaryBlocks,
		},
		{
			Type:    "message",
			Role:    "assistant",
			Content: "Hello",
		},
	}
	messages := convertInputToMessages(input, nil)
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != "assistant" {
		t.Errorf("role = %s, want assistant", messages[0].Role)
	}
	if len(messages[0].ReasoningItems) != 1 {
		t.Fatalf("expected 1 reasoning item, got %d", len(messages[0].ReasoningItems))
	}
	ri := messages[0].ReasoningItems[0]
	if ri.ID != "r_1" {
		t.Errorf("reasoning ID = %s, want r_1", ri.ID)
	}
	if ri.EncryptedContent != "enc123" {
		t.Errorf("encrypted_content = %s, want enc123", ri.EncryptedContent)
	}
	if len(ri.Summary) != 1 || ri.Summary[0].Text != "I thought about it" {
		t.Errorf("summary mismatch: %+v", ri.Summary)
	}
}

func TestConvertInputToMessages_LeftoverReasoningCreatesAssistant(t *testing.T) {
	summaryBlocks := []types.ResponseSummaryBlock{
		{Type: "summary_text", Text: "thinking..."},
	}
	input := []types.ResponseInputItem{
		{
			Type:    "message",
			Role:    "user",
			Content: "Hi",
		},
		{
			Type:             "reasoning",
			ID:               "r_2",
			EncryptedContent: "enc456",
			Summary:          &summaryBlocks,
		},
	}
	messages := convertInputToMessages(input, nil)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[1].Role != "assistant" {
		t.Errorf("leftover reasoning should create assistant message, got role=%s", messages[1].Role)
	}
	if len(messages[1].ReasoningItems) != 1 {
		t.Fatalf("expected 1 reasoning item on created assistant, got %d", len(messages[1].ReasoningItems))
	}
}

func TestConvertInputToMessages_ReasoningNotAttachedToUser(t *testing.T) {
	summaryBlocks := []types.ResponseSummaryBlock{
		{Type: "summary_text", Text: "thinking"},
	}
	input := []types.ResponseInputItem{
		{
			Type:    "reasoning",
			ID:      "r_3",
			Summary: &summaryBlocks,
		},
		{
			Type:    "message",
			Role:    "user",
			Content: "Hello",
		},
	}
	messages := convertInputToMessages(input, nil)
	// Reasoning should NOT attach to user message; should create a separate assistant message
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0].Role != "user" {
		t.Errorf("first message role = %s, want user", messages[0].Role)
	}
	if len(messages[0].ReasoningItems) != 0 {
		t.Error("reasoning should not be attached to user message")
	}
	// Leftover reasoning creates assistant
	if messages[1].Role != "assistant" {
		t.Errorf("second message role = %s, want assistant", messages[1].Role)
	}
	if len(messages[1].ReasoningItems) != 1 {
		t.Error("expected reasoning on created assistant message")
	}
}

// --- Issue 3: Assistant multipart content in Chat → Responses ---

func TestConvertAssistantMessageToInput_MultipartContent(t *testing.T) {
	msg := types.OpenAIMessage{
		Role: "assistant",
		Content: &types.OpenAIContent{
			Parts: []types.OpenAIContentPart{
				{Type: "text", Text: "Hello "},
				{Type: "text", Text: "world"},
			},
		},
	}
	items := convertAssistantMessageToInput(msg)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	// Content should be []ResponseInputContent
	contentItems, ok := items[0].Content.([]types.ResponseInputContent)
	if !ok {
		t.Fatalf("expected []ResponseInputContent, got %T", items[0].Content)
	}
	if len(contentItems) != 2 {
		t.Fatalf("expected 2 content items, got %d", len(contentItems))
	}
	if contentItems[0].Type != "output_text" || contentItems[0].Text != "Hello " {
		t.Errorf("first content item: %+v", contentItems[0])
	}
	if contentItems[1].Type != "output_text" || contentItems[1].Text != "world" {
		t.Errorf("second content item: %+v", contentItems[1])
	}
}

func TestConvertAssistantMessageToInput_TextContent(t *testing.T) {
	text := "just text"
	msg := types.OpenAIMessage{
		Role:    "assistant",
		Content: &types.OpenAIContent{Text: &text},
	}
	items := convertAssistantMessageToInput(msg)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	s, ok := items[0].Content.(string)
	if !ok {
		t.Fatalf("expected string content, got %T", items[0].Content)
	}
	if s != "just text" {
		t.Errorf("content = %q, want 'just text'", s)
	}
}

func TestConvertAssistantMessageToInput_EmptyParts(t *testing.T) {
	msg := types.OpenAIMessage{
		Role: "assistant",
		Content: &types.OpenAIContent{
			Parts: []types.OpenAIContentPart{
				{Type: "image_url"}, // not text
			},
		},
	}
	items := convertAssistantMessageToInput(msg)
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	s, ok := items[0].Content.(string)
	if !ok {
		t.Fatalf("expected string fallback, got %T", items[0].Content)
	}
	if s != "" {
		t.Errorf("content = %q, want empty string", s)
	}
}

// --- Issue 4: Failed response error propagation (streaming) ---

func TestResponsesFailedToChatChunk_WithErrorMessage(t *testing.T) {
	state := NewResponsesStreamConvertState()
	state.ID = "resp_1"
	state.Model = "gpt-4"

	event := types.ResponseStreamEvent{
		Type: "response.failed",
		Response: &types.ResponsesResult{
			Error: &types.ResponseError{Message: "rate limit exceeded"},
		},
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}
	// First chunk has error content
	if chunks[0].Choices[0].Delta.Content == nil {
		t.Fatal("first chunk should have content")
	}
	if *chunks[0].Choices[0].Delta.Content.Text != "rate limit exceeded" {
		t.Errorf("error message = %q, want 'rate limit exceeded'", *chunks[0].Choices[0].Delta.Content.Text)
	}
	// Second chunk has finish
	if chunks[1].Choices[0].FinishReason != "stop" {
		t.Errorf("finish reason = %q, want 'stop'", chunks[1].Choices[0].FinishReason)
	}
}

func TestResponsesFailedToChatChunk_WithEventMessage(t *testing.T) {
	state := NewResponsesStreamConvertState()
	event := types.ResponseStreamEvent{
		Type:    "error",
		Message: "internal error",
	}

	chunks := ConvertResponsesStreamEventToChatChunk(event, state)
	if len(chunks) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(chunks))
	}
	if *chunks[0].Choices[0].Delta.Content.Text != "internal error" {
		t.Errorf("error message = %q, want 'internal error'", *chunks[0].Choices[0].Delta.Content.Text)
	}
}

// --- Issue 6: ParallelToolCalls pointer semantics ---

func TestParallelToolCalls_NilPassthrough(t *testing.T) {
	// When not set in chat request, should remain nil in responses request
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
	}
	result := ConvertChatToResponsesRequest(req)
	if result.ParallelToolCalls != nil {
		t.Error("ParallelToolCalls should be nil when not set")
	}
}

func TestParallelToolCalls_TruePreserved(t *testing.T) {
	ptc := true
	req := types.OpenAIChatCompletionsRequest{
		Model:             "gpt-4",
		ParallelToolCalls: &ptc,
	}
	result := ConvertChatToResponsesRequest(req)
	if result.ParallelToolCalls == nil || !*result.ParallelToolCalls {
		t.Error("ParallelToolCalls should be true")
	}
}

func TestParallelToolCalls_FalsePreserved(t *testing.T) {
	ptc := false
	req := types.OpenAIChatCompletionsRequest{
		Model:             "gpt-4",
		ParallelToolCalls: &ptc,
	}
	result := ConvertChatToResponsesRequest(req)
	if result.ParallelToolCalls == nil || *result.ParallelToolCalls {
		t.Error("ParallelToolCalls should be false")
	}
}

func TestParallelToolCalls_ResponsesToChatNilPassthrough(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
		Input: []types.ResponseInputItem{
			{Type: "message", Role: "user", Content: "Hi"},
		},
	}
	result := ConvertResponsesToChatRequest(req)
	if result.ParallelToolCalls != nil {
		t.Error("ParallelToolCalls should be nil when not set")
	}
}

func TestParallelToolCalls_OmittedInJSON(t *testing.T) {
	req := types.ResponsesRequest{
		Model: "gpt-4",
	}
	data, _ := json.Marshal(req)
	var m map[string]interface{}
	json.Unmarshal(data, &m)
	if _, exists := m["parallel_tool_calls"]; exists {
		t.Error("parallel_tool_calls should be omitted from JSON when nil")
	}
}

// --- Issue 7: User and stop field handling ---

func TestConvertChatToResponsesRequest_UserPassedAsMetadata(t *testing.T) {
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
		User:  "user_123",
	}
	result := ConvertChatToResponsesRequest(req)
	if result.Metadata == nil {
		t.Fatal("Metadata should not be nil")
	}
	if result.Metadata["user"] != "user_123" {
		t.Errorf("metadata user = %q, want 'user_123'", result.Metadata["user"])
	}
}

func TestConvertChatToResponsesRequest_UserMergesWithExistingMetadata(t *testing.T) {
	req := types.OpenAIChatCompletionsRequest{
		Model:    "gpt-4",
		User:     "user_456",
		Metadata: map[string]string{"key": "value"},
	}
	result := ConvertChatToResponsesRequest(req)
	if result.Metadata["user"] != "user_456" {
		t.Errorf("metadata user = %q, want 'user_456'", result.Metadata["user"])
	}
	if result.Metadata["key"] != "value" {
		t.Errorf("metadata key = %q, want 'value'", result.Metadata["key"])
	}
}

func TestConvertChatToResponsesRequest_NoUserNoMetadata(t *testing.T) {
	req := types.OpenAIChatCompletionsRequest{
		Model: "gpt-4",
	}
	result := ConvertChatToResponsesRequest(req)
	// Metadata should remain nil (or empty) when no user is set
	if result.Metadata != nil && result.Metadata["user"] != "" {
		t.Error("should not add user metadata when user is empty")
	}
}
