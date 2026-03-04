package anthropic

import (
	"testing"
)

func TestTranslateResponsesStreamEvent_Created(t *testing.T) {
	state := NewResponsesStreamState()

	event := ResponseStreamEvent{
		Type: "response.created",
		Response: &ResponsesResult{
			ID:    "resp_001",
			Model: "gpt-4o",
			Usage: &ResponsesUsage{
				InputTokens:        100,
				InputTokensDetails: &InputTokenDetails{CachedTokens: 30},
			},
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	e := events[0]
	if e.Type != "message_start" {
		t.Errorf("Expected event type 'message_start', got %q", e.Type)
	}

	if e.Message == nil {
		t.Fatal("Expected message in message_start event")
	}

	if e.Message.ID != "resp_001" {
		t.Errorf("Expected ID 'resp_001', got %q", e.Message.ID)
	}

	if e.Message.Model != "gpt-4o" {
		t.Errorf("Expected model 'gpt-4o', got %q", e.Message.Model)
	}

	if e.Message.Role != "assistant" {
		t.Errorf("Expected role 'assistant', got %q", e.Message.Role)
	}

	if e.Message.Usage.InputTokens != 70 {
		t.Errorf("Expected input tokens 70 (100-30), got %d", e.Message.Usage.InputTokens)
	}

	if e.Message.Usage.CacheReadInputTokens != 30 {
		t.Errorf("Expected cache read input tokens 30, got %d", e.Message.Usage.CacheReadInputTokens)
	}

	if len(e.Message.Content) != 0 {
		t.Errorf("Expected empty content array, got %d items", len(e.Message.Content))
	}

	if !state.MessageStartSent {
		t.Error("Expected MessageStartSent to be true")
	}
}

func TestTranslateResponsesStreamEvent_CreatedNilResponse(t *testing.T) {
	state := NewResponsesStreamState()

	event := ResponseStreamEvent{
		Type: "response.created",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	if events[0].Message.ID != "" {
		t.Errorf("Expected empty ID, got %q", events[0].Message.ID)
	}

	if events[0].Message.Usage.InputTokens != 0 {
		t.Errorf("Expected 0 input tokens, got %d", events[0].Message.Usage.InputTokens)
	}
}

func TestTranslateResponsesStreamEvent_OutputTextDelta(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  0,
		ContentIndex: 0,
		Delta:        "Hello, world!",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 2 {
		t.Fatalf("Expected 2 events (start + delta), got %d", len(events))
	}

	// First: content_block_start
	startEvent := events[0]
	if startEvent.Type != "content_block_start" {
		t.Errorf("Expected 'content_block_start', got %q", startEvent.Type)
	}

	if startEvent.Index == nil || *startEvent.Index != 0 {
		t.Errorf("Expected index 0, got %v", startEvent.Index)
	}

	if startEvent.ContentBlock == nil || startEvent.ContentBlock.Type != "text" {
		t.Error("Expected text content block")
	}

	// Second: content_block_delta
	deltaEvent := events[1]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", deltaEvent.Type)
	}

	if deltaEvent.Delta == nil || deltaEvent.Delta.Type != "text_delta" {
		t.Error("Expected text_delta")
	}

	if deltaEvent.Delta.Text != "Hello, world!" {
		t.Errorf("Expected text 'Hello, world!', got %q", deltaEvent.Delta.Text)
	}

	if !state.BlockHasDelta[0] {
		t.Error("Expected BlockHasDelta[0] to be true")
	}
}

func TestTranslateResponsesStreamEvent_OutputTextDeltaEmpty(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil events for empty delta, got %d", len(events))
	}
}

func TestTranslateResponsesStreamEvent_OutputTextDeltaSubsequent(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// First delta opens the block
	event1 := ResponseStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  0,
		ContentIndex: 0,
		Delta:        "Hello",
	}
	TranslateResponsesStreamEvent(event1, state)

	// Second delta reuses existing block
	event2 := ResponseStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  0,
		ContentIndex: 0,
		Delta:        " world",
	}
	events := TranslateResponsesStreamEvent(event2, state)

	if len(events) != 1 {
		t.Fatalf("Expected 1 event (delta only), got %d", len(events))
	}

	if events[0].Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", events[0].Type)
	}

	if events[0].Delta.Text != " world" {
		t.Errorf("Expected text ' world', got %q", events[0].Delta.Text)
	}
}

func TestTranslateResponsesStreamEvent_ReasoningSummaryTextDelta(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:        "response.reasoning_summary_text.delta",
		OutputIndex: 0,
		Delta:       "Let me think...",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 2 {
		t.Fatalf("Expected 2 events (start + delta), got %d", len(events))
	}

	startEvent := events[0]
	if startEvent.Type != "content_block_start" {
		t.Errorf("Expected 'content_block_start', got %q", startEvent.Type)
	}

	if startEvent.ContentBlock == nil || startEvent.ContentBlock.Type != "thinking" {
		t.Error("Expected thinking content block")
	}

	deltaEvent := events[1]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", deltaEvent.Type)
	}

	if deltaEvent.Delta == nil || deltaEvent.Delta.Type != "thinking_delta" {
		t.Error("Expected thinking_delta")
	}

	if deltaEvent.Delta.Thinking != "Let me think..." {
		t.Errorf("Expected thinking 'Let me think...', got %q", deltaEvent.Delta.Thinking)
	}

	if !state.BlockHasDelta[0] {
		t.Error("Expected BlockHasDelta[0] to be true")
	}
}

func TestTranslateResponsesStreamEvent_ReasoningSummaryTextDeltaEmpty(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:  "response.reasoning_summary_text.delta",
		Delta: "",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil events for empty delta, got %d", len(events))
	}
}

func TestTranslateResponsesStreamEvent_FunctionCallArgumentsDelta(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// First: output_item.added to set up function call state
	addedEvent := ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:   "function_call",
			CallID: "call_abc",
			Name:   "search",
		},
	}
	TranslateResponsesStreamEvent(addedEvent, state)

	// Now: arguments delta
	event := ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"query":`,
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	e := events[0]
	if e.Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", e.Type)
	}

	if e.Delta == nil || e.Delta.Type != "input_json_delta" {
		t.Error("Expected input_json_delta")
	}

	if e.Delta.PartialJSON != `{"query":` {
		t.Errorf("Expected partial JSON '{\"query\":', got %q", e.Delta.PartialJSON)
	}
}

func TestTranslateResponsesStreamEvent_FunctionCallArgumentsDeltaEmpty(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:  "response.function_call_arguments.delta",
		Delta: "",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil events for empty delta, got %d", len(events))
	}
}

func TestTranslateResponsesStreamEvent_OutputItemDoneReasoning(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:             "reasoning",
			ID:               "reasoning_1",
			EncryptedContent: "encrypted_data",
			Summary:          []ResponseSummaryBlock{},
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Expect: content_block_start (thinking), thinking_delta (default text), signature_delta
	if len(events) != 3 {
		t.Fatalf("Expected 3 events, got %d", len(events))
	}

	// First: content_block_start for thinking
	if events[0].Type != "content_block_start" {
		t.Errorf("Expected 'content_block_start', got %q", events[0].Type)
	}

	if events[0].ContentBlock == nil || events[0].ContentBlock.Type != "thinking" {
		t.Error("Expected thinking content block")
	}

	// Second: thinking_delta with default text (since summary is empty)
	if events[1].Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", events[1].Type)
	}

	if events[1].Delta.Type != "thinking_delta" {
		t.Errorf("Expected 'thinking_delta', got %q", events[1].Delta.Type)
	}

	if events[1].Delta.Thinking != ThinkingText {
		t.Errorf("Expected default thinking text %q, got %q", ThinkingText, events[1].Delta.Thinking)
	}

	// Third: signature_delta
	if events[2].Delta.Type != "signature_delta" {
		t.Errorf("Expected 'signature_delta', got %q", events[2].Delta.Type)
	}

	expectedSig := "encrypted_data@reasoning_1"
	if events[2].Delta.Signature != expectedSig {
		t.Errorf("Expected signature %q, got %q", expectedSig, events[2].Delta.Signature)
	}
}

func TestTranslateResponsesStreamEvent_OutputItemDoneReasoningWithSummary(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// First send a summary delta so the block has content
	summaryDelta := ResponseStreamEvent{
		Type:        "response.reasoning_summary_text.delta",
		OutputIndex: 0,
		Delta:       "Already thinking...",
	}
	TranslateResponsesStreamEvent(summaryDelta, state)

	// Now output_item.done with non-empty summary
	event := ResponseStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:             "reasoning",
			ID:               "reasoning_1",
			EncryptedContent: "encrypted_data",
			Summary: []ResponseSummaryBlock{
				{Type: "summary_text", Text: "Already thinking..."},
			},
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Should NOT include default thinking text since summary was non-empty
	for _, e := range events {
		if e.Delta != nil && e.Delta.Type == "thinking_delta" && e.Delta.Thinking == ThinkingText {
			t.Error("Should not emit default thinking text when summary is present")
		}
	}

	// Should include signature
	found := false
	for _, e := range events {
		if e.Delta != nil && e.Delta.Type == "signature_delta" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected signature_delta event")
	}
}

func TestTranslateResponsesStreamEvent_OutputItemDoneNonReasoning(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type: "message",
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil events for non-reasoning output_item.done, got %d", len(events))
	}
}

func TestTranslateResponsesStreamEvent_Completed(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// Open a text block first
	textEvent := ResponseStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  0,
		ContentIndex: 0,
		Delta:        "Hello",
	}
	TranslateResponsesStreamEvent(textEvent, state)

	// Now complete
	event := ResponseStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResult{
			ID:     "resp_001",
			Model:  "gpt-4o",
			Status: "completed",
			Output: []ResponseOutputItem{
				{
					Type:    "message",
					Content: []ResponseOutputContent{{Type: "output_text", Text: "Hello"}},
				},
			},
			Usage: &ResponsesUsage{
				InputTokens:  100,
				OutputTokens: 25,
			},
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Expect: content_block_stop, message_delta, message_stop
	if len(events) != 3 {
		t.Fatalf("Expected 3 events, got %d", len(events))
	}

	// First: content_block_stop for the open text block
	if events[0].Type != "content_block_stop" {
		t.Errorf("Expected 'content_block_stop', got %q", events[0].Type)
	}

	// Second: message_delta with stop reason and usage
	if events[1].Type != "message_delta" {
		t.Errorf("Expected 'message_delta', got %q", events[1].Type)
	}

	if events[1].Delta == nil || events[1].Delta.StopReason != "end_turn" {
		t.Errorf("Expected stop reason 'end_turn', got %q", events[1].Delta.StopReason)
	}

	if events[1].Usage == nil {
		t.Fatal("Expected usage in message_delta")
	}

	deltaUsage, ok := events[1].Usage.(*AnthropicMessageDeltaUsage)
	if !ok {
		t.Fatalf("Expected *AnthropicMessageDeltaUsage, got %T", events[1].Usage)
	}

	if deltaUsage.OutputTokens != 25 {
		t.Errorf("Expected output tokens 25, got %d", deltaUsage.OutputTokens)
	}

	// Third: message_stop
	if events[2].Type != "message_stop" {
		t.Errorf("Expected 'message_stop', got %q", events[2].Type)
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true")
	}
}

func TestTranslateResponsesStreamEvent_CompletedNilResponse(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type: "response.completed",
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Expect: message_delta, message_stop
	if len(events) != 2 {
		t.Fatalf("Expected 2 events, got %d", len(events))
	}

	if events[0].Type != "message_delta" {
		t.Errorf("Expected 'message_delta', got %q", events[0].Type)
	}

	if events[0].Delta.StopReason != "end_turn" {
		t.Errorf("Expected stop reason 'end_turn', got %q", events[0].Delta.StopReason)
	}

	if events[1].Type != "message_stop" {
		t.Errorf("Expected 'message_stop', got %q", events[1].Type)
	}
}

func TestTranslateResponsesStreamEvent_OutputItemAddedFunctionCall(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:      "function_call",
			CallID:    "call_abc",
			Name:      "search",
			Arguments: `{"q":"test"}`,
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Expect: content_block_start (tool_use) + content_block_delta (initial args)
	if len(events) != 2 {
		t.Fatalf("Expected 2 events, got %d", len(events))
	}

	startEvent := events[0]
	if startEvent.Type != "content_block_start" {
		t.Errorf("Expected 'content_block_start', got %q", startEvent.Type)
	}

	if startEvent.ContentBlock == nil || startEvent.ContentBlock.Type != "tool_use" {
		t.Error("Expected tool_use content block")
	}

	if startEvent.ContentBlock.ID != "call_abc" {
		t.Errorf("Expected tool ID 'call_abc', got %q", startEvent.ContentBlock.ID)
	}

	if startEvent.ContentBlock.Name != "search" {
		t.Errorf("Expected tool name 'search', got %q", startEvent.ContentBlock.Name)
	}

	// Second event: initial arguments delta
	deltaEvent := events[1]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected 'content_block_delta', got %q", deltaEvent.Type)
	}

	if deltaEvent.Delta.PartialJSON != `{"q":"test"}` {
		t.Errorf("Expected partial JSON '{\"q\":\"test\"}', got %q", deltaEvent.Delta.PartialJSON)
	}
}

func TestTranslateResponsesStreamEvent_OutputItemAddedNonFunctionCall(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type: "message",
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil events for non-function_call output_item.added, got %d", len(events))
	}
}

func TestTranslateResponsesStreamEvent_Failed(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	event := ResponseStreamEvent{
		Type: "response.failed",
		Response: &ResponsesResult{
			Error: &ResponseError{
				Message: "rate limit exceeded",
			},
		},
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	if events[0].Type != "error" {
		t.Errorf("Expected 'error', got %q", events[0].Type)
	}

	if events[0].Error == nil || events[0].Error.Message != "rate limit exceeded" {
		t.Errorf("Expected error message 'rate limit exceeded', got %q", events[0].Error.Message)
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true")
	}
}

func TestTranslateResponsesStreamEvent_Error(t *testing.T) {
	state := NewResponsesStreamState()

	event := ResponseStreamEvent{
		Type:    "error",
		Message: "something went wrong",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if len(events) != 3 {
		t.Fatalf("Expected 3 events, got %d", len(events))
	}

	if events[0].Type != "message_delta" {
		t.Errorf("Expected 'message_delta', got %q", events[0].Type)
	}

	if events[1].Type != "message_stop" {
		t.Errorf("Expected 'message_stop', got %q", events[1].Type)
	}

	if events[2].Type != "error" {
		t.Errorf("Expected 'error', got %q", events[2].Type)
	}

	if events[2].Error.Message != "something went wrong" {
		t.Errorf("Expected 'something went wrong', got %q", events[2].Error.Message)
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true")
	}
}

func TestTranslateResponsesStreamEvent_UnknownType(t *testing.T) {
	state := NewResponsesStreamState()

	event := ResponseStreamEvent{
		Type: "response.unknown_event",
	}

	events := TranslateResponsesStreamEvent(event, state)

	if events != nil {
		t.Errorf("Expected nil for unknown event type, got %d events", len(events))
	}
}

func TestTranslateResponsesStreamEvent_ComplexFlow(t *testing.T) {
	state := NewResponsesStreamState()

	// 1. response.created
	events1 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type: "response.created",
		Response: &ResponsesResult{
			ID:    "resp_001",
			Model: "gpt-4o",
			Usage: &ResponsesUsage{InputTokens: 50},
		},
	}, state)

	if len(events1) != 1 || events1[0].Type != "message_start" {
		t.Error("Expected message_start event")
	}

	// 2. reasoning_summary_text.delta
	events2 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.reasoning_summary_text.delta",
		OutputIndex: 0,
		Delta:       "Analyzing...",
	}, state)

	if len(events2) != 2 {
		t.Fatalf("Expected 2 thinking events, got %d", len(events2))
	}

	// 3. output_item.done for reasoning (closes thinking)
	events3 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.output_item.done",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:             "reasoning",
			ID:               "r1",
			EncryptedContent: "enc",
			Summary:          []ResponseSummaryBlock{{Type: "summary_text", Text: "Analyzing..."}},
		},
	}, state)

	// Should include signature_delta
	hasSig := false
	for _, e := range events3 {
		if e.Delta != nil && e.Delta.Type == "signature_delta" {
			hasSig = true
		}
	}
	if !hasSig {
		t.Error("Expected signature_delta in output_item.done events")
	}

	// 4. output_text.delta (opens new text block, closes thinking block)
	events4 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:         "response.output_text.delta",
		OutputIndex:  1,
		ContentIndex: 0,
		Delta:        "Here is the answer",
	}, state)

	// Should have: content_block_stop (thinking), content_block_start (text), content_block_delta
	hasStop := false
	hasStart := false
	hasDelta := false
	for _, e := range events4 {
		switch e.Type {
		case "content_block_stop":
			hasStop = true
		case "content_block_start":
			hasStart = true
		case "content_block_delta":
			hasDelta = true
		}
	}
	if !hasStop {
		t.Error("Expected content_block_stop for thinking block")
	}
	if !hasStart {
		t.Error("Expected content_block_start for text block")
	}
	if !hasDelta {
		t.Error("Expected content_block_delta for text")
	}

	// 5. response.completed
	events5 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResult{
			ID:     "resp_001",
			Model:  "gpt-4o",
			Status: "completed",
			Output: []ResponseOutputItem{
				{Type: "message", Content: []ResponseOutputContent{{Type: "output_text", Text: "Here is the answer"}}},
			},
			Usage: &ResponsesUsage{InputTokens: 50, OutputTokens: 20},
		},
	}, state)

	// Last event should be message_stop
	lastEvent := events5[len(events5)-1]
	if lastEvent.Type != "message_stop" {
		t.Errorf("Expected final event 'message_stop', got %q", lastEvent.Type)
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true")
	}
}

func TestTranslateResponsesStreamEvent_FunctionCallWhitespaceLimit(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// Set up function call via output_item.added
	TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:   "function_call",
			CallID: "call_1",
			Name:   "test_fn",
		},
	}, state)

	// Send delta with excessive whitespace (>20 consecutive newlines)
	event := ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
	}

	events := TranslateResponsesStreamEvent(event, state)

	// Should produce error
	hasError := false
	for _, e := range events {
		if e.Type == "error" {
			hasError = true
		}
	}
	if !hasError {
		t.Error("Expected error event for excessive whitespace")
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true after whitespace limit exceeded")
	}
}

func TestTranslateResponsesStreamEvent_ParallelToolCalls(t *testing.T) {
	state := NewResponsesStreamState()
	state.MessageStartSent = true

	// Add two function calls at different output indices
	TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 0,
		Item: &ResponseOutputItem{
			Type:   "function_call",
			CallID: "call_1",
			Name:   "search",
		},
	}, state)

	TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.output_item.added",
		OutputIndex: 1,
		Item: &ResponseOutputItem{
			Type:   "function_call",
			CallID: "call_2",
			Name:   "read_file",
		},
	}, state)

	// Both blocks should be open
	if len(state.OpenBlocks) != 2 {
		t.Fatalf("Expected 2 open blocks, got %d", len(state.OpenBlocks))
	}

	// Send interleaved deltas for both tool calls
	events1 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `{"query":`,
	}, state)

	if len(events1) != 1 || events1[0].Type != "content_block_delta" {
		t.Fatalf("Expected 1 content_block_delta for call_1, got %d events", len(events1))
	}
	if *events1[0].Index != 0 {
		t.Errorf("Expected block index 0 for call_1, got %d", *events1[0].Index)
	}

	events2 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 1,
		Delta:       `{"path":`,
	}, state)

	if len(events2) != 1 || events2[0].Type != "content_block_delta" {
		t.Fatalf("Expected 1 content_block_delta for call_2, got %d events", len(events2))
	}
	if *events2[0].Index != 1 {
		t.Errorf("Expected block index 1 for call_2, got %d", *events2[0].Index)
	}

	// Both blocks should still be open
	if len(state.OpenBlocks) != 2 {
		t.Fatalf("Expected 2 open blocks after interleaved deltas, got %d", len(state.OpenBlocks))
	}

	// Send more interleaved deltas — back to call_1
	events3 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.function_call_arguments.delta",
		OutputIndex: 0,
		Delta:       `"go"}`,
	}, state)

	if len(events3) != 1 || events3[0].Type != "content_block_delta" {
		t.Fatalf("Expected 1 content_block_delta for call_1 second delta, got %d events", len(events3))
	}
	if *events3[0].Index != 0 {
		t.Errorf("Expected block index 0 for call_1 second delta, got %d", *events3[0].Index)
	}

	// Complete both
	TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.function_call_arguments.done",
		OutputIndex: 0,
		Arguments:   `{"query":"go"}`,
	}, state)

	TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type:        "response.function_call_arguments.done",
		OutputIndex: 1,
		Arguments:   `{"path":"main.go"}`,
	}, state)

	// Complete stream
	events5 := TranslateResponsesStreamEvent(ResponseStreamEvent{
		Type: "response.completed",
		Response: &ResponsesResult{
			ID:     "resp_parallel",
			Model:  "gpt-4o",
			Status: "completed",
			Output: []ResponseOutputItem{
				{Type: "function_call", CallID: "call_1", Name: "search", Arguments: `{"query":"go"}`},
				{Type: "function_call", CallID: "call_2", Name: "read_file", Arguments: `{"path":"main.go"}`},
			},
			Usage: &ResponsesUsage{InputTokens: 50, OutputTokens: 20},
		},
	}, state)

	// Should have content_block_stop for both blocks + message_delta + message_stop
	stopCount := 0
	for _, e := range events5 {
		if e.Type == "content_block_stop" {
			stopCount++
		}
	}
	if stopCount != 2 {
		t.Errorf("Expected 2 content_block_stop events, got %d", stopCount)
	}

	if !state.MessageCompleted {
		t.Error("Expected MessageCompleted to be true")
	}
}
