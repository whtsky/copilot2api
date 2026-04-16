package anthropic

import (
	"testing"
)

func TestConvertOpenAIChunkToAnthropicEvents_MessageStart(t *testing.T) {
	chunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{},
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 0,
		},
	}

	state := NewStreamState()
	events, err := ConvertOpenAIChunkToAnthropicEvents(chunk, state)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	event := events[0]
	if event.Type != "message_start" {
		t.Errorf("Expected event type 'message_start', got %q", event.Type)
	}

	if event.Message == nil {
		t.Fatal("Expected message in message_start event")
	}

	if event.Message.ID != "msg_123" {
		t.Errorf("Expected message ID 'msg_123', got %q", event.Message.ID)
	}

	if len(event.Message.Content) != 0 {
		t.Errorf("Expected empty content array, got %d items", len(event.Message.Content))
	}

	if event.Message.Usage.InputTokens != 10 {
		t.Errorf("Expected input tokens 10, got %d", event.Message.Usage.InputTokens)
	}

	// Check state
	if !state.MessageStartSent {
		t.Error("Expected MessageStartSent to be true")
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_TextContent(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true // Skip message start

	chunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					Content: &OpenAIContent{
						Text: stringPtr("Hello"),
					},
				},
			},
		},
	}

	events, err := ConvertOpenAIChunkToAnthropicEvents(chunk, state)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if len(events) != 2 {
		t.Fatalf("Expected 2 events (start + delta), got %d", len(events))
	}

	// First event should be content_block_start
	startEvent := events[0]
	if startEvent.Type != "content_block_start" {
		t.Errorf("Expected first event type 'content_block_start', got %q", startEvent.Type)
	}

	if startEvent.Index == nil || *startEvent.Index != 0 {
		t.Errorf("Expected index 0, got %v", startEvent.Index)
	}

	if startEvent.ContentBlock == nil || startEvent.ContentBlock.Type != "text" {
		t.Errorf("Expected text content block")
	}

	// Second event should be content_block_delta
	deltaEvent := events[1]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected second event type 'content_block_delta', got %q", deltaEvent.Type)
	}

	cd := contentDelta(t, deltaEvent)
	if cd.Type != "text_delta" {
		t.Errorf("Expected text_delta")
	}

	if cd.Text != "Hello" {
		t.Errorf("Expected text 'Hello', got %q", cd.Text)
	}

	// Check state
	if !state.ContentBlockOpen {
		t.Error("Expected ContentBlockOpen to be true")
	}

	if state.ContentBlockIndex != 0 {
		t.Errorf("Expected ContentBlockIndex 0, got %d", state.ContentBlockIndex)
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_ToolCall(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true

	// First chunk - tool call starts
	chunk1 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					ToolCalls: []OpenAIToolCall{
						{
							Index: intPtr(0),
							ID:    "call_1",
							Function: OpenAIToolCallFunction{
								Name: "search",
							},
						},
					},
				},
			},
		},
	}

	events1, err := ConvertOpenAIChunkToAnthropicEvents(chunk1, state)
	if err != nil {
		t.Fatalf("First conversion failed: %v", err)
	}

	if len(events1) != 1 {
		t.Fatalf("Expected 1 event from first chunk, got %d", len(events1))
	}

	// Should be content_block_start for tool_use
	event := events1[0]
	if event.Type != "content_block_start" {
		t.Errorf("Expected event type 'content_block_start', got %q", event.Type)
	}

	if event.ContentBlock == nil || event.ContentBlock.Type != "tool_use" {
		t.Errorf("Expected tool_use content block")
	}

	if event.ContentBlock.ID != "call_1" {
		t.Errorf("Expected tool ID 'call_1', got %q", event.ContentBlock.ID)
	}

	if event.ContentBlock.Name != "search" {
		t.Errorf("Expected tool name 'search', got %q", event.ContentBlock.Name)
	}

	// Second chunk - tool arguments
	chunk2 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					ToolCalls: []OpenAIToolCall{
						{
							Index: intPtr(0),
							Function: OpenAIToolCallFunction{
								Arguments: `{"query":`,
							},
						},
					},
				},
			},
		},
	}

	events2, err := ConvertOpenAIChunkToAnthropicEvents(chunk2, state)
	if err != nil {
		t.Fatalf("Second conversion failed: %v", err)
	}

	if len(events2) != 1 {
		t.Fatalf("Expected 1 event from second chunk, got %d", len(events2))
	}

	// Should be content_block_delta with input_json_delta
	deltaEvent := events2[0]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected event type 'content_block_delta', got %q", deltaEvent.Type)
	}

	cd := contentDelta(t, deltaEvent)
	if cd.Type != "input_json_delta" {
		t.Errorf("Expected input_json_delta")
	}

	if cd.PartialJSON != `{"query":` {
		t.Errorf("Expected partial JSON '{\"query\":', got %q", cd.PartialJSON)
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_ThinkingText(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true

	chunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					ReasoningText: stringPtr("Let me think about this..."),
				},
			},
		},
	}

	events, err := ConvertOpenAIChunkToAnthropicEvents(chunk, state)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if len(events) != 2 {
		t.Fatalf("Expected 2 events (start + delta), got %d", len(events))
	}

	// First event should be content_block_start for thinking
	startEvent := events[0]
	if startEvent.Type != "content_block_start" {
		t.Errorf("Expected first event type 'content_block_start', got %q", startEvent.Type)
	}

	if startEvent.ContentBlock == nil || startEvent.ContentBlock.Type != "thinking" {
		t.Errorf("Expected thinking content block")
	}

	// Second event should be content_block_delta
	deltaEvent := events[1]
	if deltaEvent.Type != "content_block_delta" {
		t.Errorf("Expected second event type 'content_block_delta', got %q", deltaEvent.Type)
	}

	cd := contentDelta(t, deltaEvent)
	if cd.Type != "thinking_delta" {
		t.Errorf("Expected thinking_delta")
	}

	if cd.Thinking != "Let me think about this..." {
		t.Errorf("Expected thinking text 'Let me think about this...', got %q", cd.Thinking)
	}

	// Check state
	if !state.ThinkingBlockOpen {
		t.Error("Expected ThinkingBlockOpen to be true")
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_Finish(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true
	state.ContentBlockOpen = true
	state.ContentBlockIndex = 0

	chunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index:        0,
				Delta:        OpenAIMessage{},
				FinishReason: "stop",
			},
		},
		Usage: &OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 25,
		},
	}

	events, err := ConvertOpenAIChunkToAnthropicEvents(chunk, state)
	if err != nil {
		t.Fatalf("Conversion failed: %v", err)
	}

	if len(events) != 3 {
		t.Fatalf("Expected 3 events (stop + delta + stop), got %d", len(events))
	}

	// First event should be content_block_stop
	stopEvent := events[0]
	if stopEvent.Type != "content_block_stop" {
		t.Errorf("Expected first event type 'content_block_stop', got %q", stopEvent.Type)
	}

	if stopEvent.Index == nil || *stopEvent.Index != 0 {
		t.Errorf("Expected stop index 0, got %v", func() interface{} {
			if stopEvent.Index != nil {
				return *stopEvent.Index
			}
			return nil
		}())
	}

	// Second event should be message_delta
	deltaEvent := events[1]
	if deltaEvent.Type != "message_delta" {
		t.Errorf("Expected second event type 'message_delta', got %q", deltaEvent.Type)
	}

	if deltaEvent.Usage == nil {
		t.Fatal("Expected usage in message_delta")
	}

	deltaUsage, ok := deltaEvent.Usage.(*AnthropicMessageDeltaUsage)
	if !ok {
		t.Fatalf("Expected *AnthropicMessageDeltaUsage, got %T", deltaEvent.Usage)
	}

	if deltaUsage.OutputTokens != 25 {
		t.Errorf("Expected output tokens 25, got %d", deltaUsage.OutputTokens)
	}

	// Third event should be message_stop
	messageStopEvent := events[2]
	if messageStopEvent.Type != "message_stop" {
		t.Errorf("Expected third event type 'message_stop', got %q", messageStopEvent.Type)
	}

	// Check state
	if state.ContentBlockOpen {
		t.Error("Expected ContentBlockOpen to be false")
	}

	if state.ContentBlockIndex != 1 {
		t.Errorf("Expected ContentBlockIndex 1, got %d", state.ContentBlockIndex)
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_ThinkingFinishWithSignature(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true

	thinkingChunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{{
			Index: 0,
			Delta: OpenAIMessage{
				ReasoningText: stringPtr("Let me think about this..."),
			},
		}},
	}
	if _, err := ConvertOpenAIChunkToAnthropicEvents(thinkingChunk, state); err != nil {
		t.Fatalf("Thinking conversion failed: %v", err)
	}

	finishChunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{{
			Index: 0,
			Delta: OpenAIMessage{
				ReasoningOpaque: stringPtr("sig_123"),
			},
			FinishReason: "stop",
		}},
		Usage: &OpenAIUsage{PromptTokens: 10, CompletionTokens: 25},
	}

	events, err := ConvertOpenAIChunkToAnthropicEvents(finishChunk, state)
	if err != nil {
		t.Fatalf("Finish conversion failed: %v", err)
	}

	if len(events) != 4 {
		t.Fatalf("Expected 4 events (signature + stop + delta + stop), got %d", len(events))
	}

	cd := contentDelta(t, events[0])
	if cd.Type != "signature_delta" {
		t.Fatalf("Expected first delta to be signature_delta, got %q", cd.Type)
	}
	if cd.Signature != "sig_123" {
		t.Fatalf("Expected signature sig_123, got %q", cd.Signature)
	}

	if events[1].Type != "content_block_stop" {
		t.Fatalf("Expected second event content_block_stop, got %q", events[1].Type)
	}
	if events[2].Type != "message_delta" {
		t.Fatalf("Expected third event message_delta, got %q", events[2].Type)
	}
	if events[3].Type != "message_stop" {
		t.Fatalf("Expected fourth event message_stop, got %q", events[3].Type)
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_ThinkingToToolCallWithSignature(t *testing.T) {
	state := NewStreamState()
	state.MessageStartSent = true

	thinkingChunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{{
			Index: 0,
			Delta: OpenAIMessage{
				ReasoningText: stringPtr("Analyzing..."),
			},
		}},
	}
	if _, err := ConvertOpenAIChunkToAnthropicEvents(thinkingChunk, state); err != nil {
		t.Fatalf("Thinking conversion failed: %v", err)
	}

	toolChunk := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{{
			Index: 0,
			Delta: OpenAIMessage{
				ReasoningOpaque: stringPtr("sig_456"),
				ToolCalls: []OpenAIToolCall{{
					Index: intPtr(0),
					ID:    "call_1",
					Function: OpenAIToolCallFunction{
						Name: "search",
					},
				}},
			},
		}},
	}

	events, err := ConvertOpenAIChunkToAnthropicEvents(toolChunk, state)
	if err != nil {
		t.Fatalf("Tool conversion failed: %v", err)
	}

	if len(events) != 3 {
		t.Fatalf("Expected 3 events (signature + stop + tool_start), got %d", len(events))
	}

	cd := contentDelta(t, events[0])
	if cd.Type != "signature_delta" {
		t.Fatalf("Expected first delta to be signature_delta, got %q", cd.Type)
	}
	if cd.Signature != "sig_456" {
		t.Fatalf("Expected signature sig_456, got %q", cd.Signature)
	}

	if events[1].Type != "content_block_stop" {
		t.Fatalf("Expected second event content_block_stop, got %q", events[1].Type)
	}
	if events[2].Type != "content_block_start" || events[2].ContentBlock == nil || events[2].ContentBlock.Type != "tool_use" {
		t.Fatalf("Expected third event tool_use content_block_start, got %#v", events[2])
	}

	if events[2].Index == nil || *events[2].Index != 1 {
		t.Fatalf("Expected tool block index 1, got %v", events[2].Index)
	}
}

func TestConvertOpenAIChunkToAnthropicEvents_ComplexFlow(t *testing.T) {
	state := NewStreamState()

	// Simulate a complex flow: message start -> thinking -> text -> tool call -> finish

	// 1. Message start
	chunk1 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{Index: 0, Delta: OpenAIMessage{}},
		},
		Usage: &OpenAIUsage{PromptTokens: 10},
	}

	events1, _ := ConvertOpenAIChunkToAnthropicEvents(chunk1, state)
	if len(events1) != 1 || events1[0].Type != "message_start" {
		t.Errorf("Expected message_start event")
	}

	// 2. Thinking
	chunk2 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					ReasoningText: stringPtr("Analyzing..."),
				},
			},
		},
	}

	events2, _ := ConvertOpenAIChunkToAnthropicEvents(chunk2, state)
	if len(events2) != 2 {
		t.Fatalf("Expected 2 thinking events, got %d", len(events2))
	}

	// 3. Text content
	chunk3 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					Content: &OpenAIContent{Text: stringPtr("I need to search")},
				},
			},
		},
	}

	events3, _ := ConvertOpenAIChunkToAnthropicEvents(chunk3, state)
	// Should close thinking and start/delta text
	if len(events3) < 3 {
		t.Errorf("Expected at least 3 events for text transition, got %d", len(events3))
	}

	// 4. Tool call
	chunk4 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index: 0,
				Delta: OpenAIMessage{
					ToolCalls: []OpenAIToolCall{
						{
							Index: intPtr(0),
							ID:    "call_1",
							Function: OpenAIToolCallFunction{
								Name: "search",
							},
						},
					},
				},
			},
		},
	}

	events4, _ := ConvertOpenAIChunkToAnthropicEvents(chunk4, state)
	// Should close text and start tool
	if len(events4) < 2 {
		t.Errorf("Expected at least 2 events for tool start, got %d", len(events4))
	}

	// 5. Finish
	chunk5 := OpenAIChatCompletionChunk{
		ID:    "msg_123",
		Model: "claude-3-sonnet-20240229",
		Choices: []OpenAIChunkChoice{
			{
				Index:        0,
				Delta:        OpenAIMessage{},
				FinishReason: "tool_calls",
			},
		},
		Usage: &OpenAIUsage{PromptTokens: 10, CompletionTokens: 15},
	}

	events5, _ := ConvertOpenAIChunkToAnthropicEvents(chunk5, state)
	if len(events5) != 3 {
		t.Fatalf("Expected 3 finish events, got %d", len(events5))
	}

	lastEvent := events5[len(events5)-1]
	if lastEvent.Type != "message_stop" {
		t.Errorf("Expected final event to be 'message_stop', got %q", lastEvent.Type)
	}
}

func TestStreamState_ToolCallTracking(t *testing.T) {
	state := NewStreamState()

	// Add a tool call
	state.ToolCalls[0] = &ToolCallState{
		ID:                  "call_1",
		Name:                "search",
		AnthropicBlockIndex: 1,
	}

	state.ContentBlockIndex = 1
	state.ContentBlockOpen = true

	// Check if tool block is open
	if !isToolBlockOpen(state) {
		t.Error("Expected tool block to be open")
	}

	// Change to different block index
	state.ContentBlockIndex = 2
	if isToolBlockOpen(state) {
		t.Error("Expected tool block to not be open for different index")
	}
}

func TestCreateErrorEvent(t *testing.T) {
	event := CreateErrorEvent("Something went wrong")

	if event.Type != "error" {
		t.Errorf("Expected event type 'error', got %q", event.Type)
	}

	if event.Error == nil || event.Error.Type != AnthropicErrorTypeAPI {
		t.Errorf("Expected error with type %q", AnthropicErrorTypeAPI)
	}

	if event.Error.Message != "Something went wrong" {
		t.Errorf("Expected error message 'Something went wrong', got %q", event.Error.Message)
	}
}