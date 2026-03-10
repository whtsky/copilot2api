package anthropic

import (
	"fmt"
	"log/slog"
	"sort"
)

// TranslateResponsesStreamEvent translates a single Responses API stream event
// into zero or more Anthropic stream events, updating state as needed.
func TranslateResponsesStreamEvent(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	switch event.Type {
	case "response.created":
		return respHandleCreated(event, state)
	case "response.output_item.added":
		return respHandleOutputItemAdded(event, state)
	case "response.output_item.done":
		return respHandleOutputItemDone(event, state)
	case "response.reasoning_summary_text.delta":
		return respHandleReasoningSummaryTextDelta(event, state)
	case "response.reasoning_summary_text.done":
		return respHandleReasoningSummaryTextDone(event, state)
	case "response.output_text.delta":
		return respHandleOutputTextDelta(event, state)
	case "response.output_text.done":
		return respHandleOutputTextDone(event, state)
	case "response.function_call_arguments.delta":
		return respHandleFunctionCallArgsDelta(event, state)
	case "response.function_call_arguments.done":
		return respHandleFunctionCallArgsDone(event, state)
	case "response.completed", "response.incomplete":
		return respHandleCompleted(event, state)
	case "response.failed":
		return respHandleFailed(event, state)
	case "error":
		return respHandleError(event, state)
	default:
		return nil
	}
}

// --- Event handlers ---

func respHandleCreated(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	state.MessageStartSent = true
	resp := event.Response

	var id, model string
	inputTokens, cachedTokens := 0, 0
	if resp != nil {
		id = resp.ID
		model = resp.Model
		if resp.Usage != nil {
			inputTokens = resp.Usage.InputTokens
			if resp.Usage.InputTokensDetails != nil {
				cachedTokens = resp.Usage.InputTokensDetails.CachedTokens
			}
			inputTokens -= cachedTokens
			if inputTokens < 0 {
				inputTokens = 0
			}
		}
	}

	return []AnthropicStreamEvent{{
		Type: "message_start",
		Message: &AnthropicMessagesResponse{
			ID:      id,
			Type:    "message",
			Role:    "assistant",
			Model:   model,
			Content: []AnthropicContentBlock{},
			Usage: AnthropicUsage{
				InputTokens:         inputTokens,
				OutputTokens:        0,
				CacheReadInputTokens: cachedTokens,
			},
		},
	}}
}

func respHandleOutputItemAdded(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	if event.Item == nil || event.Item.Type != "function_call" {
		return nil
	}

	var events []AnthropicStreamEvent
	outputIndex := event.OutputIndex
	toolCallID := event.Item.CallID
	name := event.Item.Name
	initialArgs := event.Item.Arguments

	blockIndex := respOpenFunctionCallBlock(state, outputIndex, toolCallID, name, &events)

	if initialArgs != "" {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_delta",
			Index: intPtr(blockIndex),
			Delta: &AnthropicContentDelta{
				Type:        "input_json_delta",
				PartialJSON: initialArgs,
			},
		})
		state.BlockHasDelta[blockIndex] = true
	}

	return events
}

func respHandleOutputItemDone(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	if event.Item == nil || event.Item.Type != "reasoning" {
		return nil
	}

	var events []AnthropicStreamEvent
	outputIndex := event.OutputIndex
	blockIndex := respOpenThinkingBlock(state, outputIndex, &events)

	signature := event.Item.EncryptedContent + "@" + event.Item.ID

	// If no summary text was streamed, emit an empty thinking delta so the
	// block has at least one delta event (required before signature_delta).
	if len(event.Item.Summary) == 0 {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_delta",
			Index: intPtr(blockIndex),
			Delta: &AnthropicContentDelta{
				Type:     "thinking_delta",
				Thinking: "",
			},
		})
	}

	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: intPtr(blockIndex),
		Delta: &AnthropicContentDelta{
			Type:      "signature_delta",
			Signature: signature,
		},
	})
	state.BlockHasDelta[blockIndex] = true

	return events
}

func respHandleReasoningSummaryTextDelta(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	opener := func(s *ResponsesStreamState, e *[]AnthropicStreamEvent) int {
		return respOpenThinkingBlock(s, event.OutputIndex, e)
	}
	return respEmitDelta(state, event.Delta, opener, &AnthropicContentDelta{
		Type:     "thinking_delta",
		Thinking: event.Delta,
	})
}

func respHandleReasoningSummaryTextDone(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	opener := func(s *ResponsesStreamState, e *[]AnthropicStreamEvent) int {
		return respOpenThinkingBlock(s, event.OutputIndex, e)
	}
	return respEmitDone(state, event.Text, opener, &AnthropicContentDelta{
		Type:     "thinking_delta",
		Thinking: event.Text,
	})
}

func respHandleOutputTextDelta(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	opener := func(s *ResponsesStreamState, e *[]AnthropicStreamEvent) int {
		return respOpenTextBlock(s, event.OutputIndex, event.ContentIndex, e)
	}
	return respEmitDelta(state, event.Delta, opener, &AnthropicContentDelta{
		Type: "text_delta",
		Text: event.Delta,
	})
}

func respHandleOutputTextDone(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	opener := func(s *ResponsesStreamState, e *[]AnthropicStreamEvent) int {
		return respOpenTextBlock(s, event.OutputIndex, event.ContentIndex, e)
	}
	return respEmitDone(state, event.Text, opener, &AnthropicContentDelta{
		Type: "text_delta",
		Text: event.Text,
	})
}

func respHandleFunctionCallArgsDelta(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	if event.Delta == "" {
		return nil
	}

	var events []AnthropicStreamEvent
	outputIndex := event.OutputIndex
	blockIndex := respOpenFunctionCallBlock(state, outputIndex, "", "", &events)

	fcState := state.FunctionCallStateByOutputIndex[outputIndex]
	if fcState == nil {
		slog.Warn("received function call arguments delta without open tool call block")
		respCloseAllOpenBlocks(state, &events)
		state.MessageCompleted = true
		events = append(events, respBuildErrorEvent("Received function call arguments delta without an open tool call block."))
		return events
	}

	// Whitespace validation
	nextCount, exceeded := updateWhitespaceRunState(fcState.ConsecutiveWhitespaceCount, event.Delta)
	if exceeded {
		slog.Warn("function call arguments exceeded whitespace limit")
		respCloseAllOpenBlocks(state, &events)
		state.MessageCompleted = true
		events = append(events, respBuildErrorEvent("Received function call arguments delta containing more than 20 consecutive whitespace characters."))
		return events
	}
	fcState.ConsecutiveWhitespaceCount = nextCount

	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: intPtr(blockIndex),
		Delta: &AnthropicContentDelta{
			Type:        "input_json_delta",
			PartialJSON: event.Delta,
		},
	})
	state.BlockHasDelta[blockIndex] = true

	return events
}

func respHandleFunctionCallArgsDone(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	opener := func(s *ResponsesStreamState, e *[]AnthropicStreamEvent) int {
		return respOpenFunctionCallBlock(s, event.OutputIndex, "", "", e)
	}
	events := respEmitDone(state, event.Arguments, opener, &AnthropicContentDelta{
		Type:        "input_json_delta",
		PartialJSON: event.Arguments,
	})
	delete(state.FunctionCallStateByOutputIndex, event.OutputIndex)
	return events
}

func respHandleCompleted(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent
	respCloseAllOpenBlocks(state, &events)

	if event.Response != nil {
		anthropicResp := ConvertResponsesToAnthropic(*event.Response)
		events = append(events,
			AnthropicStreamEvent{
				Type: "message_delta",
				Delta: &AnthropicMessageDelta{
					StopReason: anthropicResp.StopReason,
				},
				Usage: &AnthropicMessageDeltaUsage{OutputTokens: anthropicResp.Usage.OutputTokens},
			},
			AnthropicStreamEvent{Type: "message_stop"},
		)
	} else {
		events = append(events,
			AnthropicStreamEvent{
				Type: "message_delta",
				Delta: &AnthropicMessageDelta{
					StopReason: "end_turn",
				},
				Usage: &AnthropicMessageDeltaUsage{OutputTokens: 0},
			},
			AnthropicStreamEvent{Type: "message_stop"},
		)
	}

	state.MessageCompleted = true
	return events
}

func respHandleFailed(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent
	respCloseAllOpenBlocks(state, &events)

	message := "The response failed due to an unknown error."
	if event.Response != nil && event.Response.Error != nil && event.Response.Error.Message != "" {
		message = event.Response.Error.Message
	}

	events = append(events, respBuildErrorEvent(message))
	state.MessageCompleted = true
	return events
}

func respHandleError(event ResponseStreamEvent, state *ResponsesStreamState) []AnthropicStreamEvent {
	message := "An unexpected error occurred during streaming."
	if event.Message != "" {
		message = event.Message
	}

	var events []AnthropicStreamEvent
	respCloseOpenBlocks(state, &events)
	events = append(events,
		AnthropicStreamEvent{
			Type: "message_delta",
			Delta: &AnthropicMessageDelta{
				StopReason: "end_turn",
			},
			Usage: &AnthropicMessageDeltaUsage{OutputTokens: 0},
		},
		AnthropicStreamEvent{Type: "message_stop"},
		respBuildErrorEvent(message),
	)
	state.MessageCompleted = true
	return events
}

// --- Block management ---

func respOpenTextBlock(state *ResponsesStreamState, outputIndex, contentIndex int, events *[]AnthropicStreamEvent) int {
	key := respBlockKey(outputIndex, contentIndex)
	blockIndex, exists := state.BlockIndexByKey[key]
	if !exists {
		blockIndex = state.NextContentBlockIndex
		state.NextContentBlockIndex++
		state.BlockIndexByKey[key] = blockIndex
	}

	if !state.OpenBlocks[blockIndex] {
		respCloseNonFunctionCallBlocks(state, events)
		*events = append(*events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: intPtr(blockIndex),
			ContentBlock: &AnthropicContentBlock{
				Type: "text",
				Text: "",
			},
		})
		state.OpenBlocks[blockIndex] = true
	}

	return blockIndex
}

func respOpenThinkingBlock(state *ResponsesStreamState, outputIndex int, events *[]AnthropicStreamEvent) int {
	// Thinking blocks combine multiple summary_index into one block
	key := respBlockKey(outputIndex, 0)
	blockIndex, exists := state.BlockIndexByKey[key]
	if !exists {
		blockIndex = state.NextContentBlockIndex
		state.NextContentBlockIndex++
		state.BlockIndexByKey[key] = blockIndex
	}

	if !state.OpenBlocks[blockIndex] {
		respCloseNonFunctionCallBlocks(state, events)
		*events = append(*events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: intPtr(blockIndex),
			ContentBlock: &AnthropicContentBlock{
				Type:     "thinking",
				Thinking: "",
			},
		})
		state.OpenBlocks[blockIndex] = true
	}

	return blockIndex
}

func respOpenFunctionCallBlock(state *ResponsesStreamState, outputIndex int, toolCallID, name string, events *[]AnthropicStreamEvent) int {
	fcState := state.FunctionCallStateByOutputIndex[outputIndex]

	if fcState == nil {
		blockIndex := state.NextContentBlockIndex
		state.NextContentBlockIndex++

		if toolCallID == "" {
			toolCallID = fmt.Sprintf("tool_call_%d", blockIndex)
		}
		if name == "" {
			name = "function"
		}

		fcState = &FunctionCallStreamState{
			BlockIndex:                 blockIndex,
			ToolCallID:                 toolCallID,
			Name:                       name,
			ConsecutiveWhitespaceCount: 0,
		}
		state.FunctionCallStateByOutputIndex[outputIndex] = fcState
	}

	blockIndex := fcState.BlockIndex

	if !state.OpenBlocks[blockIndex] {
		respCloseNonFunctionCallBlocks(state, events)
		*events = append(*events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: intPtr(blockIndex),
			ContentBlock: &AnthropicContentBlock{
				Type:  "tool_use",
				ID:    fcState.ToolCallID,
				Name:  fcState.Name,
				Input: map[string]interface{}{},
			},
		})
		state.OpenBlocks[blockIndex] = true
	}

	return blockIndex
}

func respCloseBlockIfOpen(state *ResponsesStreamState, blockIndex int, events *[]AnthropicStreamEvent) {
	if !state.OpenBlocks[blockIndex] {
		return
	}
	*events = append(*events, AnthropicStreamEvent{
		Type:  "content_block_stop",
		Index: intPtr(blockIndex),
	})
	delete(state.OpenBlocks, blockIndex)
	delete(state.BlockHasDelta, blockIndex)
}

func respCloseNonFunctionCallBlocks(state *ResponsesStreamState, events *[]AnthropicStreamEvent) {
	fcBlocks := make(map[int]bool, len(state.FunctionCallStateByOutputIndex))
	for _, fc := range state.FunctionCallStateByOutputIndex {
		fcBlocks[fc.BlockIndex] = true
	}
	indices := make([]int, 0, len(state.OpenBlocks))
	for blockIndex := range state.OpenBlocks {
		if !fcBlocks[blockIndex] {
			indices = append(indices, blockIndex)
		}
	}
	sort.Ints(indices)
	for _, blockIndex := range indices {
		respCloseBlockIfOpen(state, blockIndex, events)
	}
}

func respCloseOpenBlocks(state *ResponsesStreamState, events *[]AnthropicStreamEvent) {
	indices := make([]int, 0, len(state.OpenBlocks))
	for blockIndex := range state.OpenBlocks {
		indices = append(indices, blockIndex)
	}
	sort.Ints(indices)
	for _, blockIndex := range indices {
		respCloseBlockIfOpen(state, blockIndex, events)
	}
}

func respCloseAllOpenBlocks(state *ResponsesStreamState, events *[]AnthropicStreamEvent) {
	respCloseOpenBlocks(state, events)
	// Clear function call state
	for k := range state.FunctionCallStateByOutputIndex {
		delete(state.FunctionCallStateByOutputIndex, k)
	}
}

// --- Generic delta/done helpers ---

// blockOpener opens (or reuses) a content block and returns its index.
type blockOpener func(state *ResponsesStreamState, events *[]AnthropicStreamEvent) int

// respEmitDelta is the generic delta handler: guard empty delta, open block, emit content_block_delta.
func respEmitDelta(state *ResponsesStreamState, deltaText string, openBlock blockOpener, delta interface{}) []AnthropicStreamEvent {
	if deltaText == "" {
		return nil
	}
	var events []AnthropicStreamEvent
	blockIndex := openBlock(state, &events)
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: intPtr(blockIndex),
		Delta: delta,
	})
	state.BlockHasDelta[blockIndex] = true
	return events
}

// respEmitDone is the generic done handler: open block, emit fallback delta if no delta was seen.
func respEmitDone(state *ResponsesStreamState, fallbackText string, openBlock blockOpener, delta interface{}) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent
	blockIndex := openBlock(state, &events)
	if fallbackText != "" && !state.BlockHasDelta[blockIndex] {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_delta",
			Index: intPtr(blockIndex),
			Delta: delta,
		})
	}
	return events
}

// --- Helpers ---

// blockKey is a struct key for the block index map, avoiding fmt.Sprintf per lookup.
type blockKey struct{ outputIndex, contentIndex int }

func respBlockKey(outputIndex, contentIndex int) blockKey {
	return blockKey{outputIndex, contentIndex}
}

func respBuildErrorEvent(message string) AnthropicStreamEvent {
	return AnthropicStreamEvent{
		Type: "error",
		Error: &AnthropicError{
			Type:    AnthropicErrorTypeAPI,
			Message: message,
		},
	}
}

// updateWhitespaceRunState tracks consecutive newline/tab characters
// in function call argument chunks. Returns the updated count and whether
// the limit was exceeded.
func updateWhitespaceRunState(previousCount int, chunk string) (nextCount int, exceeded bool) {
	count := previousCount
	for _, ch := range chunk {
		if ch == '\r' || ch == '\n' || ch == '\t' {
			count++
			if count > maxConsecutiveFunctionCallWhitespace {
				return count, true
			}
			continue
		}
		if ch != ' ' {
			count = 0
		}
	}
	return count, false
}
