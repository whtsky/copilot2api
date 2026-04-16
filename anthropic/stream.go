package anthropic


// intPtr returns a pointer to a copy of the given int value.
// This avoids aliasing bugs when the source variable is later mutated.
func intPtr(v int) *int {
	return &v
}

// StreamState tracks the state of Anthropic streaming conversion
type StreamState struct {
	MessageStartSent  bool
	ContentBlockIndex int
	ContentBlockOpen  bool
	ThinkingBlockOpen bool
	ToolCalls         map[int]*ToolCallState // OpenAI tool index -> Anthropic state
	Finished          bool                   // true after message_delta + message_stop emitted
}

// ToolCallState tracks individual tool call state
type ToolCallState struct {
	ID                  string
	Name                string
	AnthropicBlockIndex int
	Stopped             bool
}

// NewStreamState creates a new streaming state
func NewStreamState() *StreamState {
	return &StreamState{
		ToolCalls: make(map[int]*ToolCallState),
	}
}

// ConvertOpenAIChunkToAnthropicEvents converts OpenAI SSE chunk to Anthropic events
func ConvertOpenAIChunkToAnthropicEvents(chunk OpenAIChatCompletionChunk, state *StreamState) ([]AnthropicStreamEvent, error) {
	var events []AnthropicStreamEvent

	if len(chunk.Choices) == 0 {
		return events, nil
	}

	choice := chunk.Choices[0]
	delta := choice.Delta

	// Handle message start
	if !state.MessageStartSent {
		events = append(events, createMessageStartEvent(chunk))
		state.MessageStartSent = true
	}

	// Handle thinking text
	if delta.ReasoningText != nil && *delta.ReasoningText != "" {
		thinkingEvents := handleThinkingText(delta, state)
		events = append(events, thinkingEvents...)
	}

	// Handle content
	if delta.Content != nil {
		contentEvents := handleContent(delta, state)
		events = append(events, contentEvents...)
	}

	// Handle tool calls
	if len(delta.ToolCalls) > 0 {
		toolEvents := handleToolCalls(delta, state)
		events = append(events, toolEvents...)
	}

	// Handle finish
	if choice.FinishReason != "" {
		finishEvents := handleFinish(choice, chunk, state)
		events = append(events, finishEvents...)
	}

	return events, nil
}

func createMessageStartEvent(chunk OpenAIChatCompletionChunk) AnthropicStreamEvent {
	usage := AnthropicUsage{
		InputTokens:  0,
		OutputTokens: 0,
	}

	if chunk.Usage != nil {
		usage.InputTokens = chunk.Usage.PromptTokens
		if chunk.Usage.PromptTokensDetails != nil {
			usage.CacheReadInputTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			usage.InputTokens -= usage.CacheReadInputTokens
			if usage.InputTokens < 0 {
				usage.InputTokens = 0
			}
		}
	}

	return AnthropicStreamEvent{
		Type: "message_start",
		Message: &AnthropicMessagesResponse{
			ID:      chunk.ID,
			Type:    "message",
			Role:    "assistant",
			Model:   chunk.Model,
			Content: []AnthropicContentBlock{},
			Usage:   usage,
		},
	}
}

func handleThinkingText(delta OpenAIMessage, state *StreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	reasoningText := ""
	if delta.ReasoningText != nil {
		reasoningText = *delta.ReasoningText
	}

	// Handle edge case where content and reasoning_text come in same delta
	if delta.Content != nil && state.ContentBlockOpen {
		// Merge thinking into content - this is a server bug workaround
		return events
	}

	if !state.ThinkingBlockOpen {
		// Start thinking block
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: intPtr(state.ContentBlockIndex),
			ContentBlock: &AnthropicContentBlock{
				Type:     "thinking",
				Thinking: "",
			},
		})
		state.ThinkingBlockOpen = true
	}

	// Send thinking delta
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: intPtr(state.ContentBlockIndex),
		Delta: &AnthropicContentDelta{
			Type:     "thinking_delta",
			Thinking: reasoningText,
		},
	})

	return events
}

func handleContent(delta OpenAIMessage, state *StreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	content := ""
	if delta.Content != nil && delta.Content.Text != nil {
		content = *delta.Content.Text
	}

	signature := reasoningOpaque(delta)

	// Close thinking block if open, attaching signature to the same block.
	if state.ThinkingBlockOpen {
		events = append(events, closeThinkingBlock(state, signature)...)
	}

	// Close tool block if open
	if state.ContentBlockOpen && isToolBlockOpen(state) {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_stop",
			Index: intPtr(state.ContentBlockIndex),
		})
		state.ContentBlockIndex++
		state.ContentBlockOpen = false
	}

	// Open text block if not open
	if !state.ContentBlockOpen {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_start",
			Index: intPtr(state.ContentBlockIndex),
			ContentBlock: &AnthropicContentBlock{
				Type: "text",
				Text: "",
			},
		})
		state.ContentBlockOpen = true
	}

	// Send text delta
	events = append(events, AnthropicStreamEvent{
		Type:  "content_block_delta",
		Index: intPtr(state.ContentBlockIndex),
		Delta: &AnthropicContentDelta{
			Type: "text_delta",
			Text: content,
		},
	})

	return events
}

func handleToolCalls(delta OpenAIMessage, state *StreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	signature := reasoningOpaque(delta)
	consumedThinkingSignature := false

	// Close thinking block if open, attaching signature to the same block.
	if state.ThinkingBlockOpen {
		events = append(events, closeThinkingBlock(state, signature)...)
		consumedThinkingSignature = signature != ""
	}

	// Handle reasoning opaque in tool calls
	if state.ContentBlockOpen && !isToolBlockOpen(state) {
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_stop",
			Index: intPtr(state.ContentBlockIndex),
		})
		state.ContentBlockIndex++
		state.ContentBlockOpen = false
	}

	// Handle reasoning opaque only when there was no open thinking block to attach it to.
	if !consumedThinkingSignature && !state.ThinkingBlockOpen {
		if signature != "" {
			events = append(events, handleReasoningOpaque(signature, state)...)
		}
	}

	for _, toolCall := range delta.ToolCalls {
		if toolCall.Index == nil {
			continue
		}

		index := *toolCall.Index

		// New tool call starting
		if toolCall.ID != "" && toolCall.Function.Name != "" {
			// Close any previously open block and mark its tool as stopped
			if state.ContentBlockOpen {
				// Mark the tool at the current block index as stopped
				for _, tc := range state.ToolCalls {
					if tc.AnthropicBlockIndex == state.ContentBlockIndex {
						tc.Stopped = true
					}
				}
				events = append(events, AnthropicStreamEvent{
					Type:  "content_block_stop",
					Index: intPtr(state.ContentBlockIndex),
				})
				state.ContentBlockIndex++
				state.ContentBlockOpen = false
			}

			// Track this tool call
			anthropicBlockIndex := state.ContentBlockIndex
			state.ToolCalls[index] = &ToolCallState{
				ID:                  toolCall.ID,
				Name:                toolCall.Function.Name,
				AnthropicBlockIndex: anthropicBlockIndex,
			}

			// Start tool use block
			events = append(events, AnthropicStreamEvent{
				Type:  "content_block_start",
				Index: intPtr(anthropicBlockIndex),
				ContentBlock: &AnthropicContentBlock{
					Type:  "tool_use",
					ID:    toolCall.ID,
					Name:  toolCall.Function.Name,
					Input: map[string]interface{}{},
				},
			})
			state.ContentBlockOpen = true
		}

		// Tool arguments delta - skip if tool block already stopped
		if toolCall.Function.Arguments != "" {
			if toolCallState, exists := state.ToolCalls[index]; exists && !toolCallState.Stopped {
				events = append(events, AnthropicStreamEvent{
					Type:  "content_block_delta",
					Index: &toolCallState.AnthropicBlockIndex,
					Delta: &AnthropicContentDelta{
						Type:        "input_json_delta",
						PartialJSON: toolCall.Function.Arguments,
					},
				})
			}
		}
	}

	return events
}

func handleFinish(choice OpenAIChunkChoice, chunk OpenAIChatCompletionChunk, state *StreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	signature := reasoningOpaque(choice.Delta)
	consumedThinkingSignature := false

	// Close thinking block if open, attaching signature to the same block.
	if state.ThinkingBlockOpen {
		events = append(events, closeThinkingBlock(state, signature)...)
		consumedThinkingSignature = signature != ""
	}

	// Close any open content block
	if state.ContentBlockOpen {
		toolBlockOpen := isToolBlockOpen(state)

		// Capture the index before incrementing
		events = append(events, AnthropicStreamEvent{
			Type:  "content_block_stop",
			Index: intPtr(state.ContentBlockIndex),
		})

		state.ContentBlockOpen = false
		state.ContentBlockIndex++

		// Handle reasoning opaque for non-tool blocks when there was no open thinking block.
		if !toolBlockOpen && !consumedThinkingSignature {
			if signature != "" {
				events = append(events, handleReasoningOpaque(signature, state)...)
			}
		}
	}

	// Calculate final usage
	usage := AnthropicUsage{}
	if chunk.Usage != nil {
		usage.InputTokens = chunk.Usage.PromptTokens
		usage.OutputTokens = chunk.Usage.CompletionTokens

		if chunk.Usage.PromptTokensDetails != nil {
			usage.CacheReadInputTokens = chunk.Usage.PromptTokensDetails.CachedTokens
			usage.InputTokens -= usage.CacheReadInputTokens
			if usage.InputTokens < 0 {
				usage.InputTokens = 0
			}
		}
	}

	// Send message delta
	events = append(events, AnthropicStreamEvent{
		Type: "message_delta",
		Delta: &AnthropicMessageDelta{
			StopReason: mapOpenAIFinishReasonToAnthropic(choice.FinishReason),
		},
		Usage: &AnthropicMessageDeltaUsage{OutputTokens: usage.OutputTokens},
	})

	// Send message stop
	events = append(events, AnthropicStreamEvent{
		Type: "message_stop",
	})

	state.Finished = true

	return events
}

func handleReasoningOpaque(signature string, state *StreamState) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	if signature != "" {
		events = append(events,
			AnthropicStreamEvent{
				Type:  "content_block_start",
				Index: intPtr(state.ContentBlockIndex),
				ContentBlock: &AnthropicContentBlock{
					Type:     "thinking",
					Thinking: "",
				},
			},
			AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: intPtr(state.ContentBlockIndex),
				Delta: &AnthropicContentDelta{
					Type:     "thinking_delta",
					Thinking: "",
				},
			},
			AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: intPtr(state.ContentBlockIndex),
				Delta: &AnthropicContentDelta{
					Type:      "signature_delta",
					Signature: signature,
				},
			},
			AnthropicStreamEvent{
				Type:  "content_block_stop",
				Index: intPtr(state.ContentBlockIndex),
			},
		)
		state.ContentBlockIndex++
	}

	return events
}

func closeThinkingBlock(state *StreamState, signature string) []AnthropicStreamEvent {
	var events []AnthropicStreamEvent

	if state.ThinkingBlockOpen {
		if signature != "" {
			events = append(events, AnthropicStreamEvent{
				Type:  "content_block_delta",
				Index: intPtr(state.ContentBlockIndex),
				Delta: &AnthropicContentDelta{
					Type:      "signature_delta",
					Signature: signature,
				},
			})
		}
		events = append(events,
			AnthropicStreamEvent{
				Type:  "content_block_stop",
				Index: intPtr(state.ContentBlockIndex),
			},
		)
		state.ContentBlockIndex++
		state.ThinkingBlockOpen = false
	}

	return events
}

func reasoningOpaque(delta OpenAIMessage) string {
	if delta.ReasoningOpaque != nil {
		return *delta.ReasoningOpaque
	}
	return ""
}

func isToolBlockOpen(state *StreamState) bool {
	if !state.ContentBlockOpen {
		return false
	}

	// Check if current block index corresponds to any tool call
	for _, toolCall := range state.ToolCalls {
		if toolCall.AnthropicBlockIndex == state.ContentBlockIndex {
			return true
		}
	}

	return false
}

// CreateErrorEvent creates an error event for stream failures
func CreateErrorEvent(message string) AnthropicStreamEvent {
	return AnthropicStreamEvent{
		Type: "error",
		Error: &AnthropicError{
			Type:    AnthropicErrorTypeAPI,
			Message: message,
		},
	}
}
