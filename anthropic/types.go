package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/whtsky/copilot2api/internal/types"
)

// AnthropicMessagesRequest represents an Anthropic Messages API request
type AnthropicMessagesRequest struct {
	Model         string                 `json:"model"`
	MaxTokens     int                    `json:"max_tokens"`
	System        *AnthropicSystem       `json:"system,omitempty"`
	Messages      []AnthropicMessage     `json:"messages"`
	Tools         []AnthropicTool        `json:"tools,omitempty"`
	ToolChoice    *AnthropicToolChoice   `json:"tool_choice,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	TopK          *int                   `json:"top_k,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	Metadata      map[string]string      `json:"metadata,omitempty"`
	Thinking      *AnthropicThinking     `json:"thinking,omitempty"`
	ServiceTier   string                 `json:"service_tier,omitempty"`
	OutputConfig  *AnthropicOutputConfig `json:"output_config,omitempty"`
}

// AnthropicThinking represents thinking configuration
type AnthropicThinking struct {
	Type         string `json:"type"` // "enabled" or "adaptive"
	BudgetTokens *int   `json:"budget_tokens,omitempty"`
}

// AnthropicOutputConfig represents output configuration
type AnthropicOutputConfig struct {
	Effort string `json:"effort,omitempty"` // "low", "medium", "high", "max"
}

// AnthropicSystem can be string or []AnthropicTextBlock
type AnthropicSystem struct {
	Text   *string
	Blocks []AnthropicSystemBlock
}

type AnthropicSystemBlock struct {
	Type string `json:"type"` // "text"
	Text string `json:"text"`
}

// Custom JSON marshaling for AnthropicSystem
func (s *AnthropicSystem) MarshalJSON() ([]byte, error) {
	if s.Text != nil {
		return json.Marshal(*s.Text)
	}
	return json.Marshal(s.Blocks)
}

func (s *AnthropicSystem) UnmarshalJSON(data []byte) error {
	// Clear both fields first
	s.Text = nil
	s.Blocks = nil

	// Try to unmarshal as string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		s.Text = &str
		return nil
	}

	// Try to unmarshal as array of blocks
	var blocks []AnthropicSystemBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		s.Blocks = blocks
		return nil
	}

	return fmt.Errorf("system must be string or array of blocks")
}

// AnthropicMessage represents a message in the conversation
type AnthropicMessage struct {
	Role    string            `json:"role"` // "user" or "assistant"
	Content AnthropicContent  `json:"content"`
}

// AnthropicContent can be string or []AnthropicContentBlock
type AnthropicContent struct {
	Text   *string
	Blocks []AnthropicContentBlock
}

// Custom JSON marshaling for AnthropicContent
func (c *AnthropicContent) MarshalJSON() ([]byte, error) {
	if c.Text != nil {
		return json.Marshal(*c.Text)
	}
	return json.Marshal(c.Blocks)
}

func (c *AnthropicContent) UnmarshalJSON(data []byte) error {
	// Clear both fields first
	c.Text = nil
	c.Blocks = nil

	// Try to unmarshal as string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Text = &str
		return nil
	}

	// Try to unmarshal as array of blocks
	var blocks []AnthropicContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Blocks = blocks
		return nil
	}

	return fmt.Errorf("content must be string or array of blocks")
}

// AnthropicContentBlock represents different types of content blocks
type AnthropicContentBlock struct {
	Type string `json:"type"` // "text", "image", "tool_use", "tool_result", "thinking"

	// Text block fields
	Text string `json:"text,omitempty"`

	// Image block fields
	Source *AnthropicImageSource `json:"source,omitempty"`

	// Tool use block fields
	ID    string                 `json:"id,omitempty"`
	Name  string                 `json:"name,omitempty"`
	Input map[string]interface{} `json:"input,omitempty"`

	// Tool result block fields
	ToolUseID string           `json:"tool_use_id,omitempty"`
	Content   *AnthropicContent `json:"content,omitempty"`
	IsError   *bool             `json:"is_error,omitempty"`

	// Thinking block fields
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// AnthropicImageSource represents image data
type AnthropicImageSource struct {
	Type      string `json:"type"`       // "base64"
	MediaType string `json:"media_type"` // "image/jpeg", "image/png", etc.
	Data      string `json:"data"`       // base64 encoded data
}

// AnthropicTool represents a tool definition
type AnthropicTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	InputSchema interface{} `json:"input_schema"`
}

// AnthropicToolChoice represents tool choice configuration
type AnthropicToolChoice struct {
	Type                 string `json:"type"` // "auto", "any", "tool", "none"
	Name                 string `json:"name,omitempty"`
	DisableParallelCalls *bool  `json:"disable_parallel_tool_use,omitempty"`
}

// AnthropicMessagesResponse represents the response from Anthropic Messages API
type AnthropicMessagesResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"` // "message"
	Role         string                  `json:"role"` // "assistant"
	Model        string                  `json:"model"`
	Content      []AnthropicContentBlock `json:"content"`
	StopReason   string                  `json:"stop_reason,omitempty"`
	StopSequence *string                 `json:"stop_sequence"`
	Usage        AnthropicUsage          `json:"usage"`
}

// AnthropicUsage represents token usage information
type AnthropicUsage struct {
	InputTokens              int    `json:"input_tokens"`
	OutputTokens             int    `json:"output_tokens"`
	CacheCreationInputTokens int    `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int    `json:"cache_read_input_tokens,omitempty"`
	ServiceTier              string `json:"service_tier,omitempty"`
}

// Streaming event types
type AnthropicStreamEvent struct {
	Type         string                         `json:"type"`
	Message      *AnthropicMessagesResponse     `json:"message,omitempty"`
	Index        *int                           `json:"index,omitempty"`
	ContentBlock *AnthropicContentBlock         `json:"content_block,omitempty"`
	Delta        interface{}                    `json:"delta,omitempty"`
	Usage        interface{}                    `json:"usage,omitempty"`
	Error        *AnthropicError                `json:"error,omitempty"`
}

// AnthropicMessageDelta is the delta payload for message_delta events.
// Per Anthropic spec: always includes stop_reason and stop_sequence.
type AnthropicMessageDelta struct {
	StopReason   string  `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"` // always present, null when not applicable
}

// AnthropicContentDelta is the delta payload for content_block_delta events.
type AnthropicContentDelta struct {
	Type        string `json:"type"`                   // "text_delta", "input_json_delta", "thinking_delta", "signature_delta"
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// --- Type aliases for OpenAI types moved to internal/types ---
// These aliases allow existing anthropic package code to continue using
// the unqualified type names.

type OpenAIChatCompletionsRequest = types.OpenAIChatCompletionsRequest
type OpenAIStreamOptions = types.OpenAIStreamOptions
type OpenAIMessage = types.OpenAIMessage
type OpenAIContent = types.OpenAIContent
type OpenAIContentPart = types.OpenAIContentPart
type OpenAIImageURLPayload = types.OpenAIImageURLPayload
type OpenAIToolDefinition = types.OpenAIToolDefinition
type OpenAIFunctionSpec = types.OpenAIFunctionSpec
type OpenAIToolCall = types.OpenAIToolCall
type OpenAIToolCallFunction = types.OpenAIToolCallFunction
type OpenAIChatCompletionsResponse = types.OpenAIChatCompletionsResponse
type OpenAIChoice = types.OpenAIChoice
type OpenAIUsage = types.OpenAIUsage
type OpenAIPromptTokensDetails = types.OpenAIPromptTokensDetails
type OpenAICompletionTokensDetails = types.OpenAICompletionTokensDetails
type OpenAIChatCompletionChunk = types.OpenAIChatCompletionChunk
type OpenAIChunkChoice = types.OpenAIChunkChoice

// AnthropicMessageDeltaUsage is usage info in message_delta (only output_tokens)
type AnthropicMessageDeltaUsage struct {
	OutputTokens int `json:"output_tokens"`
}

// AnthropicErrorResponse represents an error response from the Anthropic API
type AnthropicErrorResponse struct {
	Type  string         `json:"type"` // "error"
	Error AnthropicError `json:"error"`
}

// AnthropicError represents the error object in Anthropic API responses
type AnthropicError struct {
	Type    string `json:"type"`    // error type (required)
	Message string `json:"message"` // error message (required)
}

// Error type constants for Anthropic API
const (
	AnthropicErrorTypeInvalidRequest   = "invalid_request_error"
	AnthropicErrorTypeAuthentication   = "authentication_error"
	AnthropicErrorTypePermission       = "permission_error"
	AnthropicErrorTypeNotFound         = "not_found_error"
	AnthropicErrorTypeRateLimit        = "rate_limit_error"
	AnthropicErrorTypeAPI              = "api_error"
	AnthropicErrorTypeOverloaded       = "overloaded_error"
)
