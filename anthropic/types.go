package anthropic

import (
	"encoding/json"
	"fmt"
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
	Delta        *AnthropicContentBlockDelta    `json:"delta,omitempty"`
	Usage        interface{}                     `json:"usage,omitempty"`
	Error        *AnthropicError                `json:"error,omitempty"`
}

type AnthropicContentBlockDelta struct {
	StopReason   string `json:"stop_reason,omitempty"`
	StopSequence string `json:"stop_sequence,omitempty"`
	Type        string `json:"type,omitempty"` // "text_delta", "input_json_delta", "thinking_delta", "signature_delta"
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// OpenAI Chat Completions types (used for conversion)
type OpenAIChatCompletionsRequest struct {
	Model             string                     `json:"model"`
	Messages          []OpenAIMessage            `json:"messages"`
	Tools             []OpenAIToolDefinition     `json:"tools,omitempty"`
	ToolChoice        interface{}                `json:"tool_choice,omitempty"` // string or object
	ParallelToolCalls *bool                      `json:"parallel_tool_calls,omitempty"`
	Temperature       *float64                   `json:"temperature,omitempty"`
	TopP              *float64                   `json:"top_p,omitempty"`
	MaxTokens         *int                       `json:"max_tokens,omitempty"`
	Stop              interface{}                `json:"stop,omitempty"` // string or []string
	Stream            bool                       `json:"stream,omitempty"`
	User              string                     `json:"user,omitempty"`
	Metadata          map[string]string          `json:"metadata,omitempty"`
	ThinkingBudget    *int                       `json:"thinking_budget,omitempty"`
	ReasoningText     *string                    `json:"reasoning_text,omitempty"`
	ReasoningOpaque   *string                    `json:"reasoning_opaque,omitempty"`
}

type OpenAIMessage struct {
	Role           string              `json:"role"` // "system", "user", "assistant", "tool"
	Content        *OpenAIContent      `json:"content,omitempty"`
	Name           string              `json:"name,omitempty"`
	ToolCalls      []OpenAIToolCall    `json:"tool_calls,omitempty"`
	ToolCallID     string              `json:"tool_call_id,omitempty"`
	ReasoningText  *string             `json:"reasoning_text,omitempty"`
	ReasoningOpaque *string            `json:"reasoning_opaque,omitempty"`
}

// OpenAIContent can be string or []OpenAIContentPart
type OpenAIContent struct {
	Text  *string
	Parts []OpenAIContentPart
}

// Custom JSON marshaling for OpenAIContent
func (c *OpenAIContent) MarshalJSON() ([]byte, error) {
	if c.Text != nil {
		return json.Marshal(*c.Text)
	}
	return json.Marshal(c.Parts)
}

func (c *OpenAIContent) UnmarshalJSON(data []byte) error {
	// Clear both fields first
	c.Text = nil
	c.Parts = nil

	// Try to unmarshal as string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		c.Text = &str
		return nil
	}

	// Try to unmarshal as array of parts
	var parts []OpenAIContentPart
	if err := json.Unmarshal(data, &parts); err == nil {
		c.Parts = parts
		return nil
	}

	return fmt.Errorf("content must be string or array of parts")
}

type OpenAIContentPart struct {
	Type     string                    `json:"type"` // "text", "image_url"
	Text     string                    `json:"text,omitempty"`
	ImageURL *OpenAIImageURLPayload    `json:"image_url,omitempty"`
}

type OpenAIImageURLPayload struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type OpenAIToolDefinition struct {
	Type     string              `json:"type"` // "function"
	Function OpenAIFunctionSpec  `json:"function"`
}

type OpenAIFunctionSpec struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"`
}

type OpenAIToolCall struct {
	ID       string                  `json:"id"`
	Type     string                  `json:"type"` // "function"
	Function OpenAIToolCallFunction `json:"function"`
	Index    *int                    `json:"index,omitempty"` // for streaming
}

type OpenAIToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // JSON string
}

// OpenAI Response types
type OpenAIChatCompletionsResponse struct {
	ID                string           `json:"id"`
	Object            string           `json:"object"` // "chat.completion"
	Created           int64            `json:"created"`
	Model             string           `json:"model"`
	Choices           []OpenAIChoice   `json:"choices"`
	Usage             *OpenAIUsage     `json:"usage,omitempty"`
	SystemFingerprint string           `json:"system_fingerprint,omitempty"`
}

type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason,omitempty"` // "stop", "length", "tool_calls", "content_filter"
}

type OpenAIUsage struct {
	PromptTokens            int                              `json:"prompt_tokens"`
	CompletionTokens        int                              `json:"completion_tokens"`
	TotalTokens             int                              `json:"total_tokens"`
	PromptTokensDetails     *OpenAIPromptTokensDetails       `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *OpenAICompletionTokensDetails   `json:"completion_tokens_details,omitempty"`
}

type OpenAIPromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type OpenAICompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// Streaming types
type OpenAIChatCompletionChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"` // "chat.completion.chunk"
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []OpenAIChunkChoice `json:"choices"`
	Usage   *OpenAIUsage        `json:"usage,omitempty"`
}

type OpenAIChunkChoice struct {
	Index        int           `json:"index"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason string        `json:"finish_reason,omitempty"`
}

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
