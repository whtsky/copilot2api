package types

import (
	"encoding/json"
	"fmt"
)

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
	StreamOptions     *OpenAIStreamOptions       `json:"stream_options,omitempty"`
	User              string                     `json:"user,omitempty"`
	Metadata          map[string]string          `json:"metadata,omitempty"`
	ThinkingBudget    *int                       `json:"thinking_budget,omitempty"`
	ReasoningText     *string                    `json:"reasoning_text,omitempty"`
	ReasoningOpaque   *string                    `json:"reasoning_opaque,omitempty"`
}

// OpenAIStreamOptions controls streaming behavior options.
type OpenAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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
