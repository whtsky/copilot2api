package types

// codexPhaseModel is the model ID that requires phase annotations on assistant messages.
const CodexPhaseModel = "gpt-5.3-codex"

// maxConsecutiveFunctionCallWhitespace is the maximum number of consecutive
// newline/tab characters allowed in function call arguments before we abort.
const MaxConsecutiveFunctionCallWhitespace = 20

// --- Responses API Request Types ---

// ResponsesRequest is the request payload for the Responses API.
type ResponsesRequest struct {
	Model             string              `json:"model"`
	Input             []ResponseInputItem `json:"input"`
	Instructions      *string             `json:"instructions"`
	Temperature       *float64            `json:"temperature,omitempty"`
	TopP              *float64            `json:"top_p,omitempty"`
	MaxOutputTokens   *int                `json:"max_output_tokens,omitempty"`
	Tools             []ResponseTool      `json:"tools,omitempty"`
	ToolChoice        interface{}         `json:"tool_choice,omitempty"`
	Metadata          map[string]string   `json:"metadata,omitempty"`
	Stream            bool                `json:"stream"`
	Store             bool                `json:"store"`
	ParallelToolCalls bool                `json:"parallel_tool_calls"`
	Reasoning         *ResponseReasoning  `json:"reasoning,omitempty"`
	Include           []string            `json:"include,omitempty"`
}

// ResponseReasoning configures reasoning behavior.
type ResponseReasoning struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary"`
}

// ResponseInputItem is a union struct representing all input item types:
// message, function_call, function_call_output, reasoning.
type ResponseInputItem struct {
	Type string `json:"type"`

	// message fields
	Role    string      `json:"role,omitempty"`
	Content interface{} `json:"content,omitempty"` // string or []ResponseInputContent
	Phase   string      `json:"phase,omitempty"`

	// function_call / function_call_output fields
	CallID    string      `json:"call_id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Arguments string      `json:"arguments,omitempty"`
	Status    string      `json:"status,omitempty"`
	Output    interface{} `json:"output,omitempty"` // string or []ResponseInputContent

	// reasoning fields
	ID               string                  `json:"id,omitempty"`
	Summary          *[]ResponseSummaryBlock `json:"summary,omitempty"`
	EncryptedContent string                  `json:"encrypted_content,omitempty"`
}

// ResponseInputContent represents content items within a message input.
type ResponseInputContent struct {
	Type     string `json:"type"` // "input_text", "output_text", "input_image"
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

// ResponseSummaryBlock represents a summary text block in reasoning items.
type ResponseSummaryBlock struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// ResponseTool represents a tool (function) definition.
type ResponseTool struct {
	Type        string      `json:"type"` // "function"
	Name        string      `json:"name"`
	Parameters  interface{} `json:"parameters,omitempty"`
	Strict      bool        `json:"strict"`
	Description string      `json:"description,omitempty"`
}

// --- Responses API Response Types ---

// ResponsesResult is the result object from the Responses API.
type ResponsesResult struct {
	ID                string               `json:"id"`
	Model             string               `json:"model"`
	Status            string               `json:"status"` // "completed", "incomplete", "failed"
	Output            []ResponseOutputItem `json:"output"`
	OutputText        string               `json:"output_text"`
	Usage             *ResponsesUsage      `json:"usage,omitempty"`
	IncompleteDetails *IncompleteDetails   `json:"incomplete_details,omitempty"`
	Error             *ResponseError       `json:"error,omitempty"`
}

// ResponsesUsage represents token usage in Responses API.
type ResponsesUsage struct {
	InputTokens        int                `json:"input_tokens"`
	OutputTokens       int                `json:"output_tokens"`
	InputTokensDetails *InputTokenDetails `json:"input_tokens_details,omitempty"`
}

// InputTokenDetails contains detailed input token info.
type InputTokenDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// IncompleteDetails explains why a response is incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// ResponseError represents an error within a response.
type ResponseError struct {
	Message string `json:"message"`
}

// ResponseOutputItem is a union struct for output items:
// reasoning, function_call, message.
type ResponseOutputItem struct {
	Type string `json:"type"`

	// reasoning fields
	ID               string               `json:"id,omitempty"`
	Summary          []ResponseSummaryBlock `json:"summary,omitempty"`
	EncryptedContent string               `json:"encrypted_content,omitempty"`

	// function_call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// message fields
	Content []ResponseOutputContent `json:"content,omitempty"`
}

// ResponseOutputContent represents content within an output message.
type ResponseOutputContent struct {
	Type    string `json:"type"` // "output_text", "refusal"
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`
}

// --- Responses API Stream Event Types ---

// ResponseStreamEvent is a parsed SSE event from the Responses API stream.
// Fields are populated based on the event Type.
type ResponseStreamEvent struct {
	Type         string              `json:"type"`
	Response     *ResponsesResult    `json:"response,omitempty"`
	Item         *ResponseOutputItem `json:"item,omitempty"`
	OutputIndex  int                 `json:"output_index"`
	ContentIndex int                 `json:"content_index"`
	SummaryIndex int                 `json:"summary_index"`
	Delta        string              `json:"delta,omitempty"`
	Text         string              `json:"text,omitempty"`
	Arguments    string              `json:"arguments,omitempty"`
	Message      string              `json:"message,omitempty"` // for error events
}

// --- Responses Stream Translation State ---

// ResponsesStreamState tracks the state during Responses->Anthropic stream translation.
type ResponsesStreamState struct {
	MessageStartSent               bool
	MessageCompleted               bool
	NextContentBlockIndex          int
	BlockIndexByKey                map[BlockKey]int
	OpenBlocks                     map[int]bool
	BlockHasDelta                  map[int]bool
	FunctionCallStateByOutputIndex map[int]*FunctionCallStreamState
}

// BlockKey is a struct key for the block index map.
type BlockKey struct {
	OutputIndex  int
	ContentIndex int
}

// FunctionCallStreamState tracks individual function call state during streaming.
type FunctionCallStreamState struct {
	BlockIndex                 int
	ToolCallID                 string
	Name                       string
	ConsecutiveWhitespaceCount int
}

// NewResponsesStreamState creates a new stream state for Responses->Anthropic translation.
func NewResponsesStreamState() *ResponsesStreamState {
	return &ResponsesStreamState{
		BlockIndexByKey:                make(map[BlockKey]int),
		OpenBlocks:                     make(map[int]bool),
		BlockHasDelta:                  make(map[int]bool),
		FunctionCallStateByOutputIndex: make(map[int]*FunctionCallStreamState),
	}
}
