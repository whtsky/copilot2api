package anthropic

import (
	"github.com/whtsky/copilot2api/internal/types"
)

// codexPhaseModel is the model ID that requires phase annotations on assistant messages.
const codexPhaseModel = types.CodexPhaseModel

// maxConsecutiveFunctionCallWhitespace is the maximum number of consecutive
// newline/tab characters allowed in function call arguments before we abort.
const maxConsecutiveFunctionCallWhitespace = types.MaxConsecutiveFunctionCallWhitespace

// --- Type aliases for Responses types moved to internal/types ---

type ResponsesRequest = types.ResponsesRequest
type ResponseReasoning = types.ResponseReasoning
type ResponseInputItem = types.ResponseInputItem
type ResponseInputContent = types.ResponseInputContent
type ResponseSummaryBlock = types.ResponseSummaryBlock
type ResponseTool = types.ResponseTool
type ResponsesResult = types.ResponsesResult
type ResponsesUsage = types.ResponsesUsage
type InputTokenDetails = types.InputTokenDetails
type IncompleteDetails = types.IncompleteDetails
type ResponseError = types.ResponseError
type ResponseOutputItem = types.ResponseOutputItem
type ResponseOutputContent = types.ResponseOutputContent
type ResponseStreamEvent = types.ResponseStreamEvent
type ResponsesStreamState = types.ResponsesStreamState
type FunctionCallStreamState = types.FunctionCallStreamState

// blockKey is a struct key for the block index map.
type blockKey = types.BlockKey

// NewResponsesStreamState creates a new stream state for Responses->Anthropic translation.
var NewResponsesStreamState = types.NewResponsesStreamState
