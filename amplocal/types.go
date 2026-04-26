package amplocal

import "encoding/json"

// ThreadFile represents a thread stored in ~/.local/share/amp/threads/*.json.
type ThreadFile struct {
	V               *uint64         `json:"v,omitempty"`
	ID              string          `json:"id"`
	Created         *uint64         `json:"created,omitempty"`
	Messages        []ThreadMessage `json:"messages,omitempty"`
	AgentMode       *string         `json:"agentMode,omitempty"`
	NextMessageID   *uint64         `json:"nextMessageId,omitempty"`
	Title           *string         `json:"title,omitempty"`
	Env             json.RawMessage `json:"env,omitempty"`
	Meta            json.RawMessage `json:"meta,omitempty"`
	Debug           json.RawMessage `json:"~debug,omitempty"`
	ActivatedSkills json.RawMessage `json:"activatedSkills,omitempty"`
	Relationships   json.RawMessage `json:"relationships,omitempty"`
	Archived        *bool           `json:"archived,omitempty"`
	OriginThreadID  *string         `json:"originThreadID,omitempty"`
	MainThreadID    *string         `json:"mainThreadID,omitempty"`
}

// ThreadMessage is a single message inside a thread.
type ThreadMessage struct {
	Role      *string         `json:"role,omitempty"`
	MessageID *uint64         `json:"messageId,omitempty"`
	Content   []ContentBlock  `json:"content,omitempty"`
	UserState json.RawMessage `json:"userState,omitempty"`
	AgentMode *string         `json:"agentMode,omitempty"`
	Meta      json.RawMessage `json:"meta,omitempty"`
	State     json.RawMessage `json:"state,omitempty"`
	Usage     json.RawMessage `json:"usage,omitempty"`
}

// ContentBlock represents a content block within a message.
type ContentBlock struct {
	Type      *string         `json:"type,omitempty"`
	Text      *string         `json:"text,omitempty"`
	Name      *string         `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	ToolUseID *string         `json:"tool_use_id,omitempty"`
}
