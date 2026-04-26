package amplocal

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

// ShouldHandleLocally returns true if the given URL path should be served
// from local thread data instead of proxied to ampcode.com.
func ShouldHandleLocally(path string) bool {
	p := strings.TrimPrefix(path, "/api/")
	if p == path {
		// Not under /api/ — check other known paths.
		return path == "/news.rss"
	}
	switch {
	case p == "internal":
		return true
	case p == "telemetry":
		return true
	case p == "attachments":
		return true
	case strings.HasPrefix(p, "threads/"):
		return true
	case strings.HasPrefix(p, "threads"):
		return true
	case strings.HasPrefix(p, "durable-thread-workers/"):
		return true
	case strings.HasPrefix(p, "durable-thread-workers"):
		return true
	case strings.HasPrefix(p, "users/"):
		return true
	case strings.HasPrefix(p, "users"):
		return true
	default:
		return false
	}
}

// Handler serves local amp API requests.
type Handler struct {
	state *State
}

// NewHandler creates a handler backed by the given state.
func NewHandler(state *State) *Handler {
	return &Handler{state: state}
}

// ServeThreadsFind handles GET /api/threads/find.
func (h *Handler) ServeThreadsFind(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	query := q.Get("q")
	limit := intParam(q.Get("limit"), 50)
	offset := intParam(q.Get("offset"), 0)

	entries, hasMore := h.state.Search(query, limit, offset)

	type threadResult struct {
		ID                string  `json:"id"`
		Title             *string `json:"title,omitempty"`
		CreatorUserID     string  `json:"creatorUserID"`
		Created           *uint64 `json:"created,omitempty"`
		UpdatedAt         uint64  `json:"updatedAt"`
		MessageCount      int     `json:"messageCount"`
		MatchedSearchText string  `json:"matchedSearchText,omitempty"`
	}

	threads := make([]threadResult, len(entries))
	for i, e := range entries {
		threads[i] = threadResult{
			ID:            e.ID,
			Title:         e.Title,
			CreatorUserID: "local-user",
			Created:       e.Created,
			UpdatedAt:     e.UpdatedAt,
			MessageCount:  e.MessageCount,
		}
		// Return a snippet of search text if there was a query.
		if query != "" && len(e.SearchText) > 0 {
			snippet := e.SearchText
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			threads[i].MatchedSearchText = snippet
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"threads": threads,
		"hasMore": hasMore,
	})
}

// ServeThreadMarkdown handles GET /api/threads/{id}.md.
func (h *Handler) ServeThreadMarkdown(w http.ResponseWriter, r *http.Request) {
	// Extract thread ID from path: /api/threads/{id}.md
	path := r.URL.Path
	path = strings.TrimPrefix(path, "/api/threads/")
	id := strings.TrimSuffix(path, ".md")
	if id == "" {
		http.Error(w, "missing thread id", http.StatusBadRequest)
		return
	}

	tf, err := h.state.ReadThread(id)
	if err != nil {
		if os.IsNotExist(err) {
			http.Error(w, "thread not found", http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	truncate := r.URL.Query().Get("truncate_tool_results") == "1"
	md := renderMarkdown(tf, truncate)
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	w.Write([]byte(md))
}

// TryServeInternal handles /api/internal requests. Returns true if handled.
func (h *Handler) TryServeInternal(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}

	query := r.URL.RawQuery
	method := query
	if i := strings.IndexByte(method, '&'); i >= 0 {
		method = method[:i]
	}

	switch method {
	case "uploadThread":
		h.handleUploadThread(w, r)
		return true
	case "setThreadMeta":
		h.handleSetThreadMeta(w, r)
		return true
	case "deleteThread":
		h.handleDeleteThread(w, r)
		return true
	case "getUserInfo":
		writeJSON(w, http.StatusOK, map[string]any{
			"ok": true,
			"result": map[string]any{
				"id":               "local-user",
				"email":            "local@localhost",
				"displayName":      "Local User",
				"avatarURL":        nil,
				"features":         []any{},
				"team":             nil,
				"mysteriousMessage": nil,
			},
		})
		return true
	case "shareThread":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{}})
		return true
	case "getThreadLabels":
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":     true,
			"result": map[string]any{"labels": []any{}},
		})
		return true
	case "setThreadLabels":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{}})
		return true
	case "addThreadLabels":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{}})
		return true
	case "listThreads":
		h.handleListThreads(w, r)
		return true
	case "getThread":
		h.handleGetThread(w, r)
		return true
	case "getThreadLinkInfo":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": nil})
		return true
	case "createRemoteExecutorThread":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": nil})
		return true
	case "shareThreadWithOperator":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{"url": ""}})
		return true
	case "getUserLabels":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": []any{}})
		return true
	case "threadDisplayCostInfo":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": nil})
		return true
	case "userDisplayBalanceInfo":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": nil})
		return true
	case "getUserFreeTierStatus":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{"canUseAmpFree": false, "isDailyGrantEnabled": false}})
		return true
	case "github-auth-status":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{"authenticated": true}})
		return true
	case "listTasks":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": map[string]any{"tasks": []any{}}})
		return true
	case "markAsReadMysteriousMessage":
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		return true
	default:
		// Catch-all for unknown methods in local mode — return ok stub
		slog.Debug("amplocal: unhandled internal method", "method", method)
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "result": nil})
		return true
	}
}

func (h *Handler) handleListThreads(w http.ResponseWriter, r *http.Request) {
	entries := h.state.ListAll()

	type threadSummary struct {
		ID                    string          `json:"id"`
		V                     *uint64         `json:"v,omitempty"`
		Title                 *string         `json:"title,omitempty"`
		Created               *uint64         `json:"created,omitempty"`
		UpdatedAt             uint64          `json:"updatedAt"`
		UserLastInteractedAt  uint64          `json:"userLastInteractedAt"`
		MessageCount          int             `json:"messageCount"`
		AgentMode             *string         `json:"agentMode,omitempty"`
		Archived              bool            `json:"archived"`
		UsesDTW               bool            `json:"usesDtw"`
		Env                   json.RawMessage `json:"env,omitempty"`
		Relationships         json.RawMessage `json:"relationships,omitempty"`
		SummaryStats          map[string]any  `json:"summaryStats"`
	}

	threads := make([]threadSummary, 0, len(entries))
	for _, e := range entries {
		if e.MessageCount == 0 {
			continue
		}
		threads = append(threads, threadSummary{
			ID:                   e.ID,
			V:                    e.V,
			Title:                e.Title,
			Created:              e.Created,
			UpdatedAt:            e.UpdatedAt,
			UserLastInteractedAt: e.UpdatedAt,
			MessageCount:         e.MessageCount,
			AgentMode:            e.AgentMode,
			Archived:             e.Archived,
			UsesDTW:              e.UsesDTW,
			Env:                  e.Env,
			Relationships:        e.Relationships,
			SummaryStats: map[string]any{
				"messageCount": e.MessageCount,
				"diffStats":    nil,
			},
		})
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"ok":     true,
		"result": map[string]any{"threads": threads},
	})
}

func (h *Handler) handleGetThread(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Try params.thread, params.threadId, threadId at top level
	var req struct {
		ThreadID string `json:"threadId"`
		Params   struct {
			ThreadID string `json:"threadId"`
			Thread   string `json:"thread"`
		} `json:"params"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}
	threadID := req.ThreadID
	if threadID == "" {
		threadID = req.Params.ThreadID
	}
	if threadID == "" {
		threadID = req.Params.Thread
	}
	if threadID == "" {
		slog.Debug("amplocal: getThread missing threadId", "body", string(body[:min(300, len(body))]))
		writeJSON(w, http.StatusOK, map[string]any{"ok": false, "error": map[string]any{"code": "invalid-request", "message": "missing thread id"}})
		return
	}

	// Read raw JSON to preserve all fields exactly
	rawData, err := h.state.ReadThreadRaw(threadID)
	if err != nil {
		if os.IsNotExist(err) {
			writeJSON(w, http.StatusOK, map[string]any{
				"ok": false,
				"error": map[string]any{
					"code":    "thread-not-found",
					"message": fmt.Sprintf("Thread %s not found", threadID),
				},
			})
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	// Parse as generic JSON to preserve all fields
	var threadData json.RawMessage = rawData

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"ok":true,"result":{"thread":{"data":%s},"run":{"status":"completed"}}}`+"\n", string(threadData))
}

func (h *Handler) handleUploadThread(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		slog.Error("amplocal: upload read body", "error", err, "content-encoding", r.Header.Get("Content-Encoding"))
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Amp may wrap thread data in {"params": {"thread": ...}} envelope.
	var envelope struct {
		Params struct {
			Thread json.RawMessage `json:"thread"`
		} `json:"params"`
	}
	threadData := body
	if err := json.Unmarshal(body, &envelope); err == nil && len(envelope.Params.Thread) > 0 {
		threadData = envelope.Params.Thread
	}

	// Only extract ID for file naming — write raw JSON to preserve all fields
	var partial struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(threadData, &partial); err != nil {
		slog.Error("amplocal: upload unmarshal", "error", err, "bodyLen", len(body), "snippet", string(body[:min(200, len(body))]))
		http.Error(w, "invalid json: "+err.Error(), http.StatusBadRequest)
		return
	}
	if partial.ID == "" {
		http.Error(w, "missing thread id", http.StatusBadRequest)
		return
	}

	if err := h.state.WriteThreadRaw(partial.ID, threadData); err != nil {
		slog.Error("amplocal: write thread", "id", partial.ID, "error", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (h *Handler) handleSetThreadMeta(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var req struct {
		ThreadID string          `json:"threadId"`
		Meta     json.RawMessage `json:"meta"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	if err := h.state.MergeMeta(req.ThreadID, req.Meta); err != nil {
		if os.IsNotExist(err) {
			http.Error(w, "thread not found", http.StatusNotFound)
		} else {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

func (h *Handler) handleDeleteThread(w http.ResponseWriter, r *http.Request) {
	body, err := readBody(r)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var req struct {
		ThreadID string `json:"threadId"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid json", http.StatusBadRequest)
		return
	}

	if err := h.state.DeleteThread(req.ThreadID); err != nil && !os.IsNotExist(err) {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"ok": true})
}

// ServeTelemetry handles POST /api/telemetry — no-op.
func (h *Handler) ServeTelemetry(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
}

// ServeDurableThreadWorker handles POST /api/durable-thread-workers/{id}.
func (h *Handler) ServeDurableThreadWorker(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/durable-thread-workers/")
	if id == "" {
		id = "unknown"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":           id,
		"status":       "running",
		"executorType": "local-client",
	})
}

// ServeUsers handles GET /api/users/{id}.
func (h *Handler) ServeUsers(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"id":    "local-user",
		"name":  "Local User",
		"email": "local@localhost",
	})
}

// ServeAttachments handles GET /api/attachments.
func (h *Handler) ServeAttachments(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"attachments": []any{}})
}

// ServeNewsRSS handles GET /news.rss.
func (h *Handler) ServeNewsRSS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/rss+xml; charset=utf-8")
	w.Write([]byte(`<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Amp News</title>
    <description>Local mode — no news</description>
  </channel>
</rss>`))
}

// --- helpers ---

func readBody(r *http.Request) ([]byte, error) {
	var reader io.Reader = r.Body
	if r.Header.Get("Content-Encoding") == "gzip" {
		gz, err := gzip.NewReader(r.Body)
		if err != nil {
			return nil, fmt.Errorf("gzip decode: %w", err)
		}
		defer gz.Close()
		reader = gz
	}
	return io.ReadAll(reader)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func intParam(s string, def int) int {
	if s == "" {
		return def
	}
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
		return def
	}
	return n
}

// renderMarkdown converts a thread to markdown.
func renderMarkdown(tf *ThreadFile, truncateToolResults bool) string {
	var sb strings.Builder

	// Title
	if tf.Title != nil {
		sb.WriteString("# ")
		sb.WriteString(*tf.Title)
		sb.WriteString("\n\n")
	}

	// Metadata
	sb.WriteString(fmt.Sprintf("Thread ID: %s\n", tf.ID))
	if tf.Created != nil {
		sb.WriteString(fmt.Sprintf("Created: %d\n", *tf.Created))
	}
	if tf.AgentMode != nil {
		sb.WriteString(fmt.Sprintf("Agent Mode: %s\n", *tf.AgentMode))
	}
	sb.WriteString("\n---\n\n")

	// Messages
	for _, m := range tf.Messages {
		role := "unknown"
		if m.Role != nil {
			role = *m.Role
		}
		sb.WriteString(fmt.Sprintf("## %s\n\n", role))

		for _, c := range m.Content {
			ctype := ""
			if c.Type != nil {
				ctype = *c.Type
			}

			switch ctype {
			case "text":
				if c.Text != nil {
					sb.WriteString(*c.Text)
					sb.WriteString("\n\n")
				}
			case "tool_use":
				name := ""
				if c.Name != nil {
					name = *c.Name
				}
				sb.WriteString(fmt.Sprintf("**Tool Use: %s**\n", name))
				if len(c.Input) > 0 {
					sb.WriteString("```json\n")
					sb.Write(c.Input)
					sb.WriteString("\n```\n\n")
				}
			case "tool_result":
				sb.WriteString("**Tool Result**\n")
				if truncateToolResults {
					sb.WriteString("_(truncated)_\n\n")
				} else if len(c.Content) > 0 {
					sb.WriteString("```\n")
					sb.Write(c.Content)
					sb.WriteString("\n```\n\n")
				}
			default:
				if c.Text != nil {
					sb.WriteString(*c.Text)
					sb.WriteString("\n\n")
				}
			}
		}
	}

	return sb.String()
}
