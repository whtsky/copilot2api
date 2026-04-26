package amplocal

import (
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// indexEntry holds pre-computed metadata for a single thread file.
type indexEntry struct {
	ID            string          `json:"id"`
	Title         *string         `json:"title,omitempty"`
	Created       *uint64         `json:"created,omitempty"`
	UpdatedAt     uint64          `json:"updatedAt"`
	MessageCount  int             `json:"messageCount"`
	AgentMode     *string         `json:"agentMode,omitempty"`
	V             *uint64         `json:"v,omitempty"`
	Env           json.RawMessage `json:"env,omitempty"`
	Relationships json.RawMessage `json:"relationships,omitempty"`
	UsesDTW       bool            `json:"usesDtw,omitempty"`
	Archived      bool            `json:"archived,omitempty"`
	SearchText    string          `json:"-"`
}

// State holds the in-memory thread index and the threads directory path.
type State struct {
	dir       string
	mu        sync.RWMutex
	entries   []indexEntry
	builtAt   time.Time
	staleAfter time.Duration
}

// NewState creates a new local amp state rooted at the given threads directory.
func NewState(dir string) *State {
	return &State{
		dir:        dir,
		staleAfter: 5 * time.Second,
	}
}

// ensureIndex rebuilds the index if it's stale (older than 5 seconds).
func (s *State) ensureIndex() {
	s.mu.RLock()
	fresh := time.Since(s.builtAt) < s.staleAfter
	s.mu.RUnlock()
	if fresh {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	// Double-check after acquiring write lock.
	if time.Since(s.builtAt) < s.staleAfter {
		return
	}
	s.rebuildLocked()
}

func (s *State) rebuildLocked() {
	files, err := filepath.Glob(filepath.Join(s.dir, "*.json"))
	if err != nil {
		slog.Error("amplocal: glob threads", "error", err)
		return
	}

	entries := make([]indexEntry, 0, len(files))
	for _, f := range files {
		e, err := indexFile(f)
		if err != nil {
			slog.Debug("amplocal: skip file", "path", f, "error", err)
			continue
		}
		entries = append(entries, e)
	}

	// Sort by updatedAt descending.
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].UpdatedAt > entries[j].UpdatedAt
	})

	s.entries = entries
	s.builtAt = time.Now()
}

func indexFile(path string) (indexEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return indexEntry{}, err
	}
	var tf ThreadFile
	if err := json.Unmarshal(data, &tf); err != nil {
		return indexEntry{}, err
	}

	e := indexEntry{
		ID:            tf.ID,
		Title:         tf.Title,
		Created:       tf.Created,
		MessageCount:  len(tf.Messages),
		AgentMode:     tf.AgentMode,
		V:             tf.V,
		Env:           tf.Env,
		Relationships: tf.Relationships,
	}
	if tf.Archived != nil {
		e.Archived = *tf.Archived
	}

	// updatedAt: use created as base, then check message meta for sentAt.
	if tf.Created != nil {
		e.UpdatedAt = *tf.Created
	}
	for _, m := range tf.Messages {
		if len(m.Meta) > 0 {
			var mm struct {
				SentAt *uint64 `json:"sentAt"`
			}
			if json.Unmarshal(m.Meta, &mm) == nil && mm.SentAt != nil && *mm.SentAt > e.UpdatedAt {
				e.UpdatedAt = *mm.SentAt
			}
		}
	}

	// usesDtw from meta.usesDtw
	if len(tf.Meta) > 0 {
		var meta struct {
			UsesDTW *bool `json:"usesDtw"`
		}
		if json.Unmarshal(tf.Meta, &meta) == nil && meta.UsesDTW != nil {
			e.UsesDTW = *meta.UsesDTW
		}
	}

	// Build search text from title + first 20 user/assistant text blocks.
	var sb strings.Builder
	if tf.Title != nil {
		sb.WriteString(*tf.Title)
		sb.WriteByte(' ')
	}
	textCount := 0
	for _, m := range tf.Messages {
		if textCount >= 20 {
			break
		}
		if m.Role == nil {
			continue
		}
		role := *m.Role
		if role != "user" && role != "assistant" {
			continue
		}
		for _, c := range m.Content {
			if textCount >= 20 {
				break
			}
			if c.Type != nil && *c.Type == "text" && c.Text != nil {
				sb.WriteString(*c.Text)
				sb.WriteByte(' ')
				textCount++
			}
		}
	}
	e.SearchText = strings.ToLower(sb.String())

	return e, nil
}

// Search finds threads matching the query. Returns matching entries and whether there are more.
// ListAll returns all indexed threads (sorted by updatedAt desc).
func (s *State) ListAll() []indexEntry {
	s.ensureIndex()

	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]indexEntry, len(s.entries))
	copy(result, s.entries)
	return result
}

func (s *State) Search(query string, limit, offset int) ([]indexEntry, bool) {
	s.ensureIndex()

	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 {
		limit = 50
	}

	// Parse query words, extract prefix filters.
	words := strings.Fields(strings.ToLower(query))
	var textWords []string
	for _, w := range words {
		// Skip author: and file: filters (not applicable locally).
		if strings.HasPrefix(w, "author:") || strings.HasPrefix(w, "file:") {
			continue
		}
		textWords = append(textWords, w)
	}

	var results []indexEntry
	for _, e := range s.entries {
		if len(textWords) > 0 {
			match := true
			for _, w := range textWords {
				if !strings.Contains(e.SearchText, w) {
					match = false
					break
				}
			}
			if !match {
				continue
			}
		}
		results = append(results, e)
	}

	total := len(results)
	if offset > total {
		offset = total
	}
	results = results[offset:]
	hasMore := len(results) > limit
	if hasMore {
		results = results[:limit]
	}
	return results, hasMore
}

// ReadThread reads and parses a thread file by ID.
func (s *State) ReadThread(id string) (*ThreadFile, error) {
	path := filepath.Join(s.dir, id+".json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tf ThreadFile
	if err := json.Unmarshal(data, &tf); err != nil {
		return nil, err
	}
	return &tf, nil
}

// ReadThreadRaw returns raw JSON bytes for a thread file.
func (s *State) ReadThreadRaw(id string) ([]byte, error) {
	path := filepath.Join(s.dir, id+".json")
	return os.ReadFile(path)
}

// WriteThread writes a thread file to disk (typed struct — may lose unknown fields).
func (s *State) WriteThread(tf *ThreadFile) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(tf)
	if err != nil {
		return err
	}
	path := filepath.Join(s.dir, tf.ID+".json")
	return os.WriteFile(path, data, 0o644)
}

// WriteThreadRaw writes raw JSON bytes as a thread file, preserving all fields.
func (s *State) WriteThreadRaw(id string, data []byte) error {
	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return err
	}
	path := filepath.Join(s.dir, id+".json")
	return os.WriteFile(path, data, 0o644)
}

// DeleteThread removes a thread file from disk.
func (s *State) DeleteThread(id string) error {
	path := filepath.Join(s.dir, id+".json")
	return os.Remove(path)
}

// MergeMeta reads raw thread JSON, merges the given meta fields, and writes back.
func (s *State) MergeMeta(id string, metaPatch json.RawMessage) error {
	rawData, err := s.ReadThreadRaw(id)
	if err != nil {
		return err
	}

	// Parse as generic map to preserve all fields
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(rawData, &doc); err != nil {
		return err
	}

	existing := make(map[string]json.RawMessage)
	if meta, ok := doc["meta"]; ok && len(meta) > 0 {
		if err := json.Unmarshal(meta, &existing); err != nil {
			existing = make(map[string]json.RawMessage)
		}
	}

	patch := make(map[string]json.RawMessage)
	if err := json.Unmarshal(metaPatch, &patch); err != nil {
		return err
	}
	for k, v := range patch {
		existing[k] = v
	}

	merged, err := json.Marshal(existing)
	if err != nil {
		return err
	}
	doc["meta"] = merged

	newData, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return s.WriteThreadRaw(id, newData)
}
