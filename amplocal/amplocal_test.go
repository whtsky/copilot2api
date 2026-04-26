package amplocal

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func ptr[T any](v T) *T { return &v }

func setupTestState(t *testing.T) (*State, string) {
	t.Helper()
	dir := t.TempDir()
	s := NewState(dir)
	return s, dir
}

func writeTestThread(t *testing.T, dir string, tf *ThreadFile) {
	t.Helper()
	data, err := json.Marshal(tf)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, tf.ID+".json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestIndexAndSearch(t *testing.T) {
	s, dir := setupTestState(t)

	writeTestThread(t, dir, &ThreadFile{
		ID:    "t1",
		Title: ptr("Hello World"),
		Created: ptr(uint64(1000)),
		Messages: []ThreadMessage{
			{Role: ptr("user"), Content: []ContentBlock{{Type: ptr("text"), Text: ptr("how are you")}}},
			{Role: ptr("assistant"), Content: []ContentBlock{{Type: ptr("text"), Text: ptr("I am fine")}}},
		},
	})
	writeTestThread(t, dir, &ThreadFile{
		ID:    "t2",
		Title: ptr("Golang Tips"),
		Created: ptr(uint64(2000)),
		Messages: []ThreadMessage{
			{Role: ptr("user"), Content: []ContentBlock{{Type: ptr("text"), Text: ptr("tell me about goroutines")}}},
		},
	})

	// Search all
	entries, hasMore := s.Search("", 50, 0)
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if hasMore {
		t.Fatal("unexpected hasMore")
	}
	// Should be sorted by updatedAt desc
	if entries[0].ID != "t2" {
		t.Fatalf("expected t2 first, got %s", entries[0].ID)
	}

	// Search by word
	entries, _ = s.Search("golang", 50, 0)
	if len(entries) != 1 || entries[0].ID != "t2" {
		t.Fatalf("expected t2 for 'golang', got %v", entries)
	}

	// Multi-word search (all must match)
	entries, _ = s.Search("hello you", 50, 0)
	if len(entries) != 1 || entries[0].ID != "t1" {
		t.Fatalf("expected t1 for 'hello you', got %v", entries)
	}

	// No match
	entries, _ = s.Search("nonexistent", 50, 0)
	if len(entries) != 0 {
		t.Fatalf("expected 0 entries, got %d", len(entries))
	}

	// Pagination
	entries, hasMore = s.Search("", 1, 0)
	if len(entries) != 1 || !hasMore {
		t.Fatalf("expected 1 entry with hasMore, got %d/%v", len(entries), hasMore)
	}
	entries, hasMore = s.Search("", 1, 1)
	if len(entries) != 1 || hasMore {
		t.Fatalf("expected 1 entry without hasMore, got %d/%v", len(entries), hasMore)
	}
}

func TestSearchWithPrefixFilters(t *testing.T) {
	s, dir := setupTestState(t)
	writeTestThread(t, dir, &ThreadFile{
		ID:    "t1",
		Title: ptr("Test Thread"),
		Created: ptr(uint64(1000)),
	})

	// author: and file: filters are ignored, should still return results
	entries, _ := s.Search("author:someone test", 50, 0)
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
}

func TestShouldHandleLocally(t *testing.T) {
	tests := []struct {
		path   string
		expect bool
	}{
		{"/api/internal", true},
		{"/api/telemetry", true},
		{"/api/threads/find", true},
		{"/api/threads/abc.md", true},
		{"/api/durable-thread-workers/xyz", true},
		{"/api/users/123", true},
		{"/api/attachments", true},
		{"/news.rss", true},
		{"/api/provider/openai/v1/chat/completions", false},
		{"/v1/models", false},
		{"/api/some-unknown", false},
	}
	for _, tt := range tests {
		if got := ShouldHandleLocally(tt.path); got != tt.expect {
			t.Errorf("ShouldHandleLocally(%q) = %v, want %v", tt.path, got, tt.expect)
		}
	}
}

func TestUploadThread(t *testing.T) {
	s, _ := setupTestState(t)
	h := NewHandler(s)

	tf := ThreadFile{ID: "upload-1", Title: ptr("Uploaded")}
	body, _ := json.Marshal(tf)

	req := httptest.NewRequest("POST", "/api/internal?uploadThread", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.handleUploadThread(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify file exists
	read, err := s.ReadThread("upload-1")
	if err != nil {
		t.Fatal(err)
	}
	if *read.Title != "Uploaded" {
		t.Fatalf("expected title 'Uploaded', got %v", read.Title)
	}
}

func TestUploadThreadGzip(t *testing.T) {
	s, _ := setupTestState(t)
	h := NewHandler(s)

	tf := ThreadFile{ID: "gz-1", Title: ptr("Gzipped")}
	body, _ := json.Marshal(tf)

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	gz.Write(body)
	gz.Close()

	req := httptest.NewRequest("POST", "/api/internal?uploadThread", &buf)
	req.Header.Set("Content-Encoding", "gzip")
	w := httptest.NewRecorder()
	h.handleUploadThread(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	read, err := s.ReadThread("gz-1")
	if err != nil {
		t.Fatal(err)
	}
	if *read.Title != "Gzipped" {
		t.Fatalf("expected title 'Gzipped', got %v", read.Title)
	}
}

func TestDeleteThread(t *testing.T) {
	s, dir := setupTestState(t)
	h := NewHandler(s)

	writeTestThread(t, dir, &ThreadFile{ID: "del-1", Title: ptr("Delete Me")})

	body, _ := json.Marshal(map[string]string{"threadId": "del-1"})
	req := httptest.NewRequest("POST", "/api/internal?deleteThread", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.handleDeleteThread(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	_, err := s.ReadThread("del-1")
	if !os.IsNotExist(err) {
		t.Fatalf("expected file not found, got %v", err)
	}
}

func TestSetThreadMeta(t *testing.T) {
	s, dir := setupTestState(t)
	h := NewHandler(s)

	writeTestThread(t, dir, &ThreadFile{
		ID:   "meta-1",
		Meta: json.RawMessage(`{"existing": true}`),
	})

	body, _ := json.Marshal(map[string]any{
		"threadId": "meta-1",
		"meta":     map[string]any{"newField": "value"},
	})
	req := httptest.NewRequest("POST", "/api/internal?setThreadMeta", bytes.NewReader(body))
	w := httptest.NewRecorder()
	h.handleSetThreadMeta(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	read, _ := s.ReadThread("meta-1")
	var meta map[string]any
	json.Unmarshal(read.Meta, &meta)
	if meta["existing"] != true || meta["newField"] != "value" {
		t.Fatalf("unexpected meta: %v", meta)
	}
}

func TestThreadMarkdown(t *testing.T) {
	s, dir := setupTestState(t)
	h := NewHandler(s)

	writeTestThread(t, dir, &ThreadFile{
		ID:    "md-1",
		Title: ptr("Test MD"),
		Messages: []ThreadMessage{
			{Role: ptr("user"), Content: []ContentBlock{{Type: ptr("text"), Text: ptr("hello")}}},
			{Role: ptr("assistant"), Content: []ContentBlock{
				{Type: ptr("tool_use"), Name: ptr("read_file"), Input: json.RawMessage(`{"path":"x"}`)},
			}},
			{Role: ptr("user"), Content: []ContentBlock{
				{Type: ptr("tool_result"), Content: json.RawMessage(`"file contents"`)},
			}},
		},
	})

	req := httptest.NewRequest("GET", "/api/threads/md-1.md", nil)
	w := httptest.NewRecorder()
	h.ServeThreadMarkdown(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !bytes.Contains([]byte(body), []byte("# Test MD")) {
		t.Fatal("missing title in markdown")
	}
	if !bytes.Contains([]byte(body), []byte("hello")) {
		t.Fatal("missing message text")
	}

	// Test with truncation
	req = httptest.NewRequest("GET", "/api/threads/md-1.md?truncate_tool_results=1", nil)
	w = httptest.NewRecorder()
	h.ServeThreadMarkdown(w, req)
	body = w.Body.String()
	if !bytes.Contains([]byte(body), []byte("_(truncated)_")) {
		t.Fatal("expected truncated tool result")
	}
}

func TestStubEndpoints(t *testing.T) {
	s, _ := setupTestState(t)
	h := NewHandler(s)

	// getUserInfo
	req := httptest.NewRequest("POST", "/api/internal?getUserInfo", nil)
	w := httptest.NewRecorder()
	handled := h.TryServeInternal(w, req)
	if !handled || w.Code != 200 {
		t.Fatal("getUserInfo failed")
	}

	// getThreadLabels
	req = httptest.NewRequest("POST", "/api/internal?getThreadLabels", nil)
	w = httptest.NewRecorder()
	handled = h.TryServeInternal(w, req)
	if !handled || w.Code != 200 {
		t.Fatal("getThreadLabels failed")
	}

	// telemetry
	req = httptest.NewRequest("POST", "/api/telemetry", nil)
	w = httptest.NewRecorder()
	h.ServeTelemetry(w, req)
	if w.Code != 200 {
		t.Fatal("telemetry failed")
	}

	// attachments
	req = httptest.NewRequest("GET", "/api/attachments", nil)
	w = httptest.NewRecorder()
	h.ServeAttachments(w, req)
	if w.Code != 200 {
		t.Fatal("attachments failed")
	}

	// users
	req = httptest.NewRequest("GET", "/api/users/123", nil)
	w = httptest.NewRecorder()
	h.ServeUsers(w, req)
	if w.Code != 200 {
		t.Fatal("users failed")
	}

	// news.rss
	req = httptest.NewRequest("GET", "/news.rss", nil)
	w = httptest.NewRecorder()
	h.ServeNewsRSS(w, req)
	if w.Code != 200 || w.Header().Get("Content-Type") != "application/rss+xml; charset=utf-8" {
		t.Fatal("news.rss failed")
	}

	// durable-thread-workers
	req = httptest.NewRequest("POST", "/api/durable-thread-workers/xyz", nil)
	w = httptest.NewRecorder()
	h.ServeDurableThreadWorker(w, req)
	if w.Code != 200 {
		t.Fatal("durable-thread-workers failed")
	}
}

func TestThreadsFindHTTP(t *testing.T) {
	s, dir := setupTestState(t)
	h := NewHandler(s)

	writeTestThread(t, dir, &ThreadFile{
		ID:      "find-1",
		Title:   ptr("Alpha Thread"),
		Created: ptr(uint64(1000)),
	})

	req := httptest.NewRequest("GET", "/api/threads/find?q=alpha&limit=10", nil)
	w := httptest.NewRecorder()
	h.ServeThreadsFind(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp struct {
		Threads []struct {
			ID    string `json:"id"`
			Title string `json:"title"`
		} `json:"threads"`
		HasMore bool `json:"hasMore"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp.Threads) != 1 || resp.Threads[0].ID != "find-1" {
		t.Fatalf("unexpected response: %+v", resp)
	}
}
