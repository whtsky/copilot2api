package ampsearch

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type PageContent struct {
	FullContent string   `json:"fullContent"`
	Excerpts    []string `json:"excerpts"`
}

type Backend interface {
	Search(queries []string, maxResults int) ([]SearchResult, error)
	ExtractPage(pageURL string) (*PageContent, error)
}

type Handler struct {
	backend Backend
}

func NewHandler(backend Backend) *Handler {
	return &Handler{backend: backend}
}

func (h *Handler) TryServe(w http.ResponseWriter, r *http.Request) bool {
	if r.Method != http.MethodPost {
		return false
	}

	query := r.URL.RawQuery
	method := query
	if i := strings.IndexByte(method, '&'); i >= 0 {
		method = method[:i]
	}

	switch method {
	case "webSearch2":
		h.handleWebSearch(w, r)
		return true
	case "extractWebPageContent":
		h.handleExtractPage(w, r)
		return true
	default:
		return false
	}
}

func (h *Handler) handleWebSearch(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, map[string]any{"ok": false, "error": map[string]string{"message": "failed to read body"}})
		return
	}

	var parsed struct {
		Params struct {
			SearchQueries []string `json:"searchQueries"`
			Objective     string   `json:"objective"`
			MaxResults    int      `json:"maxResults"`
		} `json:"params"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		writeJSON(w, map[string]any{"ok": false, "error": map[string]string{"message": "invalid json"}})
		return
	}

	queries := parsed.Params.SearchQueries
	if len(queries) == 0 && parsed.Params.Objective != "" {
		queries = []string{parsed.Params.Objective}
	}
	maxResults := parsed.Params.MaxResults
	if maxResults <= 0 {
		maxResults = 5
	}

	results, err := h.backend.Search(queries, maxResults)
	if err != nil {
		slog.Warn("web search failed", "error", err)
		results = []SearchResult{}
	}
	if results == nil {
		results = []SearchResult{}
	}

	writeJSON(w, map[string]any{
		"ok": true,
		"result": map[string]any{
			"results":                 results,
			"showParallelAttribution": false,
		},
	})
}

func (h *Handler) handleExtractPage(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeJSON(w, map[string]any{"ok": false, "error": map[string]string{"message": "failed to read body"}})
		return
	}

	var parsed struct {
		Params struct {
			URL string `json:"url"`
		} `json:"params"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		writeJSON(w, map[string]any{"ok": false, "error": map[string]string{"message": "invalid json"}})
		return
	}

	if parsed.Params.URL == "" {
		writeJSON(w, map[string]any{
			"ok":    false,
			"error": map[string]string{"code": "invalid-request", "message": "missing url"},
		})
		return
	}

	page, err := h.backend.ExtractPage(parsed.Params.URL)
	if err != nil {
		slog.Warn("page extraction failed", "url", parsed.Params.URL, "error", err)
		writeJSON(w, map[string]any{
			"ok":    false,
			"error": map[string]string{"code": "upstream-error", "message": err.Error()},
		})
		return
	}

	writeJSON(w, map[string]any{
		"ok":     true,
		"result": page,
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
