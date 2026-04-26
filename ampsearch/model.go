package ampsearch

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"

	"github.com/whtsky/copilot2api/internal/upstream"
)

type ModelBackend struct {
	client     *upstream.Client
	model      string
	httpClient *http.Client
}

func NewModelBackend(client *upstream.Client, model string) *ModelBackend {
	if model == "" {
		model = "gpt-5-mini"
	}
	return &ModelBackend{client: client, model: model, httpClient: &http.Client{Timeout: 30 * time.Second}}
}

func (m *ModelBackend) Search(queries []string, maxResults int) ([]SearchResult, error) {
	var allResults []SearchResult
	for _, q := range queries {
		if q == "" {
			continue
		}
		results, err := m.searchOne(q)
		if err != nil {
			slog.Warn("model search failed", "query", q, "error", err)
			continue
		}
		allResults = append(allResults, results...)
		if len(allResults) >= maxResults {
			break
		}
	}
	if len(allResults) > maxResults {
		allResults = allResults[:maxResults]
	}
	return allResults, nil
}

func (m *ModelBackend) searchOne(query string) ([]SearchResult, error) {
	body := map[string]any{
		"model": m.model,
		"input": fmt.Sprintf("Search the web for: %s. Return a list of the most relevant results.", query),
		"tools": []map[string]any{{"type": "web_search"}},
	}

	_, respBody, err := m.client.Do(context.Background(), upstream.Request{
		Method:   "POST",
		Endpoint: "/responses",
		Body:     body,
		ExtraHeaders: map[string]string{
			"Content-Type": "application/json",
		},
	})
	if err != nil {
		return nil, fmt.Errorf("copilot request failed: %w", err)
	}

	var resp struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				Annotations []struct {
					Type       string `json:"type"`
					URL        string `json:"url"`
					Title      string `json:"title"`
					StartIndex int    `json:"start_index"`
					EndIndex   int    `json:"end_index"`
				} `json:"annotations"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(respBody, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	var results []SearchResult
	seen := make(map[string]bool)

	for _, item := range resp.Output {
		if item.Type != "message" {
			continue
		}
		for _, block := range item.Content {
			if block.Type != "output_text" {
				continue
			}
			for _, ann := range block.Annotations {
				if ann.Type != "url_citation" || ann.URL == "" || seen[ann.URL] {
					continue
				}
				seen[ann.URL] = true
				title := ann.Title
				if title == "" {
					title = ann.URL
				}
				snippet := extractCitationContext(block.Text, ann.StartIndex, ann.EndIndex)
				results = append(results, SearchResult{
					Title:   title,
					URL:     ann.URL,
					Content: snippet,
				})
			}
			if len(results) == 0 && block.Text != "" {
				results = append(results, SearchResult{
					Title:   query,
					Content: block.Text,
				})
			}
		}
	}
	return results, nil
}

func extractCitationContext(text string, start, end int) string {
	if start > 0 && start <= len(text) {
		before := text[:start]
		contextStart := len(before)
		if idx := lastIndex(before, "\n\n"); idx >= 0 {
			contextStart = idx + 2
		} else if idx := lastIndex(before, ". "); idx >= 0 {
			contextStart = idx + 2
		} else if contextStart > 200 {
			contextStart = len(before) - 200
		}
		snippet := before[contextStart:]
		snippet = trimRight(snippet, "([")
		if snippet != "" {
			return snippet
		}
	}
	if end > 0 && end < len(text) {
		s := end - 200
		if s < 0 {
			s = 0
		}
		return text[s:end]
	}
	return ""
}

func (m *ModelBackend) ExtractPage(pageURL string) (*PageContent, error) {
	req, err := http.NewRequest("GET", "https://r.jina.ai/"+pageURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-No-Cache", "true")
	if key := os.Getenv("JINA_API_KEY"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("jina reader request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("jina reader returned %d", resp.StatusCode)
	}

	var body struct {
		Data struct {
			Content string `json:"content"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, err
	}
	return &PageContent{
		FullContent: body.Data.Content,
		Excerpts:    []string{},
	}, nil
}

func lastIndex(s, substr string) int {
	for i := len(s) - len(substr); i >= 0; i-- {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func trimRight(s string, cutset string) string {
	for len(s) > 0 {
		found := false
		for _, c := range cutset {
			if rune(s[len(s)-1]) == c {
				s = s[:len(s)-1]
				found = true
				break
			}
		}
		if !found {
			break
		}
	}
	return s
}
