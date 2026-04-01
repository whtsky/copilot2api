package models

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/whtsky/copilot2api/internal/upstream"
	"golang.org/x/sync/singleflight"
)

// Info contains model metadata including supported endpoints.
type Info struct {
	ID                 string   `json:"id"`
	SupportedEndpoints []string `json:"supported_endpoints"`
}

// modelsListResponse is the response from the /models endpoint.
type modelsListResponse struct {
	Data []Info `json:"data"`
}

// cacheData holds both representations derived from a single /models fetch.
type cacheData struct {
	raw    []byte           // original JSON for serving GET /v1/models
	parsed map[string]*Info // keyed by model ID for capability lookups
}

// Cache is a shared, single-fetch cache for the /models endpoint.
// It stores both the raw JSON (for proxying) and the parsed model map
// (for capability detection), populated from one upstream call.
type Cache struct {
	mu       sync.RWMutex
	data     cacheData
	valid    bool
	cachedAt time.Time
	ttl      time.Duration
	sf       singleflight.Group
	upstream *upstream.Client
}

// NewCache creates a shared models cache.
func NewCache(upstreamClient *upstream.Client, ttl time.Duration) *Cache {
	return &Cache{
		upstream: upstreamClient,
		ttl:      ttl,
	}
}

// Warm pre-populates the cache to avoid cold-cache latency on the first request.
func (c *Cache) Warm(ctx context.Context) {
	if _, _, err := c.get(ctx); err != nil {
		slog.Warn("failed to warm models cache", "error", err)
	}
}

// GetRaw returns the raw JSON bytes of the /models response.
func (c *Cache) GetRaw(ctx context.Context) ([]byte, error) {
	raw, _, err := c.get(ctx)
	return raw, err
}

// GetInfo returns the parsed model info map.
func (c *Cache) GetInfo(ctx context.Context) (map[string]*Info, error) {
	_, parsed, err := c.get(ctx)
	return parsed, err
}

func (c *Cache) get(ctx context.Context) ([]byte, map[string]*Info, error) {
	// Fast path: read lock check.
	c.mu.RLock()
	if c.valid && time.Since(c.cachedAt) < c.ttl {
		d := c.data
		c.mu.RUnlock()
		return d.raw, d.parsed, nil
	}
	c.mu.RUnlock()

	slog.Debug("models cache miss, fetching from upstream")

	// Slow path: singleflight to deduplicate concurrent fetches.
	result, err, _ := c.sf.Do("fetch", func() (interface{}, error) {
		// Double-check after acquiring singleflight.
		c.mu.RLock()
		if c.valid && time.Since(c.cachedAt) < c.ttl {
			d := c.data
			c.mu.RUnlock()
			return d, nil
		}
		c.mu.RUnlock()

		d, err := c.fetch(context.WithoutCancel(ctx))
		if err != nil {
			return nil, err
		}

		c.mu.Lock()
		c.data = d
		c.valid = true
		c.cachedAt = time.Now()
		c.mu.Unlock()

		slog.Debug("models cache refreshed", "count", len(d.parsed))
		return d, nil
	})

	if err != nil {
		return nil, nil, err
	}
	d := result.(cacheData)
	return d.raw, d.parsed, nil
}

func (c *Cache) fetch(ctx context.Context) (cacheData, error) {
	_, respData, err := c.upstream.Do(ctx, upstream.Request{
		Method:   "GET",
		Endpoint: "/models",
	})
	if err != nil {
		return cacheData{}, fmt.Errorf("models request failed: %w", err)
	}

	var modelsResp modelsListResponse
	if err := json.Unmarshal(respData, &modelsResp); err != nil {
		return cacheData{}, fmt.Errorf("failed to parse models response: %w", err)
	}

	parsed := make(map[string]*Info, len(modelsResp.Data))
	for i := range modelsResp.Data {
		parsed[modelsResp.Data[i].ID] = &modelsResp.Data[i]
	}

	return cacheData{raw: respData, parsed: parsed}, nil
}

// PickEndpoint returns the first endpoint from preferred that the model
// supports, or "" if none match. Returns "" when info is nil (unknown model),
// letting callers fall back to default behavior.
func PickEndpoint(info *Info, preferred []string) string {
	if info == nil {
		return ""
	}
	for _, ep := range preferred {
		if SupportsEndpoint(info, ep) {
			return ep
		}
	}
	return ""
}

// SupportsEndpoint reports whether info supports the given endpoint.
func SupportsEndpoint(info *Info, endpoint string) bool {
	if info == nil {
		return false
	}

	target := normalizeEndpoint(endpoint)
	for _, ep := range info.SupportedEndpoints {
		if normalizeEndpoint(ep) == target {
			return true
		}
	}
	return false
}

func normalizeEndpoint(endpoint string) string {
	normalized := strings.TrimSpace(endpoint)
	normalized = strings.TrimPrefix(normalized, "/v1")
	if normalized == "" {
		return "/"
	}
	if !strings.HasPrefix(normalized, "/") {
		normalized = "/" + normalized
	}
	return normalized
}
