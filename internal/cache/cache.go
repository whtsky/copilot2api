package cache

import (
	"context"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

// Cache is a generic TTL cache that deduplicates concurrent fetches via singleflight.
type Cache[T any] struct {
	mu       sync.RWMutex
	data     T
	valid    bool
	cachedAt time.Time
	ttl      time.Duration
	sf       singleflight.Group
}

// New creates a Cache with the given TTL.
func New[T any](ttl time.Duration) *Cache[T] {
	return &Cache[T]{ttl: ttl}
}

// Get returns the cached value if still valid, otherwise calls fetchFunc to
// refresh it. Concurrent callers share a single in-flight fetch.
func (c *Cache[T]) Get(ctx context.Context, fetchFunc func(ctx context.Context) (T, error)) (T, error) {
	// Fast path: read lock check.
	if val, ok := c.get(); ok {
		return val, nil
	}

	// Slow path: singleflight to deduplicate concurrent fetches.
	result, err, _ := c.sf.Do("fetch", func() (interface{}, error) {
		// Double-check: another goroutine may have refreshed while we waited.
		if val, ok := c.get(); ok {
			return val, nil
		}

		val, err := fetchFunc(context.WithoutCancel(ctx))
		if err != nil {
			return nil, err
		}

		c.mu.Lock()
		c.data = val
		c.valid = true
		c.cachedAt = time.Now()
		c.mu.Unlock()

		return val, nil
	})

	if err != nil {
		var zero T
		return zero, err
	}
	return result.(T), nil
}

// get returns the cached value under a read lock if it's valid and not expired.
func (c *Cache[T]) get() (T, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.valid && time.Since(c.cachedAt) < c.ttl {
		return c.data, true
	}
	var zero T
	return zero, false
}
