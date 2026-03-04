package sse

import "net/http"

// BeginSSE sets the standard SSE response headers on w.
func BeginSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
}
