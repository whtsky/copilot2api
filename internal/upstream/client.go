package upstream

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/whtsky/copilot2api/internal/copilot"
)

// TokenProvider abstracts the auth.Client methods needed by Client.
type TokenProvider interface {
	// GetToken returns a valid bearer token for the upstream API.
	GetToken(ctx context.Context) (string, error)
	// GetBaseURL returns the base URL for the upstream API (e.g. "https://...").
	GetBaseURL() string
}

// Client makes requests to the upstream Copilot API.
type Client struct {
	TokenProvider TokenProvider
	HTTPClient    *http.Client
	Debug         bool
}

// NewTransport creates a shared http.Transport suitable for upstream requests.
func NewTransport() *http.Transport {
	return &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 30 * time.Second}).DialContext,
		TLSClientConfig:      &tls.Config{MinVersion: tls.VersionTLS12},
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       120 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 5 * time.Minute,
	}
}

// NewClient creates a new upstream Client.
// If transport is non-nil it is used for the underlying http.Client;
// otherwise a new Transport is created.
func NewClient(tp TokenProvider, transport *http.Transport, debug bool) *Client {
	if transport == nil {
		transport = NewTransport()
	}
	return &Client{
		TokenProvider: tp,
		HTTPClient: &http.Client{
			// No client-level Timeout — it kills long-running streaming requests.
			// Non-streaming requests use per-request context timeouts instead.
			Transport: transport,
		},
		Debug: debug,
	}
}

// Request configures a single upstream request.
type Request struct {
	Method      string
	Endpoint    string
	Body        interface{} // []byte, io.Reader, or JSON-marshalable struct; nil for no body
	QueryString string      // raw query string to append (without leading '?')
	Stream      bool        // if true, returns *http.Response instead of reading body
	ExtraHeaders map[string]string // additional headers to set after copilot headers
}

const (
	defaultNonStreamTimeout = 5 * time.Minute
	maxErrBody              = 1 << 20  // 1MB for error bodies
	maxRespBody             = 50 << 20 // 50MB for response bodies
	MaxRequestBody          = 10 << 20 // 10MB for incoming request bodies
)

// Do executes a request against the upstream Copilot API.
//
// For non-streaming (Stream=false): returns (*Response, nil) or (nil, error).
// For streaming (Stream=true): returns (*Response, nil) where Response.StreamResp
// is set, or (nil, error). The caller must close the HTTP response body.
func (c *Client) Do(ctx context.Context, r Request) (*http.Response, []byte, error) {
	token, err := c.TokenProvider.GetToken(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get valid token: %w", err)
	}

	// Resolve body to io.Reader
	var bodyReader io.Reader
	switch v := r.Body.(type) {
	case nil:
		// no body
	case []byte:
		bodyReader = bytes.NewReader(v)
	case io.Reader:
		bodyReader = v
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	// Build URL
	baseURL := c.TokenProvider.GetBaseURL()
	upstreamURL := baseURL + r.Endpoint
	if r.QueryString != "" {
		upstreamURL += "?" + r.QueryString
	}

	// Apply timeout for non-streaming requests
	reqCtx := ctx
	var cancel context.CancelFunc
	if !r.Stream {
		reqCtx, cancel = context.WithTimeout(ctx, defaultNonStreamTimeout)
		defer cancel()
	}

	method := r.Method
	if method == "" {
		method = "POST"
	}

	req, err := http.NewRequestWithContext(reqCtx, method, upstreamURL, bodyReader)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	copilot.AddHeaders(req, token)

	if r.Stream {
		req.Header.Set("Accept", "text/event-stream")
	}

	for k, v := range r.ExtraHeaders {
		req.Header.Set(k, v)
	}

	// Debug log: outgoing request
	if c.Debug {
		if bodyReader != nil {
			if br, ok := bodyReader.(*bytes.Reader); ok {
				rawBytes := make([]byte, br.Len())
				br.Read(rawBytes)
				br.Seek(0, io.SeekStart)
				slog.Debug("upstream request", "method", method, "url", upstreamURL, "body", truncateStr(string(rawBytes), 2000))
			}
		} else {
			slog.Debug("upstream request", "method", method, "url", upstreamURL)
		}
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, int64(maxErrBody)+1))
		if len(errBody) > maxErrBody {
			return nil, nil, fmt.Errorf("upstream error response too large (exceeds %d bytes)", maxErrBody)
		}
		slog.Debug("upstream error response", "endpoint", r.Endpoint, "status", resp.StatusCode, "body", truncateStr(string(errBody), 2000))
		return nil, nil, &UpstreamError{
			StatusCode: resp.StatusCode,
			Body:       errBody,
		}
	}

	if r.Stream {
		return resp, nil, nil
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxRespBody)+1))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read response: %w", err)
	}
	if len(respData) > maxRespBody {
		return nil, nil, fmt.Errorf("upstream response too large (exceeds %d bytes)", maxRespBody)
	}
	slog.Debug("upstream response", "endpoint", r.Endpoint, "status", resp.StatusCode, "body", truncateStr(string(respData), 2000))
	return nil, respData, nil
}

func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
