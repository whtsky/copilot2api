package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/whtsky/copilot2api/anthropic"
	"github.com/whtsky/copilot2api/auth"
	"github.com/whtsky/copilot2api/gemini"
	"github.com/whtsky/copilot2api/internal/models"
	"github.com/whtsky/copilot2api/internal/upstream"
	"github.com/whtsky/copilot2api/proxy"
)

var version = "dev"

func main() {
	var (
		port        = flag.Int("port", 0, "Server port (env: COPILOT2API_PORT, default: 7777)")
		host        = flag.String("host", "", "Server host (env: COPILOT2API_HOST, default: 127.0.0.1)")
		tokenDir    = flag.String("token-dir", "", "Token storage directory (env: COPILOT2API_TOKEN_DIR, default: ~/.config/copilot2api)")
		showVersion = flag.Bool("version", false, "Show version and exit")
		debug       = flag.Bool("debug", false, "Enable debug logging (env: COPILOT2API_DEBUG)")
	)
	flag.Parse()

	// Apply debug env var
	if !*debug {
		if v := os.Getenv("COPILOT2API_DEBUG"); v != "" {
			if enabled, err := strconv.ParseBool(v); err == nil {
				*debug = enabled
			}
		}
	}

	// Apply env var defaults
	if *host == "" {
		if v := os.Getenv("COPILOT2API_HOST"); v != "" {
			*host = v
		} else {
			*host = "127.0.0.1"
		}
	}
	if *port == 0 {
		if v := os.Getenv("COPILOT2API_PORT"); v != "" {
			if p, err := strconv.Atoi(v); err == nil {
				*port = p
			}
		}
		if *port == 0 {
			*port = 7777
		}
	}
	if *tokenDir == "" {
		if v := os.Getenv("COPILOT2API_TOKEN_DIR"); v != "" {
			*tokenDir = v
		}
	}

	if *showVersion {
		fmt.Printf("copilot2api version %s\n", version)
		os.Exit(0)
	}

	// Set up logging
	logLevel := slog.LevelInfo
	if *debug {
		logLevel = slog.LevelDebug
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: logLevel,
	}))
	slog.SetDefault(logger)

	// Determine token directory
	if *tokenDir == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			slog.Error("failed to get home directory", "error", err)
			os.Exit(1)
		}
		*tokenDir = filepath.Join(homeDir, ".config", "copilot2api")
	}

	// Initialize auth client
	authClient, err := auth.NewClient(*tokenDir)
	if err != nil {
		slog.Error("failed to initialize auth client", "error", err)
		os.Exit(1)
	}

	// Ensure we're authenticated before starting the server. This runs the
	// interactive device flow if needed and verifies a valid Copilot token.
	ctx := context.Background()
	if err := authClient.EnsureAuthenticated(ctx); err != nil {
		slog.Error("authentication failed", "error", err)
		os.Exit(1)
	}

	// Shared HTTP transport for all upstream requests
	transport := upstream.NewTransport()

	// Shared models cache — a single fetch populates both raw JSON (for
	// proxying GET /v1/models) and parsed model info (for capability detection).
	upstreamClient := upstream.NewClient(authClient, transport)
	modelsCache := models.NewCache(upstreamClient, 5*time.Minute)

	// Initialize proxy handler
	proxyHandler := proxy.NewHandler(authClient, transport, modelsCache)

	// Initialize Anthropic handler
	anthropicHandler := anthropic.NewHandler(authClient, transport, modelsCache)

	// Initialize Gemini handler
	geminiHandler := gemini.NewHandler(authClient, transport, modelsCache)

	// Set up routes
	mux := http.NewServeMux()

	// Core routes
	mux.Handle("/v1/chat/completions", proxyHandler)
	mux.Handle("/v1/models", proxyHandler)
	mux.Handle("/v1/embeddings", proxyHandler)
	mux.Handle("/v1/responses", proxyHandler)
	mux.Handle("/v1/messages", anthropicHandler)
	mux.Handle("/v1beta/models", geminiHandler)
	mux.Handle("/v1beta/models/", geminiHandler)
	mux.HandleFunc("/usage", proxyHandler.HandleUsage)

	// AmpCode routes — strip /amp prefix so existing handlers see /v1/...
	mux.Handle("/amp/v1/chat/completions", http.StripPrefix("/amp", proxyHandler))
	mux.Handle("/amp/v1/models", http.StripPrefix("/amp", proxyHandler))
	mux.Handle("/amp/v1/responses", http.StripPrefix("/amp", proxyHandler))
	mux.Handle("/amp/v1/embeddings", http.StripPrefix("/amp", proxyHandler))

	// AmpCode provider-specific routes
	mux.Handle("/api/provider/openai/v1/chat/completions", http.StripPrefix("/api/provider/openai", proxyHandler))
	mux.Handle("/api/provider/openai/v1/responses", http.StripPrefix("/api/provider/openai", proxyHandler))
	mux.Handle("/api/provider/openai/v1/models", http.StripPrefix("/api/provider/openai", proxyHandler))
	mux.Handle("/api/provider/anthropic/v1/messages", http.StripPrefix("/api/provider/anthropic", anthropicHandler))
	mux.Handle("/api/provider/google/v1beta/models", http.StripPrefix("/api/provider/google", geminiHandler))
	mux.Handle("/api/provider/google/v1beta/models/", http.StripPrefix("/api/provider/google", geminiHandler))

	// AmpCode management — reverse proxy to ampcode.com for auth, threads, etc.
	// AI inference stays on Copilot API (routes above); only metadata hits ampcode.com.
	ampBackend, _ := url.Parse("https://ampcode.com")
	ampReverseProxy := newAmpReverseProxy(ampBackend)
	mux.Handle("/api/", ampReverseProxy)
	mux.HandleFunc("/amp/v1/login", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "https://ampcode.com/login", http.StatusFound)
	})
	mux.HandleFunc("/amp/auth/cli-login", func(w http.ResponseWriter, r *http.Request) {
		target := "https://ampcode.com/auth/cli-login"
		if r.URL.RawQuery != "" {
			target += "?" + r.URL.RawQuery
		}
		http.Redirect(w, r, target, http.StatusFound)
	})

	// Pre-warm models cache to avoid cold-cache latency on first request
	go func() {
		slog.Debug("warming models cache")
		modelsCache.Warm(ctx)
		slog.Info("models cache warmed")
	}()

	// Create server
	server := &http.Server{
		Addr:              fmt.Sprintf("%s:%d", *host, *port),
		ReadHeaderTimeout: 10 * time.Second,
		// No ReadTimeout — ReadHeaderTimeout protects against slowloris.
		// ReadTimeout would kill long-lived SSE streaming connections.
		IdleTimeout: 120 * time.Second,
		Handler:     logAllRequests(mux),
	}

	// Start server in goroutine
	serverErr := make(chan error, 1)
	go func() {
		slog.Info("starting server", "host", *host, "port", *port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serverErr <- err
		}
	}()

	// Wait for interrupt signal or server error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-quit:
	case err := <-serverErr:
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}

	slog.Info("shutting down server")

	// Give the server 30 seconds to finish handling existing requests
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		slog.Error("server forced to shutdown", "error", err)
		os.Exit(1)
	}

	slog.Info("server stopped")
}

// newAmpReverseProxy creates a reverse proxy to ampcode.com that forwards the
// client's auth headers. Used for amp CLI management calls (getUserInfo,
// threads, telemetry, etc.) — no AI credits are consumed.
func newAmpReverseProxy(target *url.URL) *httputil.ReverseProxy {
	return &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			slog.Debug("amp proxy", "method", req.Method, "path", req.URL.Path, "query", req.URL.RawQuery)
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host
		},
	}
}

func logAllRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slog.Debug("incoming request", "method", r.Method, "path", r.URL.Path, "query", r.URL.RawQuery)
		next.ServeHTTP(w, r)
	})
}
