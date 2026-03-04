package auth

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type TokenStorage struct {
	TokenDir string
}

type StoredCredentials struct {
	GitHubToken  string       `json:"github_token"`
	CopilotToken *CopilotToken `json:"copilot_token,omitempty"`
}

// NewTokenStorage creates a new token storage instance
func NewTokenStorage(tokenDir string) (*TokenStorage, error) {
	// Ensure directory exists
	if err := os.MkdirAll(tokenDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create token directory: %w", err)
	}
	if err := os.Chmod(tokenDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to set token directory permissions: %w", err)
	}

	return &TokenStorage{TokenDir: tokenDir}, nil
}

func (s *TokenStorage) credentialsPath() string {
	return filepath.Join(s.TokenDir, "credentials.json")
}

// LoadCredentials loads stored credentials from disk
func (s *TokenStorage) LoadCredentials() (*StoredCredentials, error) {
	path := s.credentialsPath()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &StoredCredentials{}, nil // Return empty credentials if file doesn't exist
		}
		return nil, fmt.Errorf("failed to read credentials file: %w", err)
	}

	var creds StoredCredentials
	if err := json.Unmarshal(data, &creds); err != nil {
		return nil, fmt.Errorf("failed to parse credentials: %w", err)
	}

	return &creds, nil
}

// SaveCredentials saves credentials to disk atomically using temp file + rename.
func (s *TokenStorage) SaveCredentials(creds *StoredCredentials) error {
	path := s.credentialsPath()

	data, err := json.MarshalIndent(creds, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal credentials: %w", err)
	}

	// Write to a temp file in the same directory, then rename for atomicity.
	tmpPath := path + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0600); err != nil {
		return fmt.Errorf("failed to write credentials temp file: %w", err)
	}
	if err := os.Chmod(tmpPath, 0600); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to set credentials temp file permissions: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("failed to rename credentials file: %w", err)
	}

	return nil
}