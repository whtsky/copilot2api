package auth

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	GitHubClientID      = "Iv1.b507a08c87ecfe98"
	GitHubDeviceCodeURL = "https://github.com/login/device/code"
	GitHubTokenURL      = "https://github.com/login/oauth/access_token"
	GitHubScope         = "read:user"
)

type DeviceCodeResponse struct {
	DeviceCode      string `json:"device_code"`
	UserCode        string `json:"user_code"`
	VerificationURI string `json:"verification_uri"`
	ExpiresIn       int    `json:"expires_in"`
	Interval        int    `json:"interval"`
}

type AccessTokenResponse struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	Error       string `json:"error"`
}

// InitiateDeviceFlow starts the GitHub Device Flow OAuth process
func InitiateDeviceFlow() (*DeviceCodeResponse, error) {
	data := url.Values{
		"client_id": {GitHubClientID},
		"scope":     {GitHubScope},
	}

	req, err := http.NewRequest("POST", GitHubDeviceCodeURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("failed to create device code request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := sharedHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to request device code: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("device code request failed with status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	var deviceResp DeviceCodeResponse
	if err := json.Unmarshal(body, &deviceResp); err != nil {
		return nil, fmt.Errorf("failed to parse device code response: %w", err)
	}

	return &deviceResp, nil
}

// PollForAccessToken polls GitHub for the access token after user authorization
func PollForAccessToken(deviceCode string, interval int, timeout time.Duration) (string, error) {
	pollInterval := time.Duration(interval) * time.Second
	if pollInterval <= 0 {
		pollInterval = 5 * time.Second
	}
	timeoutTimer := time.NewTimer(timeout)
	pollTicker := time.NewTicker(pollInterval)
	defer timeoutTimer.Stop()
	defer pollTicker.Stop()

	for {
		select {
		case <-timeoutTimer.C:
			return "", fmt.Errorf("polling timeout exceeded")
		case <-pollTicker.C:
			token, err := checkAccessToken(deviceCode)
			if err != nil {
				// Continue polling on certain errors
				if strings.Contains(err.Error(), "authorization_pending") {
					continue
				}
				if strings.Contains(err.Error(), "slow_down") {
					// GitHub requests us to slow down, increase poll interval by 5 seconds
					pollInterval += 5 * time.Second
					pollTicker.Stop()
					pollTicker = time.NewTicker(pollInterval)
					continue
				}
				return "", err
			}
			if token != "" {
				return token, nil
			}
		}
	}
}

func checkAccessToken(deviceCode string) (string, error) {
	data := url.Values{
		"client_id":   {GitHubClientID},
		"device_code": {deviceCode},
		"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
	}

	req, err := http.NewRequest("POST", GitHubTokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return "", fmt.Errorf("failed to create access token request: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := sharedHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to check access token: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	var tokenResp AccessTokenResponse
	if err := json.Unmarshal(body, &tokenResp); err != nil {
		return "", fmt.Errorf("failed to parse token response: %w", err)
	}

	if tokenResp.Error != "" {
		return "", fmt.Errorf("token error: %s", tokenResp.Error)
	}

	return tokenResp.AccessToken, nil
}