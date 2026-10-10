//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package config

import (
	"fmt"

	"github.com/spf13/viper"
)

type AuthenticationConfig struct {
	DisablePasswordLogin bool `mapstructure:"disable_password_login"`
	EnableRegister       bool `mapstructure:"enable_register"`
}

type APIServerConfig struct {
	MCP      MCPConfig
	Codex    CodexConfig
	Host     string `mapstructure:"host"`
	HTTPPort int    `mapstructure:"http_port"`
	// TrustedProxies lists the IPs / CIDRs whose X-Forwarded-For and
	// X-Real-IP headers are trusted when resolving the client address.
	// nil means "not configured" and resolves to common.DefaultTrustedProxies
	// (loopback, i.e. the nginx bundled in the ragflow image). An explicit
	// list replaces that default rather than extending it, and an empty
	// list trusts no proxy at all.
	TrustedProxies []string `mapstructure:"trusted_proxies"`

	Authentication AuthenticationConfig `mapstructure:"authentication"`
}

func (c *Config) ParseAPIServerConfig(v *viper.Viper) error {
	if err := c.parseMCPConfig(v); err != nil {
		return err
	}
	if err := c.parseCodexConfig(v); err != nil {
		return err
	}

	// Default Admin config
	c.apiServer.Host = "localhost"
	c.apiServer.HTTPPort = 9380

	if !v.IsSet("ragflow") {
		return c.deriveCodexMCPBase()
	}
	sub := v.Sub("ragflow")
	if sub == nil {
		return c.deriveCodexMCPBase()
	}

	if sub.IsSet("host") {
		c.apiServer.Host = sub.GetString("host")
	}

	if sub.IsSet("http_port") {
		c.apiServer.HTTPPort = sub.GetInt("http_port")
	}

	if sub.IsSet("trusted_proxies") {
		proxies := sub.GetStringSlice("trusted_proxies")
		if proxies == nil {
			proxies = []string{}
		}
		c.apiServer.TrustedProxies = proxies
	}

	c.parseAuthenticationConfig(v)

	return c.deriveCodexMCPBase()
}

// deriveCodexMCPBase fills codex.mcp_public_base from the API server's own address when
// the operator did not set it explicitly, so a simple deployment needs no extra config.
// A wildcard bind address is not reachable, so it maps to loopback.
func (c *Config) deriveCodexMCPBase() error {
	if c.apiServer.Codex.Endpoint == "" || c.apiServer.Codex.MCPPublicBase != "" {
		return nil
	}
	host := c.apiServer.Host
	switch host {
	case "", "0.0.0.0", "::", "localhost":
		host = "127.0.0.1"
	}
	c.apiServer.Codex.MCPPublicBase = fmt.Sprintf("http://%s:%d", host, c.apiServer.HTTPPort)
	return nil
}

func (c *Config) parseAuthenticationConfig(v *viper.Viper) {
	apiServerConfig := &c.apiServer
	apiServerConfig.Authentication.DisablePasswordLogin = false
	apiServerConfig.Authentication.EnableRegister = true

	if !v.IsSet("authentication") {
		return
	}
	sub := v.Sub("authentication")
	if sub == nil {
		return
	}

	if sub.IsSet("disable_password_login") {
		apiServerConfig.Authentication.DisablePasswordLogin = sub.GetBool("disable_password_login")
	}

	if sub.IsSet("enable_register") {
		apiServerConfig.Authentication.EnableRegister = sub.GetBool("enable_register")
	}
}

func (c *Config) DisablePasswordLogin() bool {
	if c.environments.DisablePasswordLogin != nil {
		return *c.environments.DisablePasswordLogin
	}
	return c.apiServer.Authentication.DisablePasswordLogin
}

func (c *Config) EnableRegister() bool {
	if c.environments.EnableRegister != nil {
		return *c.environments.EnableRegister
	}
	return c.apiServer.Authentication.EnableRegister
}

func (c *Config) GetAPIServerConfig() APIServerConfig {
	return c.apiServer
}
