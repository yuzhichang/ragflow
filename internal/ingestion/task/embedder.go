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

package task

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"ragflow/internal/dao"
	"ragflow/internal/entity/models"
	componentpkg "ragflow/internal/ingestion/component"
	"ragflow/internal/service"
)

type embedder struct {
	model *models.EmbeddingModel
}

func (e *embedder) MaxTokens() int {
	if e == nil || e.model == nil {
		return 0
	}
	return e.model.MaxTokens
}

func (e *embedder) BatchSize() int {
	if e == nil || e.model == nil {
		return models.DefaultEmbeddingBatchSize
	}
	return e.model.ResolveBatchSize()
}

func (e *embedder) Encode(ctx context.Context, texts []string) ([]componentpkg.EmbeddingResult, error) {
	if e.model.ModelDriver == nil {
		return nil, fmt.Errorf("embedder: embedding model driver is nil for model %v", e.model.ModelName)
	}
	config := &models.EmbeddingConfig{Dimension: 0}
	req := models.EmbedRequest{Texts: texts}

	var (
		embeds []models.EmbeddingData
		err    error
	)
	// The cooldown is scoped to this provider instance: workers embedding
	// through the same credentials+endpoint back off together, everything else
	// keeps running.
	quota := e.quotaKey()
	for attempt := 1; attempt <= maxEncodeAttempts; attempt++ {
		// Sit out any cooldown that another worker's 429 started. The provider
		// quota is shared, so a worker that keeps firing while the window is
		// exhausted only earns more 429s - and the whole point of the cooldown
		// is that one worker's rejection damps all of them.
		if cerr := waitOutCooldown(ctx, quota); cerr != nil {
			return nil, cerr
		}
		embeds, err = e.model.ModelDriver.Embed(ctx, e.model.ModelName, req, e.model.APIConfig, config, nil)
		if err == nil {
			break
		}
		if !isRateLimitErr(err) {
			return nil, err
		}
		if attempt == maxEncodeAttempts {
			return nil, err
		}
		wait := encodeBackoff(attempt, err)
		startCooldown(quota, wait)
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(wait):
		}
	}
	if err != nil {
		return nil, err
	}

	vecs := make([]componentpkg.EmbeddingResult, len(embeds))
	for i, v := range embeds {
		vecs[i] = componentpkg.EmbeddingResult{Vector: v.Embedding, TokenCount: v.TokenCount}
	}
	return vecs, nil
}

// Retry policy for rate-limited embedding calls.
//
// A TPM-limited provider (SiliconFlow bge-m3 answers "429 ... TPM limit
// reached") refills its quota on a one-minute window, so a retry schedule that
// exhausts itself inside a single window cannot succeed: the 2s/4s/8s/16s of the
// original implementation always ran out while the quota was still spent, and
// every concurrent worker kept hammering the same exhausted quota, turning a
// transient throttle into a FAILED document whose chunks had already been
// deleted. The schedule below spans roughly two and a half windows, and the
// shared cooldown makes the workers back off together instead of independently.
//
// These are variables rather than constants so tests can shrink them.
var (
	maxEncodeAttempts   = 6
	encodeBackoffBase   = 5 * time.Second
	encodeBackoffMax    = 120 * time.Second
	encodeBackoffJitter = 0.2
)

// providerQuotaKey identifies the thing a TPM quota actually belongs to: one
// provider instance (endpoint + credentials + model). See quotaKey.
type providerQuotaKey string

// rateLimitCooldowns parks callers per provider instance - and only per provider
// instance.
//
// The unit is the provider instance, not the process: a TPM quota is owned by
// the credentials, so a worker that got throttled on provider A must not stall
// ingestion that talks to provider B, and must not stall a different dataset
// embedding through a different model. It cannot be the embedder value either:
// newEmbedderResolver builds a fresh embedder for every tokenizer invocation
// (one per document), so state hung off the value would never be shared by the
// workers that are actually competing for the same quota. A keyed registry
// gives exactly the required scope - all workers embedding through the same
// provider instance back off together, everyone else is untouched.
var rateLimitCooldowns = struct {
	sync.Mutex
	until map[providerQuotaKey]time.Time
}{until: make(map[providerQuotaKey]time.Time)}

// startCooldown parks callers sharing `key` for at least d, extending an active
// cooldown but never shortening it.
func startCooldown(key providerQuotaKey, d time.Duration) {
	if d <= 0 {
		return
	}
	deadline := time.Now().Add(d)
	rateLimitCooldowns.Lock()
	defer rateLimitCooldowns.Unlock()
	if existing, ok := rateLimitCooldowns.until[key]; !ok || deadline.After(existing) {
		rateLimitCooldowns.until[key] = deadline
	}
	// Keys are bounded by the number of configured provider instances, so this
	// is housekeeping rather than a leak fix.
	if len(rateLimitCooldowns.until) > 64 {
		for k, until := range rateLimitCooldowns.until {
			if !until.After(time.Now()) {
				delete(rateLimitCooldowns.until, k)
			}
		}
	}
}

// cooldownRemaining reports how long the cooldown for `key` still has to run.
func cooldownRemaining(key providerQuotaKey) time.Duration {
	rateLimitCooldowns.Lock()
	defer rateLimitCooldowns.Unlock()
	return time.Until(rateLimitCooldowns.until[key])
}

// waitOutCooldown blocks until the cooldown for `key` expires, honoring ctx.
func waitOutCooldown(ctx context.Context, key providerQuotaKey) error {
	remaining := cooldownRemaining(key)
	if remaining <= 0 {
		return nil
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// quotaKey derives the provider-instance key the cooldown is scoped to. The API
// key decides who owns the quota, so it is part of the key but hashed: the key
// must never carry a plaintext secret.
func (e *embedder) quotaKey() providerQuotaKey {
	if e == nil || e.model == nil {
		return providerQuotaKey("embedder:unknown")
	}
	var baseURL, region, apiKey string
	if cfg := e.model.APIConfig; cfg != nil {
		baseURL = derefString(cfg.BaseURL)
		region = derefString(cfg.Region)
		apiKey = derefString(cfg.ApiKey)
	}
	sum := sha256.Sum256([]byte(apiKey))
	return providerQuotaKey(fmt.Sprintf("%s|%s|%s|%x", baseURL, region, derefString(e.model.ModelName), sum[:8]))
}

// derefString safely dereferences an optional string.
func derefString(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}

// encodeBackoff returns how long to wait before retry `attempt` (1-based) of a
// call that failed with `err`. A provider-supplied Retry-After wins; otherwise
// the delay grows exponentially with jitter so concurrent workers spread out
// instead of retrying in lockstep.
func encodeBackoff(attempt int, err error) time.Duration {
	if hint, ok := retryAfterFromError(err); ok {
		if hint > encodeBackoffMax {
			return encodeBackoffMax
		}
		return hint
	}
	delay := float64(encodeBackoffBase) * math.Pow(2, float64(attempt-1))
	if delay > float64(encodeBackoffMax) {
		delay = float64(encodeBackoffMax)
	}
	if encodeBackoffJitter > 0 {
		spread := delay * encodeBackoffJitter
		delay += (rand.Float64()*2 - 1) * spread
	}
	if delay < float64(time.Second) {
		delay = float64(time.Second)
	}
	return time.Duration(delay)
}

// retryAfterPattern matches the seconds form of Retry-After surfaced by the
// model drivers ("retry-after: 30", "retry_after=30", "retry after 30 seconds").
var retryAfterPattern = regexp.MustCompile(`(?i)retry[-_ ]?after["'\s:=]+(\d+(\.\d+)?)`)

// retryAfterFromError extracts a Retry-After hint in seconds from an error
// message, if the driver carried one.
func retryAfterFromError(err error) (time.Duration, bool) {
	if err == nil {
		return 0, false
	}
	m := retryAfterPattern.FindStringSubmatch(err.Error())
	if m == nil {
		return 0, false
	}
	seconds, perr := strconv.ParseFloat(m[1], 64)
	if perr != nil || seconds <= 0 {
		return 0, false
	}
	return time.Duration(seconds * float64(time.Second)), true
}

// isRateLimitErr reports whether err is an HTTP 429 / rate-limit / quota error
// (e.g. SiliconFlow "TPM limit reached"). It matches on the status code and the
// common rate-limit wording, which is stable across providers.
func isRateLimitErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "429") ||
		strings.Contains(msg, "too many requests") ||
		strings.Contains(msg, "rate limit") ||
		strings.Contains(msg, "rate limiting") ||
		strings.Contains(msg, "tpm limit")
}

// newEmbedderResolver builds the production embedder resolver used by the
// Tokenizer component. It always resolves the embedder from the dataset's
// configured embd_id (looked up by kbID) and returns that embd_id alongside the
// embedder, so the Tokenizer can key its per-chunk cache on the dataset-bound
// model. If the dataset has no embd_id configured, it returns a nil embedder and
// an empty embd_id (no embedding). Kept as a constructor over injectable deps so
// the resolution logic stays unit-testable without a live model provider / DB.
func newEmbedderResolver(
	getKBEmbdID func(ctx context.Context, kbID string) (string, error),
	getEmbeddingModel func(ctx context.Context, tenantID, embdID string) (*models.EmbeddingModel, error),
) componentpkg.EmbedderResolver {
	// The resolver derives the embedding model exclusively from the
	// knowledgebase's configured embd_id — never from any DSL-supplied
	// identifier. It returns embdID alongside the embedder so the tokenizer can
	// key its per-chunk cache on the dataset-bound model: when a KB's embedding
	// model changes, embdID changes, the cache key changes, and no stale vector
	// is ever served.
	return func(ctx context.Context, tenantID, kbID string) (componentpkg.Embedder, string, error) {
		embdID, err := getKBEmbdID(ctx, kbID)
		if err != nil {
			return nil, "", fmt.Errorf("embedder: resolve kb embd_id for kb_id=%s: %w", kbID, err)
		}
		embdID = strings.TrimSpace(embdID)
		if embdID == "" {
			return nil, "", nil
		}
		model, err := getEmbeddingModel(ctx, tenantID, embdID)
		if err != nil {
			return nil, "", err
		}
		if model == nil {
			return nil, "", fmt.Errorf("embedder: resolved embedding model is nil for embd_id=%s", embdID)
		}
		return &embedder{model: model}, embdID, nil
	}
}

// init wires the production embedder resolver into the component package. The
// component package must not import internal/service (dependency direction),
// so the concrete resolver is injected here - the task package is the
// composition root for ingestion runs.
func init() {
	componentpkg.DefaultEmbedderResolver = newEmbedderResolver(
		func(ctx context.Context, kbID string) (string, error) {
			kb, err := dao.NewKnowledgebaseDAO().GetByID(ctx, dao.DB, kbID)
			if err != nil {
				return "", err
			}
			if kb == nil {
				return "", nil
			}
			return kb.EmbdID, nil
		},
		service.NewModelProviderService().GetEmbeddingModel,
	)
}
