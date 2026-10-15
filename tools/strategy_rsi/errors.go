//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//

package main

import (
	"context"
	"errors"
	"math/rand"
	"net"
	"os"
	"regexp"
	"strings"
	"time"
)

// failClass is how a failed call is handled (plan §6.4): quota/network -> retry
// (after reconcile), timeout/permanent -> skip or abandon the batch update.
type failClass int

const (
	failTimeout failClass = iota
	failQuota
	failNetwork
	failPermanent
)

func (f failClass) String() string {
	switch f {
	case failTimeout:
		return "timeout"
	case failQuota:
		return "quota"
	case failNetwork:
		return "network"
	default:
		return "permanent"
	}
}

// quotaMarkers / networkMarkers are matched (case-insensitive) against *transport error*
// text (never against a model answer's body -- see answerErrorPhrases).
var quotaMarkers = []string{
	"rate limit", "429", "529", "quota", "用量上限", "额度", "速率限制",
	"high demand", "overloaded", "too many requests", "token plan",
}

var networkMarkers = []string{
	"connection reset", "connection refused", "no such host", "eof", "broken pipe",
	"i/o timeout", "tls", "unexpected eof", "network is unreachable",
}

var timeoutMarkers = []string{
	"timed out", "timeout", "deadline exceeded", "context deadline",
}

// answerErrorPhrases are the ONLY signals that a returned ANSWER is really a decorated
// provider error. They must be multi-word / explicit: matching short substrings against
// arbitrary model text produces catastrophic false positives -- "Geoffrey Chaucer"
// contains "eof", and any date/number can contain "429"/"529".
var answerErrorPhrases = []string{
	"**error**", "api error", "rate limit exceeded", "用量上限", "额度不足", "quota exceeded",
	"connection reset by peer", "connection refused", "no such host",
	"context deadline exceeded", "request timed out", "i/o timeout", "too many requests",
}

var statusCodeRE = regexp.MustCompile(`\b(429|529)\b`)

// classify inspects a TRANSPORT error and returns its class.
func classify(err error) failClass {
	if err == nil {
		return failPermanent
	}
	// Explicit Go timeout signals win over text heuristics.
	if errors.Is(err, context.DeadlineExceeded) {
		return failTimeout
	}
	if os.IsTimeout(err) {
		return failTimeout
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return failTimeout
	}
	var hs *httpStatusError
	if errors.As(err, &hs) {
		switch {
		case hs.Status == 429:
			return failQuota
		case hs.Status >= 500:
			return failNetwork
		default:
			return failPermanent
		}
	}
	msg := strings.ToLower(err.Error())
	if containsAny(msg, timeoutMarkers) {
		return failTimeout
	}
	if containsAny(msg, quotaMarkers) {
		return failQuota
	}
	if containsAny(msg, networkMarkers) {
		return failNetwork
	}
	return failPermanent
}

// answerLooksLikeError reports whether a returned answer is really a decorated provider
// error rather than a genuine model answer. Conservative by construction: an ordinary
// answer must never be misread as a failure (that would retry forever).
func answerLooksLikeError(answer string) bool {
	m := strings.ToLower(answer)
	if containsAny(m, answerErrorPhrases) {
		return true
	}
	if statusCodeRE.MatchString(m) && containsAny(m, []string{"error", "limit", "rate", "quota"}) {
		return true
	}
	return false
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, strings.ToLower(sub)) {
			return true
		}
	}
	return false
}

// retryPolicy is the exponential backoff used for quota/network failures (§6.4).
type retryPolicy struct {
	initial time.Duration
	max     time.Duration
	jitter  float64
}

func (p retryPolicy) delay(attempt int) time.Duration {
	if p.initial <= 0 {
		p.initial = time.Second
	}
	if p.max <= 0 {
		p.max = 60 * time.Second
	}
	d := p.initial
	for i := 0; i < attempt && d < p.max; i++ {
		d *= 2
	}
	if d > p.max {
		d = p.max
	}
	if p.jitter > 0 {
		// jitter in [1-jitter, 1+jitter]
		factor := 1 + (rand.Float64()*2-1)*p.jitter
		d = time.Duration(float64(d) * factor)
	}
	if d < 0 {
		d = 0
	}
	return d
}

func (p retryPolicy) sleep(ctx context.Context, attempt int) error {
	t := time.NewTimer(p.delay(attempt))
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// callWithRetry repeatedly runs fn until it succeeds. A timeout or permanent failure is
// returned immediately (the caller decides skip/abandon); quota/network failures are
// retried endlessly after an exponential backoff, as required by the plan.
//
// NOTE: the caller must make fn side-effect-safe across retries -- resending a chat turn
// duplicates it (the server persists the user turn before calling the model), so fn is
// expected to reconcile first (see rsi.go).
func callWithRetry(ctx context.Context, p retryPolicy, fn func(attempt int) error) error {
	for attempt := 0; ; attempt++ {
		err := fn(attempt)
		if err == nil {
			return nil
		}
		switch classify(err) {
		case failQuota, failNetwork:
			if serr := p.sleep(ctx, attempt); serr != nil {
				return serr
			}
			continue
		default:
			return err
		}
	}
}
