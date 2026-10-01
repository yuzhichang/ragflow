package main

// The HTTP layer and the two chat entry points (port of JsonHttpClient,
// create_session, ask_ragflow, ask_judge / ask_judge_with_retry and the
// answer-text extraction). One *http.Client is shared: unlike
// requests.Session it is safe for concurrent use, so the Python client's
// clone() has no Go counterpart.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

const (
	defaultTimeoutSeconds = 2400
	defaultMaxRetries     = 0
	defaultBackoffSeconds = 2.0
)

// GRADER_TEMPLATE is THE judge prompt, used for every dataset. Provenance:
// copied byte-for-byte from GRADER_TEMPLATE in the BrowseComp-Plus scoring
// script (scripts_evaluation/evaluate_run.py), because a submission is only
// comparable with the published numbers when the same judge prompt produced
// the verdicts. The verdict is BINARY (correct: yes/no) and the judge reports
// the confidence found in the answer (0-100, 100 when the answer states none).
const graderTemplate = `Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.`

// judgeTransientMarkers: a judge call that comes back as a provider error is
// retried before the row is recorded as unjudged. The backend decorates
// provider failures into the ANSWER TEXT ("**ERROR**: minimax API error: ..."),
// so they never surface as a transport error - retrying has to inspect the
// text. Without this, one rate-limit burst silently produced rows with no
// verdict at all (and an unjudged row counts as incorrect).
var judgeTransientMarkers = []string{
	"速率限制", "用量上限", "rate limit", "429", "529", "status 5",
	"timed out", "timeout", "connection reset", "connection refused",
}

// benchmarkError ports BenchmarkError.
type benchmarkError struct{ msg string }

func (e *benchmarkError) Error() string { return e.msg }

func errf(format string, args ...any) error {
	return &benchmarkError{msg: fmt.Sprintf(format, args...)}
}

// httpClient ports JsonHttpClient.
type httpClient struct {
	baseURL    string
	apiKey     string
	maxRetries int
	backoffSec float64
	underlying *http.Client
}

func newClient(cfg map[string]any) *httpClient {
	apiKey := strAny(cfg, "ragflow_api_key", "")
	if apiKey == "" {
		apiKey = os.Getenv("RAGFLOW_API_KEY")
	}
	if apiKey == "" {
		// Mirror the Python ValueError; the caller reports it.
		panic("ragflow_api_key is required in the config (or set the RAGFLOW_API_KEY environment variable)")
	}
	backend, _ := cfg["backend"].(map[string]any)
	baseURL := strAny(backend, "base_url", "")
	if baseURL == "" {
		baseURL = fmt.Sprintf("%s://%s:%s",
			strAny(backend, "scheme", "http"), strAny(backend, "host", "127.0.0.1"), strAny(backend, "port", "80"))
	}
	return &httpClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		maxRetries: defaultMaxRetries,
		backoffSec: defaultBackoffSeconds,
		underlying: &http.Client{Timeout: defaultTimeoutSeconds * time.Second},
	}
}

func (c *httpClient) post(path string, body map[string]any) (any, error) {
	return c.request(path, body, false)
}

func (c *httpClient) postEventstream(path string, body map[string]any) (any, error) {
	stream := map[string]any{"stream": true}
	for k, v := range body {
		stream[k] = v
	}
	return c.request(path, stream, true)
}

func (c *httpClient) request(path string, body map[string]any, eventstream bool) (any, error) {
	target := c.baseURL + path
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	accept := "application/json"
	if eventstream {
		accept = "text/event-stream"
	}
	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		backoff := time.Duration(c.backoffSec*float64(int64(1)<<uint(attempt))) * time.Second
		req, err := http.NewRequest("POST", target, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Accept", accept)
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")
		resp, err := c.underlying.Do(req)
		if err != nil {
			lastErr = err
			if attempt < c.maxRetries && shouldRetry(lastErr) {
				time.Sleep(backoff)
				continue
			}
			break
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if eventstream && resp.StatusCode < 400 && strings.Contains(resp.Header.Get("Content-Type"), "text/event-stream") {
			return collectEventstreamAnswer(string(data)), nil
		}
		decoded := decodeResponse(data)
		if resp.StatusCode >= 400 {
			lastErr = errf("HTTP %d from %s: %s", resp.StatusCode, target, compact(decoded))
		} else if m, ok := decoded.(map[string]any); ok {
			if code := asInt(m["code"]); code != nil && *code != 0 {
				lastErr = errf("RAGFlow code %d from %s: %s", *code, target, strAny(m, "message", compact(m)))
			} else if inner, ok := m["data"]; ok {
				return inner, nil
			} else {
				return decoded, nil
			}
		} else {
			return decoded, nil
		}
		if lastErr != nil && (attempt >= c.maxRetries || !shouldRetry(lastErr)) {
			break
		}
		if lastErr != nil {
			time.Sleep(backoff)
		}
	}
	return nil, lastErr
}

func decodeResponse(data []byte) any {
	var payload any
	if err := json.Unmarshal(data, &payload); err != nil {
		return map[string]any{"raw": string(data)}
	}
	return payload
}

// shouldRetry ports _should_retry: transport errors and 5xx/timeout bubbles
// are worth another attempt; a 4xx is not.
func shouldRetry(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "HTTP 5") || strings.Contains(msg, "timed out") || strings.Contains(msg, "Timeout")
}

// collectEventstreamAnswer ports _collect_eventstream_answer: assemble the
// answer from the SSE stream's data lines.
func collectEventstreamAnswer(raw string) string {
	answer := ""
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimRight(line, "\r")
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(line[5:])
		if data == "" || data == "[DONE]" {
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal([]byte(data), &payload); err != nil {
			continue
		}
		if code := asInt(payload["code"]); code != nil && *code != 0 {
			// Python raises out of the collection loop; the error rides in the
			// answer text the caller already handles (**ERROR**).
			continue
		}
		answer = appendAnswerPart(answer, payload)
	}
	return answer
}

func appendAnswerPart(answer string, payload map[string]any) string {
	if data, ok := payload["data"].(map[string]any); ok {
		if s, ok := data["answer"].(string); ok {
			return answer + s
		}
		if _, has := data["reference"]; has {
			return answer
		}
	}
	if s, ok := payload["answer"].(string); ok {
		return answer + s
	}
	if choices, ok := payload["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			if delta, ok := choice["delta"].(map[string]any); ok {
				if s, ok := delta["content"].(string); ok {
					return answer + s
				}
			}
		}
	}
	return answer
}

// extractAnswerText ports extract_answer_text: dig the answer text out of the
// many payload shapes the chat API and the agent modes return.
func extractAnswerText(payload any) string {
	switch p := payload.(type) {
	case nil:
		return ""
	case string:
		return p
	case map[string]any:
		for _, key := range []string{"answer", "content", "text", "message"} {
			switch value := p[key].(type) {
			case string:
				return value
			case map[string]any:
				if nested := extractAnswerText(value); nested != "" {
					return nested
				}
			}
		}
		if data, ok := p["data"]; ok {
			return extractAnswerText(data)
		}
		if choices, ok := p["choices"].([]any); ok && len(choices) > 0 {
			return extractAnswerText(choices[0])
		}
	}
	return ""
}

// extractSessionID ports _extract_session_id.
func extractSessionID(payload any) string {
	m, ok := payload.(map[string]any)
	if !ok {
		return ""
	}
	for _, key := range []string{"session_id", "conversation_id", "id"} {
		if s, ok := m[key].(string); ok && s != "" {
			return s
		}
	}
	if data, ok := m["data"].(map[string]any); ok {
		return extractSessionID(data)
	}
	return ""
}

// createSession ports create_session.
func createSession(c *httpClient, chatID string) (string, error) {
	payload, err := c.post("/api/v1/chats/"+url.PathEscape(chatID)+"/sessions",
		map[string]any{"name": fmt.Sprintf("qa-benchmark-%s", time.Now().Format("20060102-150405"))})
	if err != nil {
		return "", err
	}
	sessionID := extractSessionID(payload)
	if sessionID == "" {
		return "", errf("Could not create chat session: %s", compact(payload))
	}
	return sessionID, nil
}

// askRagflow ports ask_ragflow: one chat completion against the answer chat.
func askRagflow(c *httpClient, cfg map[string]any, question, sessionID string) (any, error) {
	chatCfg, _ := cfg["ragflow_chat"].(map[string]any)
	body := map[string]any{
		"chat_id":  strAny(cfg, "chat_id", ""),
		"question": question,
		"stream":   boolOf(chatCfg["stream"]),
	}
	if sessionID != "" {
		body["session_id"] = sessionID
	}
	if v := strAny(chatCfg, "llm_id", ""); v != "" {
		body["llm_id"] = v
	}
	for _, key := range []string{"quote", "refine_multiturn", "temperature", "top_p", "max_tokens"} {
		if v, ok := chatCfg[key]; ok {
			body[key] = v
		}
	}
	// agent_mode (e.g. "smart-reasoning") drives the agentic ReAct loop; in
	// that mode the chat API reads `reasoning` as part of the agentic request,
	// so it is pinned to 2 unless the config overrides it.
	body["reasoning"] = asIntDefault(chatCfg["reasoning"], 2)
	if v := strAny(chatCfg, "agent_mode", ""); v != "" {
		body["agent_mode"] = v
	}
	if boolOf(body["stream"]) {
		return c.postEventstream("/api/v1/chat/completions", body)
	}
	return c.post("/api/v1/chat/completions", body)
}

func boolOf(v any) bool {
	b, _ := v.(bool)
	return b
}

// renderJudgePrompt ports render_judge_prompt. The template is not
// configurable: it is the BrowseComp-Plus scoring script's reference prompt,
// so letting a config rewrite it would silently break comparability with
// published numbers.
var thinkRE = regexp.MustCompile(`(?s)<think>.*?</think>`)

func renderJudgePrompt(row map[string]any) string {
	return strings.NewReplacer(
		"{question}", strAny(row, "question", ""),
		"{correct_answer}", strAny(row, "gold_answer", ""),
		"{response}", thinkRE.ReplaceAllString(strAny(row, "ragflow_answer", ""), ""),
	).Replace(graderTemplate)
}

// judgeLLMIDs ports judge_llm_ids: the llm_ids to try when judging, in order.
// The judge chat is a PLAIN chat (no agent_mode), so it has NO failover chain
// - it is pinned to the single instance it is bound to, and when that
// instance's plan is exhausted every judgement fails. Asking for another of
// the tenant's instances explicitly is what lets judging survive a
// single-instance quota wall.
func judgeLLMIDs(cfg map[string]any) []string {
	if list, ok := cfg["judge_llm_ids"].([]any); ok {
		var ids []string
		for _, id := range list {
			if s := strings.TrimSpace(fmt.Sprintf("%v", id)); s != "" {
				ids = append(ids, s)
			}
		}
		if len(ids) > 0 {
			return ids
		}
	}
	if single := strings.TrimSpace(strAny(cfg, "judge_llm_id", "")); single != "" {
		return []string{single}
	}
	return []string{""}
}

// isTransientJudgeAnswer ports _is_transient_judge_answer: True when a judge
// reply is a provider failure rather than a verdict. A retry can only help
// when the failure is the transient kind, hence the marker test on top of the
// ERROR prefix.
func isTransientJudgeAnswer(answer string) bool {
	if !strings.HasPrefix(strings.TrimLeft(answer, " \t\n"), "**ERROR**") {
		return false
	}
	for _, marker := range judgeTransientMarkers {
		if strings.Contains(answer, marker) {
			return true
		}
	}
	return false
}

var judgeMaxRetries = 3
var judgeRetryBackoff = []time.Duration{2 * time.Second, 5 * time.Second, 10 * time.Second}

// askJudgeWithRetry ports ask_judge_with_retry: ask_judge with backoff
// retries, then with the next configured instance. Two failure classes,
// handled in order of cost: a transient burst (retried on the SAME instance
// with backoff) and an exhausted instance (retried on the NEXT llm_id).
func askJudgeWithRetry(c *httpClient, cfg map[string]any, row map[string]any) string {
	answer := ""
	for _, llmID := range judgeLLMIDs(cfg) {
		payload, judgeErr := askJudge(c, cfg, row, llmID)
		if judgeErr != nil {
			answer = "**ERROR**: " + judgeErr.Error()
		} else {
			answer = extractAnswerText(payload)
		}
		for attempt := 0; attempt < judgeMaxRetries; attempt++ {
			if !isTransientJudgeAnswer(answer) {
				return answer
			}
			time.Sleep(judgeRetryBackoff[min(attempt, len(judgeRetryBackoff)-1)])
			payload, judgeErr := askJudge(c, cfg, row, llmID)
			if judgeErr != nil {
				answer = "**ERROR**: " + judgeErr.Error()
			} else {
				answer = extractAnswerText(payload)
			}
		}
		// Retries on this instance are exhausted; fall through to the next id
		// (when the remaining failure is a plan wall, not a burst).
	}
	return answer
}

// askJudge ports ask_judge.
func askJudge(c *httpClient, cfg map[string]any, row map[string]any, llmID string) (any, error) {
	body := map[string]any{
		"chat_id":  strAny(cfg, "judge_chat_id", ""),
		"question": renderJudgePrompt(row),
		"stream":   boolOf(cfg["judge_stream"]),
	}
	if llmID != "" {
		body["llm_id"] = llmID
	}
	if boolOf(body["stream"]) {
		return c.postEventstream("/api/v1/chat/completions", body)
	}
	return c.post("/api/v1/chat/completions", body)
}
