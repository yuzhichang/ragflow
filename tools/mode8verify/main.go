// Command mode8verify checks, from the Codex side, that a RAGFlow mode-8 turn really exercised
// the capabilities mode 8 exposes. It drives the chat API and then reads the Codex app-server's
// own rollouts, because the API answer alone cannot tell whether the model used a tool or
// merely asserted a result.
//
// Three capabilities are probed:
//
//	MCP    knowledge-base retrieval tools (search_bm25_chunks / search_semantic_chunks /
//	       grep_chunks / list_chunks) reached the agent_rag tools through the MCP bridge
//	code   the model generated Python and ran it, yielding an exact computed value
//	agent  a spawned sub-agent received its delegated task and did work of its own
//
// The agent probe is the one that catches the third-party-provider failure mode: when the
// collaboration task is delivered as an unreadable item, the child reports "no task" and
// idles while the parent waits (upstream openai/codex#34833, #36376, #46939).
//
// Subcommands:
//
//	ask      send one question through /api/v1/chat/completions and print the answer
//	inspect  summarize the newest Codex rollouts (no API call needed)
//	verify   run the canned probe for each capability and assert the expected signal
//
// Example:
//
//	RAGFLOW_API_KEY=... RAGFLOW_CHAT_ID=... go run ./tools/mode8verify verify
//	go run ./tools/mode8verify inspect -n 4
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "mode8verify:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return usage()
	}
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "ask":
		return cmdAsk(rest)
	case "inspect":
		return cmdInspect(rest)
	case "verify":
		return cmdVerify(rest)
	case "-h", "--help", "help":
		return usage()
	default:
		return fmt.Errorf("unknown subcommand %q\n\n%s", cmd, usageText)
	}
}

const usageText = `usage: mode8verify <ask|inspect|verify> [flags]

  ask      send one question through the chat completions API
  inspect  summarize the newest Codex rollouts (no API call)
  verify   probe MCP retrieval, the code tool, and sub-agent delegation

Common flags:
  -base-url    RAGFlow API base URL (default http://127.0.0.1:9380)
  -api-key     RAGFlow API key (default $RAGFLOW_API_KEY)
  -chat-id     chat/conversation id (default $RAGFLOW_CHAT_ID)
  -reasoning   reasoning level; mode 8 is 8 (default 8)
  -codex-home  Codex home holding sessions/ (default $CODEX_HOME or ~/.codex)
  -timeout     per-request timeout in seconds (default 900)`

func usage() error {
	fmt.Println(usageText)
	return nil
}

// ---------------------------------------------------------------- configuration

type config struct {
	baseURL   string
	apiKey    string
	chatID    string
	reasoning int
	codexHome string
	timeout   time.Duration
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func defaultCodexHome() string {
	if v := os.Getenv("CODEX_HOME"); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".codex"
	}
	return filepath.Join(home, ".codex")
}

// ---------------------------------------------------------------- HTTP client

type askResult struct {
	Answer    string
	RefChunks int
	Raw       json.RawMessage
}

func (c *config) ask(question, sessionID string) (*askResult, error) {
	if c.apiKey == "" {
		return nil, errors.New("no API key: pass -api-key or set RAGFLOW_API_KEY")
	}
	if c.chatID == "" {
		return nil, errors.New("no chat id: pass -chat-id or set RAGFLOW_CHAT_ID")
	}
	body := map[string]any{
		"chat_id":   c.chatID,
		"question":  question,
		"stream":    false,
		"reasoning": c.reasoning,
	}
	if sessionID != "" {
		body["session_id"] = sessionID
	}
	raw, err := c.post("/api/v1/chat/completions", body)
	if err != nil {
		return nil, err
	}
	var parsed struct {
		Code int `json:"code"`
		Data struct {
			Answer    string `json:"answer"`
			Reference struct {
				Chunks []json.RawMessage `json:"chunks"`
			} `json:"reference"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("decode completion response: %w", err)
	}
	return &askResult{Answer: parsed.Data.Answer, RefChunks: len(parsed.Data.Reference.Chunks), Raw: raw}, nil
}

func (c *config) createSession() (string, error) {
	raw, err := c.post("/api/v1/chats/"+url.PathEscape(c.chatID)+"/sessions",
		map[string]any{"name": fmt.Sprintf("mode8verify-%s", time.Now().Format("20060102-150405"))})
	if err != nil {
		return "", err
	}
	var parsed struct {
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
		ID string `json:"id"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("decode session response: %w", err)
	}
	if parsed.Data.ID != "" {
		return parsed.Data.ID, nil
	}
	return parsed.ID, nil
}

func (c *config) post(path string, body any) (json.RawMessage, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(c.baseURL, "/")+path, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	client := &http.Client{Timeout: c.timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("POST %s: http %d: %s", path, resp.StatusCode, truncate(string(data), 300))
	}
	return json.RawMessage(data), nil
}

// ---------------------------------------------------------------- codex rollouts

// thread is one Codex rollout: a root turn or a spawned sub-agent.
type thread struct {
	ID        string
	Path      string
	Parent    string
	AgentPath string
	Sub       bool
	Model     string
	Tools     map[string]int
	// ShellCmds are the command lines this thread actually executed locally, so a probe
	// can check WHAT ran (e.g. that the model generated and ran Python), not merely that
	// the shell was used.
	ShellCmds []string
	// Delivery of a delegated task, which is how the sub-agent probe distinguishes a real
	// handoff from the third-party-provider failure mode:
	//   TaskAsUser        a standard `message`/`user` item — the provider-neutral path
	//   TaskAgentMessage  Codex's own `agent_message` item (readable only to OpenAI)
	//   TaskOpaque        the payload arrived as encrypted_content
	TaskAsUser       bool
	TaskAgentMessage bool
	TaskOpaque       bool
	Errors           []string
}

// retrievalTools are the agent_rag tools mode 8 exposes over MCP.
var retrievalTools = []string{"search_bm25_chunks", "search_semantic_chunks", "grep_chunks", "list_chunks"}

func (t *thread) callsAny(names []string) (string, bool) {
	for _, n := range names {
		for tool, count := range t.Tools {
			if count > 0 && strings.Contains(tool, n) {
				return tool, true
			}
		}
	}
	return "", false
}

func (t *thread) shellCalls() (string, bool) {
	return t.callsAny([]string{"exec_command", "shell", "CommandExecution"})
}

// pythonExecution returns the first locally-executed command that invokes Python. mode 8 runs
// with a read-only sandbox, so "generate and run a script" means `python3 -c …` or a heredoc
// (`python3 - <<'PY' … PY`) rather than writing a .py file.
func (t *thread) pythonExecution() (string, bool) {
	for _, cmd := range t.ShellCmds {
		if containsPythonInvocation(cmd) {
			return cmd, true
		}
	}
	return "", false
}

func containsPythonInvocation(cmd string) bool {
	for _, token := range []string{"python3", "python", "py -3"} {
		if strings.Contains(cmd, token) {
			return true
		}
	}
	return false
}

// commandFromExecArgs pulls the command line out of an exec_command / shell call's arguments.
// Codex passes it as `cmd`; a list of argv tokens is joined back for display.
func commandFromExecArgs(tool, args string) (string, bool) {
	switch tool {
	case "exec_command", "shell":
	default:
		return "", false
	}
	var parsed struct {
		Cmd     json.RawMessage `json:"cmd"`
		Command json.RawMessage `json:"command"`
	}
	if json.Unmarshal([]byte(args), &parsed) != nil {
		return "", false
	}
	raw := parsed.Cmd
	if len(raw) == 0 {
		raw = parsed.Command
	}
	if len(raw) == 0 {
		return "", false
	}
	var asString string
	if json.Unmarshal(raw, &asString) == nil && asString != "" {
		return asString, true
	}
	var asList []string
	if json.Unmarshal(raw, &asList) == nil && len(asList) > 0 {
		return strings.Join(asList, " "), true
	}
	return "", false
}

func loadThreads(codexHome string, limit int) ([]*thread, error) {
	dir := filepath.Join(codexHome, "sessions")
	var files []string
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // ignore unreadable subtrees
		}
		if !d.IsDir() && strings.HasSuffix(path, ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("scan %s: %w", dir, err)
	}
	sort.Slice(files, func(i, j int) bool {
		ai, _ := os.Stat(files[i])
		aj, _ := os.Stat(files[j])
		return ai.ModTime().After(aj.ModTime())
	})
	if limit > 0 && len(files) > limit {
		files = files[:limit]
	}
	threads := make([]*thread, 0, len(files))
	for _, f := range files {
		t, err := loadThread(f)
		if err != nil {
			continue
		}
		threads = append(threads, t)
	}
	return threads, nil
}

// Loaded rollout before/after a probe lets us attribute new threads to that probe.
func snapshotRollouts(codexHome string) (map[string]bool, error) {
	dir := filepath.Join(codexHome, "sessions")
	seen := map[string]bool{}
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".jsonl") {
			seen[path] = true
		}
		return nil
	})
	return seen, err
}

func loadThread(path string) (*thread, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t := &thread{Path: path, Tools: map[string]int{}}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var entry struct {
			Type    string          `json:"type"`
			Payload json.RawMessage `json:"payload"`
		}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		switch entry.Type {
		case "session_meta":
			var meta struct {
				ID     string          `json:"id"`
				Source json.RawMessage `json:"source"`
			}
			if json.Unmarshal(entry.Payload, &meta) == nil {
				t.ID = meta.ID
				t.parseSource(meta.Source)
			}
		case "turn_context":
			var tc struct {
				Model string `json:"model"`
			}
			if json.Unmarshal(entry.Payload, &tc) == nil && tc.Model != "" {
				t.Model = tc.Model
			}
		case "response_item":
			t.parseResponseItem(entry.Payload)
		case "event_msg":
			t.parseEvent(entry.Payload)
		}
	}
	return t, nil
}

func (t *thread) parseSource(raw json.RawMessage) {
	var asString string
	if json.Unmarshal(raw, &asString) == nil && asString != "" {
		return // root thread (e.g. "vscode")
	}
	var asObject struct {
		Subagent struct {
			ThreadSpawn struct {
				ParentThreadID string `json:"parent_thread_id"`
				AgentPath      string `json:"agent_path"`
			} `json:"thread_spawn"`
		} `json:"subagent"`
	}
	if json.Unmarshal(raw, &asObject) == nil {
		t.Sub = true
		t.Parent = asObject.Subagent.ThreadSpawn.ParentThreadID
		t.AgentPath = asObject.Subagent.ThreadSpawn.AgentPath
	}
}

func (t *thread) parseResponseItem(raw json.RawMessage) {
	var item struct {
		Type      string `json:"type"`
		Name      string `json:"name"`
		Role      string `json:"role"`
		Arguments string `json:"arguments"`
		Content   []struct {
			Type             string `json:"type"`
			Text             string `json:"text"`
			EncryptedContent string `json:"encrypted_content"`
		} `json:"content"`
	}
	if json.Unmarshal(raw, &item) != nil {
		return
	}
	switch item.Type {
	case "function_call":
		if item.Name != "" {
			t.Tools[item.Name]++
		}
		if command, ok := commandFromExecArgs(item.Name, item.Arguments); ok {
			t.ShellCmds = append(t.ShellCmds, command)
		}
	case "message":
		for _, c := range item.Content {
			if item.Role == "user" && strings.Contains(c.Text, "Message Type: NEW_TASK") {
				t.TaskAsUser = true
			}
		}
	case "agent_message":
		t.TaskAgentMessage = true
		for _, c := range item.Content {
			if c.Type == "encrypted_content" && c.EncryptedContent != "" {
				t.TaskOpaque = true
			}
		}
	}
}

func (t *thread) parseEvent(raw json.RawMessage) {
	var event struct {
		Type  string `json:"type"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
		Item *struct {
			Type string `json:"type"`
		} `json:"item"`
	}
	if json.Unmarshal(raw, &event) != nil {
		return
	}
	switch event.Type {
	case "task_complete":
		if event.Error != nil && event.Error.Message != "" {
			t.Errors = append(t.Errors, event.Error.Message)
		}
	case "turn_aborted":
		t.Errors = append(t.Errors, "turn_aborted")
	case "item_completed":
		if event.Item != nil && event.Item.Type == "CommandExecution" {
			t.Tools["CommandExecution"]++
		}
	}
}

func (t *thread) describe() string {
	kind := "root"
	label := t.ID
	if t.Sub {
		kind = "sub"
		label = fmt.Sprintf("%s (%s -> %s)", t.ID, t.AgentPath, t.Parent)
	}
	tools := make([]string, 0, len(t.Tools))
	for name, count := range t.Tools {
		tools = append(tools, fmt.Sprintf("%s=%d", name, count))
	}
	sort.Strings(tools)
	line := fmt.Sprintf("  [%s] %s model=%s tools={%s}", kind, label, t.Model, strings.Join(tools, ", "))
	if cmd, ok := t.pythonExecution(); ok {
		line += fmt.Sprintf(" python=%q", truncate(cmd, 70))
	}
	if t.Sub {
		delivery := "none recorded"
		switch {
		case t.TaskAsUser:
			delivery = "user-message"
		case t.TaskAgentMessage && t.TaskOpaque:
			delivery = "agent_message(encrypted)"
		case t.TaskAgentMessage:
			delivery = "agent_message(plaintext)"
		}
		line += fmt.Sprintf(" task=%s", delivery)
	}
	if len(t.Errors) > 0 {
		line += fmt.Sprintf(" errors=%v", t.Errors)
	}
	return line
}

// ---------------------------------------------------------------- subcommands

func cmdAsk(args []string) error {
	fs := flag.NewFlagSet("ask", flag.ContinueOnError)
	var c config
	fs.StringVar(&c.baseURL, "base-url", envOr("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"), "RAGFlow API base URL")
	fs.StringVar(&c.apiKey, "api-key", os.Getenv("RAGFLOW_API_KEY"), "RAGFlow API key")
	fs.StringVar(&c.chatID, "chat-id", os.Getenv("RAGFLOW_CHAT_ID"), "chat id")
	fs.IntVar(&c.reasoning, "reasoning", 8, "reasoning level")
	fs.StringVar(&c.codexHome, "codex-home", defaultCodexHome(), "Codex home")
	secs := fs.Int("timeout", 900, "per-request timeout in seconds")
	question := fs.String("q", "", "question to ask (required)")
	session := fs.String("session", "", "reuse this session id instead of creating one")
	jsonOut := fs.Bool("json", false, "print the raw JSON response")
	if err := fs.Parse(args); err != nil {
		return err
	}
	c.timeout = time.Duration(*secs) * time.Second
	if *question == "" {
		return errors.New("ask: -q is required")
	}
	sessionID := *session
	if sessionID == "" {
		id, err := c.createSession()
		if err != nil {
			return fmt.Errorf("create session: %w", err)
		}
		sessionID = id
		fmt.Printf("session=%s\n", sessionID)
	}
	res, err := c.ask(*question, sessionID)
	if err != nil {
		return err
	}
	if *jsonOut {
		fmt.Println(string(res.Raw))
		return nil
	}
	fmt.Printf("reference_chunks=%d\n\n%s\n", res.RefChunks, res.Answer)
	return nil
}

func cmdInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	codexHome := fs.String("codex-home", defaultCodexHome(), "Codex home")
	n := fs.Int("n", 6, "how many recent rollouts to summarize")
	if err := fs.Parse(args); err != nil {
		return err
	}
	threads, err := loadThreads(*codexHome, *n)
	if err != nil {
		return err
	}
	if len(threads) == 0 {
		return fmt.Errorf("no rollouts found under %s/sessions", *codexHome)
	}
	fmt.Printf("newest %d Codex rollout(s) under %s/sessions:\n", len(threads), *codexHome)
	for _, t := range threads {
		fmt.Println(t.describe())
	}
	return nil
}

// probe is one capability check: a question plus the signal the probe must produce.
type probe struct {
	name     string
	question string
	// check runs against the threads created while the probe's question was in flight and
	// the answer it produced. A nil error passes; an error means failure.
	check func(threads []*thread, answer string) error
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	var c config
	fs.StringVar(&c.baseURL, "base-url", envOr("RAGFLOW_BASE_URL", "http://127.0.0.1:9380"), "RAGFlow API base URL")
	fs.StringVar(&c.apiKey, "api-key", os.Getenv("RAGFLOW_API_KEY"), "RAGFlow API key")
	fs.StringVar(&c.chatID, "chat-id", os.Getenv("RAGFLOW_CHAT_ID"), "chat id")
	fs.IntVar(&c.reasoning, "reasoning", 8, "reasoning level")
	fs.StringVar(&c.codexHome, "codex-home", defaultCodexHome(), "Codex home")
	secs := fs.Int("timeout", 900, "per-request timeout in seconds")
	if err := fs.Parse(args); err != nil {
		return err
	}
	c.timeout = time.Duration(*secs) * time.Second

	probes := []probe{
		{
			name: "mcp",
			question: "请用知识库检索工具（search_bm25_chunks / search_semantic_chunks / grep_chunks / list_chunks）" +
				"查找「量子纠错」的相关内容，给出一个要点，并在句末附上出处引用。",
			check: func(threads []*thread, _ string) error {
				for _, t := range threads {
					if tool, ok := t.callsAny(retrievalTools); ok {
						return fmt.Errorf("ok: %s called %s", t.ID, tool)
					}
				}
				return errors.New("no retrieval tool call in any new thread")
			},
		},
		{
			name: "code",
			question: "请生成一段 Python 脚本精确计算 2 的 100 次方减 1，并用代码执行工具运行这段脚本，" +
				"在回答中给出完整的十进制结果。",
			check: func(threads []*thread, answer string) error {
				const want = "1267650600228229401496703205375"
				for _, t := range threads {
					cmd, ok := t.pythonExecution()
					if !ok {
						continue
					}
					if !strings.Contains(answer, want) {
						return fmt.Errorf("Python ran (%s) but the answer lacks the exact value %s", truncate(cmd, 80), want)
					}
					return fmt.Errorf("ok: %s ran generated Python (%s) and the answer carries the exact value",
						t.ID, truncate(cmd, 80))
				}
				// Distinguish "the shell was never used" from "it ran something that is not
				// Python": the probe is about executing GENERATED code, not any command.
				for _, t := range threads {
					if len(t.ShellCmds) > 0 {
						return fmt.Errorf("shell ran but no Python execution: %s", truncate(t.ShellCmds[0], 80))
					}
				}
				return errors.New("no local code execution in any new thread")
			},
		},
		{
			name: "agent",
			question: "请使用 spawn_agent 派一个子智能体（显式传入 fork_turns=\"none\"），让它用知识库检索工具调研「量子纠错」" +
				"并给出两个要点；等它返回后再把结果汇总成最终回答。",
			check: func(threads []*thread, _ string) error {
				// The delegated task is a retrieval task, so the only proof the handoff
				// actually worked is the CHILD running a retrieval tool. A child that
				// idles, or only pokes around the sandbox with exec_command, means the
				// task never reached it (or was unreadable) — the upstream
				// third-party-provider failure this probe exists to catch.
				sawSub := false
				for _, t := range threads {
					if !t.Sub {
						continue
					}
					sawSub = true
					delivery := "user-message"
					switch {
					case t.TaskOpaque:
						delivery = "encrypted payload"
					case t.TaskAgentMessage && !t.TaskAsUser:
						delivery = "agent_message item (not provider-neutral)"
					}
					tool, ok := t.callsAny(retrievalTools)
					if !ok {
						return fmt.Errorf("sub-agent %s (task via %s) called no retrieval tool; it did not receive/act on its task",
							t.ID, delivery)
					}
					return fmt.Errorf("ok: sub-agent %s (task via %s) ran %s", t.ID, delivery, tool)
				}
				if !sawSub {
					return errors.New("no sub-agent thread was created")
				}
				return errors.New("sub-agent inspection failed")
			},
		},
	}

	failures := 0
	for _, p := range probes {
		before, err := snapshotRollouts(c.codexHome)
		if err != nil {
			return err
		}
		sessionID, err := c.createSession()
		if err != nil {
			return fmt.Errorf("create session: %w", err)
		}
		res, err := c.ask(p.question, sessionID)
		if err != nil {
			fmt.Printf("[%s] FAIL: %v\n", p.name, err)
			failures++
			continue
		}
		if strings.Contains(res.Answer, "**ERROR**") {
			fmt.Printf("[%s] FAIL: turn reported an error: %s\n", p.name, truncate(res.Answer, 200))
			failures++
			continue
		}
		created, err := newThreadsSince(c.codexHome, before)
		if err != nil {
			return err
		}
		if err := p.check(created, res.Answer); err != nil {
			verdict := "FAIL"
			if strings.HasPrefix(err.Error(), "ok:") {
				verdict = "PASS"
			} else {
				failures++
			}
			fmt.Printf("[%s] %s: %s\n", p.name, verdict, strings.TrimPrefix(err.Error(), "ok: "))
		}
	}
	if failures > 0 {
		return fmt.Errorf("%d of %d probes failed", failures, len(probes))
	}
	return nil
}

func newThreadsSince(codexHome string, before map[string]bool) ([]*thread, error) {
	after, err := snapshotRollouts(codexHome)
	if err != nil {
		return nil, err
	}
	var created []*thread
	for path := range after {
		if before[path] {
			continue
		}
		if t, err := loadThread(path); err == nil {
			created = append(created, t)
		}
	}
	sort.Slice(created, func(i, j int) bool { return created[i].ID < created[j].ID })
	return created, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
