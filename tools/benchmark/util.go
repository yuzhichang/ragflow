package main

// Shared small helpers: dynamic-JSON accessors mirroring the Python
// original's dict handling, path resolution and the JSONL/JSON IO used by
// every phase. Field order in the written artefacts matters (the leaderboard
// is a submission document), so the ordered builders live in leaderboard.go.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// Dynamic JSON accessors (port of _as_float / _as_int / _as_string_list / ...)
// ---------------------------------------------------------------------------

func asFloat(v any) *float64 {
	switch x := v.(type) {
	case nil:
		return nil
	case float64:
		return &x
	case int:
		f := float64(x)
		return &f
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return nil
		}
		return &f
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(x), 64)
		if err != nil {
			return nil
		}
		return &f
	case bool:
		return nil
	default:
		return nil
	}
}

func asInt(v any) *int {
	f := asFloat(v)
	if f == nil {
		return nil
	}
	n := int(*f)
	return &n
}

func asIntDefault(v any, def int) int {
	if n := asInt(v); n != nil {
		return *n
	}
	return def
}

func asStringList(v any) []string {
	if v == nil || v == "" {
		return nil
	}
	switch x := v.(type) {
	case string:
		if x == "" {
			return nil
		}
		return []string{x}
	case float64:
		return []string{fmt.Sprintf("%v", x)}
	case bool:
		return []string{fmt.Sprintf("%v", x)}
	case []any:
		var out []string
		for _, item := range x {
			if item == nil {
				continue
			}
			out = append(out, fmt.Sprintf("%v", item))
		}
		return out
	default:
		return nil
	}
}

func strAny(m map[string]any, key, def string) string {
	if m == nil {
		return def
	}
	switch x := m[key].(type) {
	case string:
		if x == "" {
			return def
		}
		return x
	case nil:
		return def
	default:
		return fmt.Sprintf("%v", x)
	}
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func absPath(p string) string {
	abs, err := filepath.Abs(expandHome(p))
	if err != nil {
		return p
	}
	return abs
}

func expandHome(p string) string {
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

func resolvePath(value, baseDir string) string {
	p := expandHome(value)
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Clean(filepath.Join(baseDir, p))
}

func resolveOutputDir(value string) string {
	timestamp := time.Now().Format("20060102_150405")
	p := expandHome(strings.ReplaceAll(value, "<timestamp>", timestamp))
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	wd, _ := os.Getwd()
	return filepath.Clean(filepath.Join(wd, p))
}

func loadJSONMap(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("invalid JSON in %s: %w", path, err)
	}
	return payload, nil
}

func writeJSON(path string, payload any) {
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		fmt.Printf("[io] could not marshal %s: %v\n", path, err)
		return
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		fmt.Printf("[io] could not write %s: %v\n", path, err)
	}
}

func readJSONL(path string) []map[string]any {
	if !fileExists(path) {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("[io] could not read %s: %v\n", path, err)
		return nil
	}
	var rows []map[string]any
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			fmt.Printf("[io] invalid JSONL at %s:%d: %v\n", path, i+1, err)
			return rows
		}
		rows = append(rows, row)
	}
	return rows
}

func toAnySlice(values []string) []any {
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}

func compact(payload any) string {
	var text string
	if s, ok := payload.(string); ok {
		text = s
	} else {
		data, err := json.Marshal(payload)
		if err != nil {
			text = fmt.Sprintf("%v", payload)
		} else {
			text = string(data)
		}
	}
	if len(text) > 1000 {
		return text[:1000]
	}
	return text
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }

func round6(v float64) float64 { return math.Round(v*1e6) / 1e6 }

// naturalKey ports _natural_key: numeric ids sort numerically with a fixed
// width, everything else sorts lexically after them.
func naturalKey(value string) string {
	if n, err := strconv.Atoi(value); err == nil {
		return fmt.Sprintf("0:%012d", n)
	}
	return "1:" + value
}

// ---------------------------------------------------------------------------
// Resume accounting
// ---------------------------------------------------------------------------

// resumeSplit splits the deduped rows into (clean, persistent-error,
// transient-error) counts - the [resume] line's triple.
func resumeSplit(rows []map[string]any) (int, int, int) {
	answered, persistent := 0, 0
	for _, row := range rows {
		errText := strings.TrimSpace(strAny(row, "ragflow_error", ""))
		switch {
		case errText == "":
			answered++
		case isPersistentRagflowError(errText):
			persistent++
		}
	}
	return answered, persistent, len(rows) - answered - persistent
}

// persistentErrorMarkers: markers of an error that makes a retry pointless -
// the provider refused the INPUT itself (content policy), so the same
// question fails identically on every attempt (q744: five identical
// "input new_sensitive" failures across as many resumes). Everything else
// (quota walls, connection aborts, timeouts, retry exhaustion with a
// recoverable cause) stays transient and is retried; an unknown error also
// defaults to transient - one wasted retry beats one silently abandoned
// question.
var persistentErrorMarkers = []string{
	"new_sensitive",
	"sensitive content",
	"content_filter",
	"content filter",
}

// isPersistentRagflowError ports _is_persistent_ragflow_error.
func isPersistentRagflowError(err string) bool {
	text := strings.ToLower(err)
	for _, marker := range persistentErrorMarkers {
		if strings.Contains(text, marker) {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Run provenance (port of _record_run_provenance)
// ---------------------------------------------------------------------------

// recordRunProvenance stamps the run's git commit into
// _prompt_fingerprint.json, once. The archived rows must answer "what code
// produced them" without log archaeology (the #46 regression review needed
// exactly that, and had to reconstruct it from launch scripts). A fingerprint
// that already carries a commit is left untouched: a resumed run must not
// overwrite the commit of the run that produced the rows.
func recordRunProvenance(outputDir string) string {
	fingerprint := filepath.Join(outputDir, fingerprintName)
	existing := map[string]any{}
	if fileExists(fingerprint) {
		data, err := os.ReadFile(fingerprint)
		if err != nil {
			fmt.Printf("[run] could not read %s: %v; a fresh one will be written\n", fingerprintName, err)
		} else if err := json.Unmarshal(data, &existing); err != nil {
			fmt.Printf("[run] could not read %s: %v; a fresh one will be written\n", fingerprintName, err)
			existing = map[string]any{}
		}
	}
	if commit, _ := existing["git_commit"].(string); commit != "" {
		return commit
	}
	out, err := exec.Command("git", "rev-parse", "--short=9", "HEAD").Output()
	if err != nil {
		fmt.Printf("[run] git commit not recorded: %v\n", err)
		return ""
	}
	commit := strings.TrimSpace(string(out))
	existing["git_commit"] = commit
	if _, ok := existing["git_commit_recorded_at"]; !ok {
		existing["git_commit_recorded_at"] = time.Now().Format("2006-01-02 15:04:05")
	}
	data, _ := json.MarshalIndent(existing, "", "  ")
	tmp := fingerprint + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err == nil {
		os.Rename(tmp, fingerprint)
	}
	fmt.Printf("[run] git commit recorded: %s -> %s\n", commit, fingerprintName)
	return commit
}

// ---------------------------------------------------------------------------
// Sorting helper for the {id: {...}} question mapping
// ---------------------------------------------------------------------------

func sortIDs(ids []string) {
	sort.Slice(ids, func(i, j int) bool { return naturalKey(ids[i]) < naturalKey(ids[j]) })
}

// ---------------------------------------------------------------------------
// Doc-id normalisation (port of _normalize_benchmark_doc_id and friends)
// ---------------------------------------------------------------------------

// normalizeDocID ports _normalize_benchmark_doc_id: strip any directory path
// and a trailing ".md" - the corpus documents are addressed by their stem.
func normalizeDocID(value any) string {
	if value == nil {
		return ""
	}
	text := strings.TrimSpace(fmt.Sprintf("%v", value))
	if text == "" {
		return ""
	}
	filename := text
	if i := strings.LastIndexAny(filename, "/\\"); i >= 0 {
		filename = filename[i+1:]
	}
	if strings.HasSuffix(strings.ToLower(filename), ".md") {
		filename = filename[:len(filename)-3]
	}
	return strings.TrimSpace(filename)
}

// uniqueDocIDs ports _unique_doc_ids: first occurrence wins, empties dropped.
func uniqueDocIDs(values any) []string {
	seen := map[string]bool{}
	var ids []string
	list, ok := values.([]any)
	if !ok {
		if s := normalizeDocID(values); s != "" {
			return []string{s}
		}
		return nil
	}
	for _, value := range list {
		normalized := normalizeDocID(value)
		if normalized == "" || seen[normalized] {
			continue
		}
		seen[normalized] = true
		ids = append(ids, normalized)
	}
	return ids
}

// asDocIDList ports _as_doc_id_list: accept a scalar, a list, or missing.
func asDocIDList(value any) []string {
	if value == nil || value == "" {
		return nil
	}
	list, ok := value.([]any)
	if !ok {
		return uniqueDocIDs(value)
	}
	return uniqueDocIDs(list)
}

// uniqueIDs ports _unique_ids: deduplicate opaque identifiers (chunk ids),
// preserving the backend's own order - NO document-name normalisation.
func uniqueIDs(values any) []string {
	list, ok := values.([]any)
	if !ok {
		return nil
	}
	seen := map[string]bool{}
	var ids []string
	for _, value := range list {
		text := strings.TrimSpace(fmtAny(value))
		if text == "" || seen[text] {
			continue
		}
		seen[text] = true
		ids = append(ids, text)
	}
	return ids
}

func fmtAny(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(data)
}

func timeNow() time.Time { return time.Now() }
