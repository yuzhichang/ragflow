package agentic_rag

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Some tool arguments are routinely sent in a shape the declared JSON schema
// does not ask for. The 830-question BrowseComp run logged 46 such calls and
// each one is a wasted turn: the tool errors, the model re-sends the same
// shape, and the round trip is gone. The four shapes actually observed:
//
//	think:       {"total_thoughts": "3"}                 number as a string
//	list_chunks: {"anchor_chunk_ids": {"0": "c1"}}       list as an object
//	todo_write:  {"steps": {"1": {...}}}                 list as an object
//	grep_chunks: {"query": "1[\\u2013\\-]3 feet"}        JSON-style escape
//
// Tolerance is OPT-IN, one field at a time: a field that must also accept a
// neighbouring shape is declared with one of the types below. Every other field
// stays a plain Go type and keeps failing loudly, so the schema still means
// what it says and a genuinely malformed call is still an error.

// flexInt is an int that also accepts a string, with or without prose around
// it: 3, "3", " 3 ", "3 thoughts" -> 3, 3, 3, 3; "" and null -> 0.
type flexInt int

func (f *flexInt) UnmarshalJSON(data []byte) error {
	raw := strings.TrimSpace(string(data))
	if raw == "" || raw == "null" {
		*f = 0
		return nil
	}
	if raw[0] == '"' {
		var text string
		if err := json.Unmarshal(data, &text); err != nil {
			return err
		}
		raw = strings.TrimSpace(text)
		if raw == "" {
			*f = 0
			return nil
		}
	}
	if number, err := strconv.ParseFloat(raw, 64); err == nil {
		*f = flexInt(number)
		return nil
	}
	if lead := leadingNumber.FindString(raw); lead != "" {
		number, err := strconv.ParseFloat(lead, 64)
		if err != nil {
			return fmt.Errorf("expected an integer, got %q", raw)
		}
		*f = flexInt(number)
		return nil
	}
	return fmt.Errorf("expected an integer, got %q", raw)
}

// leadingNumber matches a number at the start of a string ("3 thoughts").
var leadingNumber = regexp.MustCompile(`^[-+]?\d+(\.\d+)?`)

// flexStrings is a []string that also accepts the list shapes models send:
//
//	["a","b"]              the declared shape
//	"a"                    one value where a list goes
//	{"0":"a","1":"b"}      an index-keyed object
//	{"queries": ["a","b"]} one key wrapping the list
//	{"queries": "a"}       one key wrapping a single value
//
// Keys only order the result (numerically when they are numbers); they are not
// copied into the values.
type flexStrings []string

func (s *flexStrings) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*s = nil
		return nil
	}
	switch trimmed[0] {
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return err
		}
		values := make([]string, 0, len(items))
		for _, item := range items {
			value, ok := scalarText(item)
			if !ok {
				return fmt.Errorf("expected string items, got %s", shorten(item))
			}
			values = append(values, value)
		}
		*s = values
		return nil
	case '{':
		var object map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &object); err != nil {
			return err
		}
		if values, ok := unwrapSingleList(object); ok {
			*s = values
			return nil
		}
		keys := make([]string, 0, len(object))
		for key := range object {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool { return naturalKeyLess(keys[i], keys[j]) })
		values := make([]string, 0, len(keys))
		for _, key := range keys {
			value, ok := scalarText(object[key])
			if !ok {
				return fmt.Errorf("expected string values, got %s", shorten(object[key]))
			}
			values = append(values, value)
		}
		*s = values
		return nil
	default:
		value, ok := scalarText(trimmed)
		if !ok {
			return fmt.Errorf("expected a string or a list of strings, got %s", shorten(trimmed))
		}
		*s = []string{value}
		return nil
	}
}

// flexPlanSteps is a []planStep that also accepts todo_write's other shapes:
// one step object, a keyed object of steps (the key becomes the step id), a
// bare string, or a list of strings.
type flexPlanSteps []planStep

func (s *flexPlanSteps) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		*s = nil
		return nil
	}
	switch trimmed[0] {
	case '[':
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return err
		}
		steps := make([]planStep, 0, len(items))
		for _, item := range items {
			step, err := decodePlanStep(item)
			if err != nil {
				return err
			}
			steps = append(steps, step)
		}
		*s = steps
		return nil
	case '{':
		var object map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &object); err != nil {
			return err
		}
		if inner, ok := object["steps"]; ok {
			var steps flexPlanSteps
			if err := steps.UnmarshalJSON(inner); err != nil {
				return err
			}
			*s = steps
			return nil
		}
		if isPlanStepObject(object) {
			step, err := decodePlanStep(trimmed)
			if err != nil {
				return err
			}
			*s = []planStep{step}
			return nil
		}
		keys := make([]string, 0, len(object))
		for key := range object {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool { return naturalKeyLess(keys[i], keys[j]) })
		steps := make([]planStep, 0, len(keys))
		for _, key := range keys {
			step, err := decodePlanStep(object[key])
			if err != nil {
				return err
			}
			if step.ID == "" {
				step.ID = key
			}
			steps = append(steps, step)
		}
		*s = steps
		return nil
	default:
		step, err := decodePlanStep(trimmed)
		if err != nil {
			return err
		}
		*s = []planStep{step}
		return nil
	}
}

// planStepTextFields are the field names a bare string stands for, most
// specific first ("steps": ["locate the film"]).
var planStepTextFields = []string{"description", "step", "task", "title", "name", "text", "detail"}

// planStepIDFields and planStepStatusFields name the same field under its
// common aliases.
var (
	planStepIDFields     = []string{"id", "step_id", "stepId"}
	planStepStatusFields = []string{"status", "state"}
)

// isPlanStepObject reports whether an object is a single step rather than a map
// of steps: it carries at least one of the fields a step has.
func isPlanStepObject(object map[string]json.RawMessage) bool {
	for _, name := range append(append([]string{}, planStepIDFields...), append(planStepTextFields, planStepStatusFields...)...) {
		if _, ok := object[name]; ok {
			return true
		}
	}
	return false
}

// decodePlanStep reads one step, accepting the field aliases models use and
// filling in an id when they omit it.
func decodePlanStep(raw json.RawMessage) (planStep, error) {
	text, isText := scalarText(raw)
	if isText && len(bytes.TrimSpace(raw)) > 0 && bytes.TrimSpace(raw)[0] != '{' {
		if strings.TrimSpace(text) == "" {
			return planStep{}, fmt.Errorf("expected a non-empty step description")
		}
		return planStep{Description: strings.TrimSpace(text), Status: planStepPending}, nil
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		return planStep{}, fmt.Errorf("expected a step object, got %s", shorten(raw))
	}
	readField := func(names []string) string {
		for _, name := range names {
			if value, ok := object[name]; ok {
				if text, ok := scalarText(value); ok {
					return strings.TrimSpace(text)
				}
			}
		}
		return ""
	}
	step := planStep{
		ID:          readField(planStepIDFields),
		Description: readField(planStepTextFields),
		Status:      normalizePlanStatus(readField(planStepStatusFields)),
	}
	if step.Description == "" {
		return planStep{}, fmt.Errorf("expected a step description, got %s", shorten(raw))
	}
	return step, nil
}

// planStepPending is the status a step without one is shown as.
const planStepPending = "pending"

// normalizePlanStatus folds the status spellings models mix up.
func normalizePlanStatus(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "", "pending", "todo", "to do", "not started", "open":
		return planStepPending
	case "in_progress", "in progress", "in-progress", "doing", "current", "started":
		return "in_progress"
	case "completed", "complete", "done", "finished", "closed":
		return "completed"
	default:
		return strings.ToLower(strings.TrimSpace(status))
	}
}

// unwrapSingleList handles {"queries": ["a","b"]} and {"queries": "a"} - one key
// holding the list that was declared on the field itself.
func unwrapSingleList(object map[string]json.RawMessage) ([]string, bool) {
	if len(object) != 1 {
		return nil, false
	}
	for _, value := range object {
		trimmed := bytes.TrimSpace(value)
		if len(trimmed) == 0 {
			return nil, false
		}
		if trimmed[0] != '[' {
			text, ok := scalarText(trimmed)
			if !ok {
				return nil, false
			}
			return []string{text}, true
		}
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return nil, false
		}
		values := make([]string, 0, len(items))
		for _, item := range items {
			text, ok := scalarText(item)
			if !ok {
				return nil, false
			}
			values = append(values, text)
		}
		return values, true
	}
	return nil, false
}

// scalarText renders a JSON scalar as text: strings as they are, numbers and
// booleans as the literal that spells them. Objects and arrays do not qualify.
func scalarText(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "", false
	}
	switch trimmed[0] {
	case '{', '[':
		return "", false
	case '"':
		var text string
		if err := json.Unmarshal(trimmed, &text); err != nil {
			return "", false
		}
		return text, true
	case 'n':
		return "", true
	default:
		return string(trimmed), true
	}
}

// naturalKeyLess orders numeric-looking keys numerically, so {"10": …, "9": …}
// keeps the order the model wrote instead of a lexicographic one.
func naturalKeyLess(left, right string) bool {
	leftNumber, leftErr := strconv.Atoi(left)
	rightNumber, rightErr := strconv.Atoi(right)
	if leftErr == nil && rightErr == nil {
		return leftNumber < rightNumber
	}
	return left < right
}

// shorten trims a payload for an error message.
func shorten(raw json.RawMessage) string {
	const limit = 80
	text := strings.TrimSpace(string(raw))
	if len(text) <= limit {
		return text
	}
	return text[:limit] + "..."
}

// regexUnicodeEscape matches the \uXXXX / \UXXXXXXXX escapes a model writes for
// other engines: RE2 has no \u form, so the pattern would be rejected outright.
var regexUnicodeEscape = regexp.MustCompile(`\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})`)

// normalizeRegexPattern rewrites those escapes into the characters they mean
// ("1[\u2013\-]3 feet" -> "1[–\-]3 feet"), so a JSON-style escape does not fail
// a grep that meant a literal dash.
func normalizeRegexPattern(pattern string) string {
	if !strings.Contains(pattern, `\u`) && !strings.Contains(pattern, `\U`) {
		return pattern
	}
	return regexUnicodeEscape.ReplaceAllStringFunc(pattern, func(match string) string {
		code, err := strconv.ParseInt(strings.TrimLeft(match[2:], "0"), 16, 64)
		if err != nil {
			code = 0
		}
		if code > 0x10FFFF {
			return match
		}
		return string(rune(code))
	})
}
