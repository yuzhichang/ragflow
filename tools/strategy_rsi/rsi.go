//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// rating is one self-reflection score for an in-force strategy (basis=used).
type rating struct {
	StrategyID string `json:"strategy_id"`
	Score      int    `json:"score"`
	Reason     string `json:"reason,omitempty"`
}

// questionOutcome is one question's result within a batch.
type questionOutcome struct {
	Index   int // batch position
	QID     string
	Correct bool
	Skipped bool
	Err     string
	Lessons []lesson
	Ratings []rating
	Answer  string
	Verdict *verdict
	// SentQuestion is the EXACT text sent to the model (strategy block + question).
	SentQuestion string
	// SessionID is the session actually used for the (final) test turn.
	SessionID string
}

// reflectionReply is the model's self-reflection JSON (plan §6.1). Scores are pointers so
// a MISSING score is distinguishable from an explicit 0 and can be rejected (never
// zero-filled).
type reflectionReply struct {
	Ratings []struct {
		StrategyID string `json:"strategy_id"`
		Score      *int   `json:"score"`
		Reason     string `json:"reason"`
	} `json:"ratings"`
	Lessons []struct {
		Text     string `json:"text"`
		Score    *int   `json:"score"`
		Reason   string `json:"reason"`
		Replaces string `json:"replaces"`
	} `json:"lessons"`
}

// reflectionPromptTemplate is the fixed head of the self-reflection prompt (fingerprinted).
const reflectionPromptTemplate = "Your search did not find the correct answer."

// reflectionPrompt builds the same-session self-reflection prompt (§6.1). It lists the
// in-force strategies for rating/reference and explicitly ALLOWS restatements and
// replacements -- dedup/mapping is the batch stage's job, not the model's here.
func reflectionPrompt(set *strategySet, gold string, maxLessons int) string {
	return fmt.Sprintf(`Your search did not find the correct answer.
The correct answer is: %s

Search strategies currently in force (shown for reference — you may RATE them, and you may
give a lesson that RESTATES one (it will be mapped to it) or that REPLACES one, by naming its id):
%s
Reflect on how you searched, then give AT MOST %d lessons to improve the SEARCH STRATEGY
(NOT the answer to this specific question), subject to these rules:
- TERSE: each item is ONE imperative sentence, <= 30 words. No preamble, no explanation, no examples.
- FOCUSED: target a decision axis (planning / clue splitting / subagent assignment /
  parallelization / cross-verification / dedupe). A restatement is allowed — it will be mapped
  to the existing strategy, do NOT drop it.
- REPLACEMENTS ALLOWED: to correct an in-force strategy you believe is wrong, emit a lesson
  with "replaces":"<strategy_id>" instead of dropping it.
- CORPUS-AGNOSTIC: no entity names, proper nouns, years/numbers, quoted phrases, URLs, or any
  wording that only makes sense for this corpus or this question. Use only generic methodology
  words (search / clue / subagent / parallel / assign / verify / merge).
Rate EACH in-force strategy with an integer -2..+2 for how it helped this search:
-2 misleading, -1 unhelpful, 0 neutral/no evidence, +1 helpful, +2 very helpful. Add a short reason.
Also give each new lesson a predicted usefulness score on the same scale, and a short reason;
do not claim a new lesson was actually used.
Output ONLY JSON:
{"ratings":[{"strategy_id":"...","score":0,"reason":"..."}],
 "lessons":[{"text":"...","score":1,"reason":"...","replaces":"<strategy_id or omitted>"}]}
The lessons array must contain at most %d items.`, gold, set.strategyListForPrompt(), maxLessons, maxLessons)
}

// parseReflection validates the reflection JSON. A rating needs an in-force id, a score
// in [-2,2] and no duplicate; a lesson needs non-empty text and a valid score. Anything
// invalid is DROPPED (never zero-filled), and at most maxLessons lessons are kept.
func parseReflection(text string, active map[string]bool, maxLessons int) ([]lesson, []rating, error) {
	cleaned := strings.TrimSpace(thinkRE.ReplaceAllString(text, ""))
	var rep reflectionReply
	if err := json.Unmarshal([]byte(cleaned), &rep); err != nil {
		lo, hi := strings.Index(cleaned, "{"), strings.LastIndex(cleaned, "}")
		if lo < 0 || hi <= lo || json.Unmarshal([]byte(cleaned[lo:hi+1]), &rep) != nil {
			return nil, nil, fmt.Errorf("could not parse reflection JSON from: %s", truncate(cleaned, 300))
		}
	}
	var ratings []rating
	seen := map[string]bool{}
	for _, r := range rep.Ratings {
		if r.Score == nil || *r.Score < -2 || *r.Score > 2 || !active[r.StrategyID] || seen[r.StrategyID] {
			continue
		}
		seen[r.StrategyID] = true
		ratings = append(ratings, rating{StrategyID: r.StrategyID, Score: *r.Score, Reason: r.Reason})
	}
	var lessons []lesson
	for _, l := range rep.Lessons {
		if maxLessons > 0 && len(lessons) >= maxLessons {
			break
		}
		if strings.TrimSpace(l.Text) == "" || l.Score == nil || *l.Score < -2 || *l.Score > 2 {
			continue
		}
		lessons = append(lessons, lesson{Text: l.Text, Score: *l.Score, Reason: l.Reason, Replaces: l.Replaces})
	}
	return lessons, ratings, nil
}

// reconcileState is the four-state reconcile outcome (§6.4).
type reconcileState int

const (
	reconcileUnknown reconcileState = iota
	reconcileDone
	reconcileAbsent
)

// matchTurn inspects stored history for our question: Done (user turn + assistant reply),
// Absent (not present), or Unknown (present without a reply / cannot tell).
func matchTurn(hist []sessionTurn, question string) (reconcileState, string) {
	norm := func(s string) string { return strings.TrimSpace(strings.Join(strings.Fields(s), " ")) }
	want := norm(question)
	qi := -1
	for i, t := range hist {
		if t.Role == "user" && norm(t.Content) == want {
			qi = i
		}
	}
	if qi < 0 {
		return reconcileAbsent, ""
	}
	for j := qi + 1; j < len(hist); j++ {
		if hist[j].Role == "assistant" && strings.TrimSpace(hist[j].Content) != "" {
			return reconcileDone, hist[j].Content
		}
	}
	return reconcileUnknown, ""
}

// runQuestion executes one question within a batch: test with the frozen strategy block,
// judge it, and (on a wrong answer) self-reflect on the same session for ≤2 lessons.
func (e *executor) runQuestion(ctx context.Context, snap batchSnapshot, rec trainsetRecord, qpos int) questionOutcome {
	out := questionOutcome{Index: qpos, QID: rec.ID}
	if e.client == nil {
		out.Skipped = true
		out.Err = "no client"
		return out
	}

	question := snap.strategyBlock()
	if question != "" {
		question += "\n\n"
	}
	question += rec.Question
	out.SentQuestion = question

	session, err := e.createSession(ctx)
	if err != nil {
		out.Skipped = true
		out.Err = "create session: " + err.Error()
		return out
	}
	answer, session, rerr := e.askIsolated(ctx, session, question)
	out.SessionID = session
	if rerr != nil {
		out.Skipped = true
		out.Err = rerr.Error()
		return out
	}
	out.Answer = answer

	v, jerr := e.judgeAnswer(ctx, rec, answer)
	if jerr != nil {
		out.Skipped = true
		out.Err = "judge: " + jerr.Error()
		return out
	}
	out.Verdict = &v
	out.Correct = v.Correct
	if v.Correct {
		return out
	}

	lessons, ratings, lerr := e.reflect(ctx, session, rec, snap, qpos)
	if lerr != nil {
		// Self-reflection failure != question failure: keep the verdict, no lessons.
		out.Err = "reflect: " + lerr.Error()
		return out
	}
	out.Lessons = lessons
	out.Ratings = ratings
	return out
}

// createSession opens a session, retrying transport-class failures the way the ask path
// does. Opening a session is infrastructure, not a question outcome: a bare
// "connection refused" used to mark the question skipped, so a brief API outage "completed"
// whole batches with zero scored questions and advanced the cursor past them.
func (e *executor) createSession(ctx context.Context) (string, error) {
	var sid string
	err := callWithRetry(ctx, e.retry, func(int) error {
		id, err := e.client.createSession(ctx, e.cfg.ChatID)
		if err != nil {
			return err
		}
		sid = id
		return nil
	})
	if err != nil {
		return "", err
	}
	return sid, nil
}

// askIsolated sends one test turn, starting a NEW session before any quota/network
// retry: an uncertain outcome is isolated rather than duplicated on the original
// session (§6.4). Timeouts skip the question.
func (e *executor) askIsolated(ctx context.Context, session, question string) (string, string, error) {
	cur := session
	for attempt := 0; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return "", cur, err
		}
		res, err := e.client.ask(ctx, e.cfg.ChatID, question, cur, e.cfg.Reasoning)
		if err == nil {
			if answerLooksLikeError(res.Answer) {
				err = fmt.Errorf("answer is a provider error: %s", truncate(res.Answer, 120))
			} else {
				if res.SessionID != "" {
					cur = res.SessionID
				}
				return res.Answer, cur, nil
			}
		}
		if ctx.Err() != nil {
			return "", cur, ctx.Err()
		}
		switch classify(err) {
		case failTimeout:
			return "", cur, fmt.Errorf("timeout (skip): %w", err)
		case failPermanent:
			return "", cur, err
		default: // quota / network
			if hist, ok := e.client.sessionHistory(ctx, e.cfg.ChatID, cur); ok {
				if st, ans := matchTurn(hist, question); st == reconcileDone {
					return ans, cur, nil
				}
			}
			if serr := e.retry.sleepClass(ctx, classify(err), attempt); serr != nil {
				return "", cur, serr
			}
			ns, cerr := e.client.createSession(ctx, e.cfg.ChatID)
			if cerr != nil {
				return "", cur, cerr
			}
			e.logf("ask: %s failure, isolated retry on a new session (was %s)", classify(err), cur)
			cur = ns
		}
	}
}

// reflect asks the failed session for ≤2 lessons. On quota/network the plan says retry
// after reconcile; when the outcome cannot be determined the lesson set is abandoned
// (the verdict is kept) rather than risk rereading a stale reflection.
func (e *executor) reflect(ctx context.Context, session string, rec trainsetRecord, snap batchSnapshot, qpos int) ([]lesson, []rating, error) {
	set := snap.Strategies
	if set == nil {
		set = newStrategySet()
	}
	active := map[string]bool{}
	for _, id := range set.ActiveIDs {
		active[id] = true
	}
	prompt := reflectionPrompt(set, rec.GoldAnswer, e.cfg.MaxLessons)
	assign := func(ls []lesson) []lesson {
		for i := range ls {
			ls[i].TentativeID = fmt.Sprintf("t%d.%d.%d.%d", snap.Pass, snap.Index, qpos, i)
			ls[i].QID = rec.ID
		}
		return ls
	}

	for attempt := 0; ; attempt++ {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		// Keeps the ANSWERING engine (cfg.Reasoning) on purpose: reflection continues the
		// answering session, so switching engines here would drop the agent's own context.
		res, err := e.client.ask(ctx, e.cfg.ChatID, prompt, session, e.cfg.Reasoning)
		if err == nil && !answerLooksLikeError(res.Answer) {
			ls, rs, perr := parseReflection(res.Answer, active, e.cfg.MaxLessons)
			if perr != nil {
				return nil, nil, perr
			}
			return assign(ls), rs, nil
		}
		if err == nil {
			err = fmt.Errorf("reflection is a provider error: %s", truncate(res.Answer, 120))
		}
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		switch classify(err) {
		case failQuota, failNetwork:
			if hist, ok := e.client.sessionHistory(ctx, e.cfg.ChatID, session); ok {
				st, ans := matchTurn(hist, prompt)
				if st == reconcileDone {
					if ls, rs, perr := parseReflection(ans, active, e.cfg.MaxLessons); perr == nil {
						return assign(ls), rs, nil
					}
				}
				if st == reconcileUnknown {
					return nil, nil, fmt.Errorf("reflection outcome unknown; abandoning lessons")
				}
			} else {
				return nil, nil, fmt.Errorf("reflection outcome unknown (no history); abandoning lessons")
			}
			if serr := e.retry.sleepClass(ctx, classify(err), attempt); serr != nil {
				return nil, nil, serr
			}
			continue
		default:
			return nil, nil, err
		}
	}
}

// judgeAnswer runs the grader on a FRESH session (§6.3: never reuse a gold-bearing one).
// It uses UtilityReasoning, not the answering engine: the benchmark's grader sends no
// "reasoning" (the regular pipeline), and a frozen evaluation is only comparable to the
// baseline when the SAME grader harness produces both numbers.
func (e *executor) judgeAnswer(ctx context.Context, rec trainsetRecord, answer string) (verdict, error) {
	prompt := graderPrompt(rec.Question, answer, rec.GoldAnswer)
	var v verdict
	err := callWithRetry(ctx, e.retry, func(attempt int) error {
		res, err := e.client.ask(ctx, e.cfg.JudgeChatID, prompt, "", e.cfg.UtilityReasoning)
		if err != nil {
			return err
		}
		if answerLooksLikeError(res.Answer) {
			return fmt.Errorf("judge reply is a provider error: %s", truncate(res.Answer, 120))
		}
		pv, perr := parseGraderVerdict(res.Answer)
		if perr != nil {
			return perr
		}
		v = pv
		return nil
	})
	return v, err
}
