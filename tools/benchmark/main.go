// Command benchmark is the Go port of scripts/ragflow_benchmark.py: a
// dataset-agnostic QA benchmark runner for RAGFlow.
//
// How a run works:
//   - One question loader - .jsonl, JSON list, or JSON {id: {...}} mapping.
//   - An answer phase (a RAGFlow chat answers each question) followed by a judge
//     phase (a SEPARATE judge chat scores each answer), both resumable and both
//     parallelisable via concurrency.
//   - One judge scorer: the 0/2/4 rule book first, tolerant fallbacks after.
//   - One result artefact: leaderboard.json (the BrowseComp-Plus submission JSON
//     plus per-question judgement/usage extensions) carries everything a run
//     produces; answers.jsonl stays a pure record of what the agent returned.
//
// Reliability semantics carried over from the Python original:
//   - A question counts as finished when its row carries an answer and no
//     ragflow_error, OR when its error is PERSISTENT (the provider refused the
//     input itself - content policy, e.g. minimax's "input new_sensitive"; a
//     retry reproduces the refusal). Transient errors never write a damaged
//     row: the question parks ERROR_RETRY_WAIT_SEC (30 min) in process and
//     retries until it succeeds, so no external watcher is needed.
//   - A provider quota wall is ridden out in process: the phase stops
//     dispatching, waits --wall-wait seconds, then re-enters itself and
//     retries exactly the work still missing. Unbounded unless
//     --wall-max-waits N is given.
//   - Skipping finished work IS the resume: repeating the same command
//     continues the run directory; --overwrite discards it. Each run records
//     its git commit into _prompt_fingerprint.json once.
//
// Usage:
//
//	benchmark --config <conf.json> [--ids 11,33] [--dry-run]
//	benchmark --config <conf.json> --run outputs/<dir> --skip-answers
//	benchmark --config <conf.json>            # re-run a batch: same command
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultConfigPath  = "qa_benchmark_conf.json"
	defaultOutputDir   = "outputs/qa_benchmark_<timestamp>"
	defaultAnswersName = "answers.jsonl"
	defaultLeaderboard = "leaderboard.json"
	fingerprintName    = "_prompt_fingerprint.json"
)

func main() {
	os.Exit(run())
}

func run() int {
	fs := flag.NewFlagSet("benchmark", flag.ContinueOnError)
	configPath := fs.String("config", defaultConfigPath, "Path to the benchmark config JSON")
	concurrency := fs.Int("concurrency", 0, "Questions answered (and judged) in parallel; overrides the config's concurrency (default 1 = serial)")
	idsFlag := fs.String("ids", "", "Comma-separated question ids; overrides question_ids.include from the config")
	runDir := fs.String("run", "", "Reuse an existing output directory instead of creating a new one")
	overwrite := fs.Bool("overwrite", false, "Drop existing JSONL files instead of resuming")
	dryRun := fs.Bool("dry-run", false, "Print the plan without calling any API")
	skipAnswers := fs.Bool("skip-answers", false, "Judge an existing answers.jsonl")
	skipJudge := fs.Bool("skip-judge", false, "Only collect answers")
	wallWait := fs.Float64("wall-wait", wallRetryWaitSec, "Seconds to sleep when the provider Token Plan is exhausted, before retrying the pending work in process")
	wallMaxWaitsFlag := fs.Int("wall-max-waits", wallMaxWaits, "Give up after this many plan-wall waits instead of waiting indefinitely (0 = unlimited; a bounded run exits 1)")
	if err := fs.Parse(os.Args[1:]); err != nil {
		return 2
	}

	cfg, err := loadJSONMap(*configPath)
	if err != nil {
		fmt.Println(err)
		return 2
	}
	baseDir := filepath.Dir(*configPath)

	outputCfg, _ := cfg["output"].(map[string]any)
	var outputDir string
	if *runDir != "" {
		outputDir = absPath(*runDir)
	} else {
		outputDir = resolveOutputDir(strAny(outputCfg, "output_dir", defaultOutputDir))
	}
	answersPath := filepath.Join(outputDir, strAny(outputCfg, "answers_jsonl", defaultAnswersName))
	leaderboardPath := filepath.Join(outputDir, strAny(outputCfg, "leaderboard_json", defaultLeaderboard))

	// CLI --ids replaces question_ids.include for this run; everything else in
	// the question_ids block (random is moot, exclude still applies) is kept.
	// Precedence: --ids > include_file > include.
	selection, _ := cfg["question_ids"].(map[string]any)
	if selection == nil {
		selection = map[string]any{}
	}
	includeFile := strings.TrimSpace(strAny(selection, "include_file", ""))
	if includeFile != "" && *idsFlag == "" {
		fileIDs := readIDList(resolvePath(includeFile, baseDir))
		if len(fileIDs) == 0 {
			fmt.Printf("question_ids.include_file is missing or empty: %s\n", resolvePath(includeFile, baseDir))
			return 2
		}
		selection["include"] = toAnySlice(fileIDs)
		fmt.Printf("[ids] %d id(s) from %s\n", len(fileIDs), resolvePath(includeFile, baseDir))
	}
	if *idsFlag != "" {
		var parts []any
		for _, part := range strings.Split(*idsFlag, ",") {
			if part = strings.TrimSpace(part); part != "" {
				parts = append(parts, part)
			}
		}
		selection["include"] = parts
	}

	if fileExists(answersPath) && !*overwrite {
		existing := dedupeLast(readJSONL(answersPath))
		answered, persistent, transient := resumeSplit(existing)
		judged := countSettledJudgements(readJudgements(leaderboardPath))
		fmt.Printf("[resume] %s: %d row(s) - %d answered + %d persistent-error (skipped), %d transient failed/aborted will be retried; %d verdict(s) already stored\n",
			filepath.Base(answersPath), len(existing), answered, persistent, transient, judged)
	}

	conc := *concurrency
	if conc == 0 {
		conc = asIntDefault(cfg["concurrency"], 1)
	}
	if conc < 1 {
		conc = 1
	}

	var selected []*question
	if *skipAnswers {
		if !fileExists(answersPath) {
			fmt.Printf("--skip-answers needs an existing answers file: %s\n", answersPath)
			return 2
		}
		fmt.Printf("Reusing answers file %s\n", answersPath)
	} else {
		questionsPath := resolvePath(questionsPathOf(cfg), baseDir)
		questions, err := loadQuestions(questionsPath)
		if err != nil {
			fmt.Println(err)
			return 2
		}
		selected, err = selectQuestions(questions, selection)
		if err != nil {
			fmt.Println(err)
			return 2
		}
		fmt.Printf("Loaded %d questions from %s\n", len(questions), questionsPath)
		if dataset, _ := cfg["dataset"].(map[string]any); dataset != nil {
			fmt.Printf("Dataset: %s - %s\n", strAny(dataset, "name", "(unnamed)"), strAny(dataset, "description", "(no description)"))
			if cp := strAny(dataset, "corpus_path", ""); cp != "" {
				fmt.Printf("Corpus: %s\n", cp)
			}
		}
		fmt.Printf("Selected %d question(s) (concurrency=%d)\n", len(selected), conc)
	}
	fmt.Printf("Output directory: %s\n", outputDir)

	if *dryRun {
		for i, q := range selected {
			if i >= 10 {
				break
			}
			fmt.Printf("- %s: %.100s\n", q.questionID, q.question)
		}
		if len(selected) > 10 {
			fmt.Printf("... %d more\n", len(selected)-10)
		}
		return 0
	}

	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		fmt.Println(err)
		return 1
	}

	// Provenance before any phase: the fingerprint carries the commit of the
	// run that produced the rows, so --overwrite must drop a stale one, while
	// a resume keeps the original.
	fingerprintPath := filepath.Join(outputDir, fingerprintName)
	if *overwrite && fileExists(fingerprintPath) {
		os.Remove(fingerprintPath)
	}
	recordRunProvenance(outputDir)

	if !*skipAnswers {
		if *overwrite && fileExists(answersPath) {
			os.Remove(answersPath)
		}
		answerAborted := runAnswerPhase(newClient(cfg), cfg, selected, answersPath, conc,
			time.Duration(*wallWait*float64(time.Second)), *wallMaxWaitsFlag)
		if answerAborted {
			fmt.Printf("Run gave up waiting for the plan at the answer phase; %s holds the completed rows.\n", answersPath)
			return 1
		}
		if *skipJudge {
			fmt.Printf("Answers written to %s\n", answersPath)
			return 0
		}
	}

	judgements, judgeAborted := runJudgePhase(newClient(cfg), cfg, answersPath, leaderboardPath, conc,
		time.Duration(*wallWait*float64(time.Second)), *wallMaxWaitsFlag)

	// The leaderboard export scores retrieval recall against each question's
	// EVIDENCE documents, which live in the questions file rather than in the
	// answers - an answers.jsonl from an older run predates the field, and
	// --skip-answers never re-reads the questions at all.
	allRows := dedupeLast(readJSONL(answersPath))
	evidenceByID := map[string][]string{}
	if p := questionsPathOf(cfg); p != "" {
		evidenceByID = loadEvidenceDocIDs(resolvePath(p, baseDir))
	}
	merged := mergeJudgements(allRows, judgements)
	leaderboard := buildLeaderboard(merged, cfg, evidenceByID)
	writeJSON(leaderboardPath, leaderboard)

	if judgeAborted {
		fmt.Println("Run gave up waiting for the plan at the judge phase; re-run the same command to continue.")
		return 1
	}
	fmt.Printf("Leaderboard submission written to %s\n", leaderboardPath)
	fmt.Println(leaderboardSummary(leaderboard))
	return 0
}
