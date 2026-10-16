# RSI baseline — why 99 of 150 questions failed

Analysis of the no-strategy baseline for the RSI train set
(`train/browsecomp_top150.jsonl`), run by
`tools/benchmark/rsi_baseline_conf.json` (mode 8) into
`outputs/rsi_baseline_20261015/`.

- **Result**: Accuracy **34%** (51/150), Recall 28.97%, Surfaced Recall 53.71%.
- **Sources**: `answers.jsonl` (150 rows) and `leaderboard.json`
  (`per_query_judgements` 150 rows, `per_query_metrics`).
- **Per-bucket id lists**: `train/baseline_wrong_buckets.json`
  (sha256 `d42cb184d13898f4b6b0a232bcab78f885d89a6a5a366b18bbb52706a264e816`).

## Headline

Of the 99 wrong answers, **55% are retrieval misses** and **31% had the gold
document read and still failed**. Crucially, **every expected document exists in
the corpus** (183/183, 122/122, 138/138 across the groups), so none of these
questions is unanswerable — the failures are search/synthesis, not missing data.

## Classification (n = 99)

| Bucket | n | % | Meaning |
|---|---|---|---|
| **A1** retrieval miss + gave a (wrong) answer | 29 | 29% | the gold doc never appeared in retrieval; the model still asserted an answer |
| **A2** retrieval miss + honest refusal | 25 | 25% | never found it, and said so |
| **B1** gold doc **read**, mis-synthesised | 31 | 31% | recall=1 and `gold_doc_served`, yet the conclusion is wrong |
| **B2** evidence surfaced but **not read** | 14 | 14% | the doc was in the results, but was never opened |

Retrieval-side failures (A1+A2) = **54 (55%)**; evidence-in-hand failures
(B1+B2) = **45 (45%)**.

Bucket rule: `recall` comes from `per_query_metrics`; "read" =
`gold_doc_served` non-empty; "gave an answer" = the answer text is non-empty and
contains no refusal marker.

## Three signals that matter

### 1. Failure is mistargeted effort, not giving up early

| Group | LLM calls | tool calls | tokens |
|---|---|---|---|
| Correct | 15.3 | 21.7 | 1.22M |
| Wrong (all) | **22.6** | **33.8** | **2.27M** |
| B (evidence in hand) | **25.8** | **38.1** | **2.64M** |

The wrong questions burned *more* of everything — and the ones that had the
evidence burned the most. This is thrashing, not laziness.

### 2. Wrong answers used *more* of every search tool

```
CORRECT: bm25=10.9  semantic=5.0  grep=3.2  list=2.2
WRONG  : bm25=15.7  semantic=8.5  grep=7.3  list=1.9
```

More of the same searches does not help; the searches need to change *form*
(decompose constraints, search each clue in parallel, aim at the constraint).

### 3. The grader is not the problem — 34% is real

- Mean judge confidence on the wrong rows is **96**, and **84/99 are 100** —
  the judge is certain they are wrong.
- Only 4 rows are "near misses", and each is a genuine entity difference:
  `Eder` vs `Elder`, `Professor` vs `Dr`, `in Biafra` vs `at Biafra`,
  `Batman` vs `Pirates`. These puzzle questions require exact strings.

## What this implies for the RSI strategy loop

| Bucket | n | Strategy direction |
|---|---|---|
| A1+A2 retrieval miss | **54** | **parallel multi-clue retrieval** — split the multi-constraint puzzle into independent clues and give each clue its own subagent, instead of re-issuing one query serially |
| B1 evidence read, wrong | **31** | **cross-verification** — check the candidate against *every* constraint before committing |
| B2 surfaced, not read | **14** | **depth discipline** — read the surfaced candidate document before answering |

The largest block (54) is retrieval recall, which is exactly what the
query-strategy RSI is built to move — the baseline confirms the train-set choice
is well aimed.
