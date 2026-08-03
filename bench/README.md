# Bench artefacts

`bench.py` at the repo root measures **one endpoint configuration** and appends a row to a results
file. What lives here is everything needed to reproduce a specific measurement, plus the results of
the ones already made.

The private corpus — real voice notes with hand-written references — is **not** here and never will
be; it is gitignored as `corpus/` at the repo root. That corpus is what ranks *models*. What is here
is a synthetic corpus that answers a narrower question, and does so reproducibly.

## `make-synthetic-corpus.sh`

Builds ten short French recordings with macOS `say`, encoded to WhatsApp's own shape (mono, 16 kHz,
~16 kbps Opus). Ground truth is exact because the sentences are authored — the synthesiser speaks
the reference verbatim.

Every sentence mixes two classes of proper noun: names that are also French common words (`Loup`,
`Colombe`, `Rose`, `Olivier`, `Pierre`) and controls that are not (`Grenoble`, `Chamonix`,
`Mathilde`, `Thibault`, `Fabien`, `Annecy`). Each note ships a `.bias.txt` naming the ones in it,
which is what a real client supplies from its participant list.

**What it can tell you, and what it cannot.** The speech is clean and synthetic, so an absolute WER
from it does not transfer to a note recorded in a car. It transfers for *spelling* questions, where
the model is choosing between homophones from a language prior the audio quality does not touch.
Use it for biasing; use the private corpus for models.

```sh
./bench/make-synthetic-corpus.sh ./corpus-synth
./bench.py --corpus ./corpus-synth --endpoint "$RUNPOD_TRANSCRIBE_ENDPOINT_ID" --label voxtral/none
```

## `score-names.py`

Per-name accuracy across every run in a results file — the metric WER cannot give you.

Mean WER moves for every reason at once: a name misspelled, a number written `9h` rather than
`neuf heures`, a word genuinely misheard. On short recordings each is worth several percent, so a
WER delta says *something changed*, not *the biasing worked*. This counts occurrences of each proper
noun spelled the way the reference spells it — case-insensitively on the token, exactly on the
letters, because `Loup` vs `Lou` is the entire question.

```sh
./bench/score-names.py bench-results.jsonl
```

## `results-2026-08-03-bias.jsonl`

The three biasing arms, same corpus, same day, against endpoint `3yeuhw4te12ns0`
(`mistralai/Voxtral-Small-24B-2507`, A100 80 GB).

| `WORKER_BIAS_MODE` | mean WER | `Loup` | `Thibault` | `Rose` |
|---|---|---|---|---|
| `none` | 17.36 % | 0/10 | 0/2 | 1/3 |
| `hotwords` | 17.36 % | 0/10 | 0/2 | 1/3 |
| **`chat`** | **5.24 %** | **9/10** | **2/2** | **3/3** |

🔴 **`hotwords` has no effect whatsoever.** The worker reported `bias_mode: hotwords` and sent the
terms, and all ten transcripts came back byte-identical to the `none` arm. At `temperature=0`
identical output means identical conditioning — vLLM accepts the parameter and never passes it to
Voxtral. An API accepting a field is not evidence the model received it.

⚠️ **`chat` paraphrases occasionally**, which is the risk that mode was always carrying. The output
guard never fired across the ten, and one greeting still came back reworded: *"Coucou, ici Colombe"*
→ *"Salut, c'est Colombe"*. No length or commentary signal can catch that. Watch it in production.

**A template update does not restart a running worker.** Warm workers keep the old env, so the next
arm silently re-measures the previous mode. Cycle them (`workers_max: 0` → apply → wait for zero →
`workers_max: 3` → apply), then wait for the job host, which lags and answers `409 ENDPOINT_PAUSED`.
Each run records the `bias_mode` the worker actually reported — check it before believing a result.
