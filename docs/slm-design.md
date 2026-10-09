# SLM design: "Cradle", a baby-inspired training recipe for tiny language models

Goal: train a small language model from scratch with little data (a few million words), on a normal laptop
CPU, in minutes to an hour, using a training method modeled on how infants acquire language. This doc covers
the research behind the recipe, the recipe itself, how progress is measured, and the measured results of the
prototype.

## 1. What the research says (short survey)

**How children acquire language**
- *Listening comes before talking.* Infants track transitional probabilities between syllables by about 8
  months (Saffran, Aslin & Newport 1996), and comprehension stays ahead of production for years.
- *Child-directed speech (CDS) is a curriculum in itself.* It is short, repetitive, and simple, and it grows
  more complex as the child ages. AO-CHILDES orders transcripts by the child's age, and BabyBERTa (Huebner et
  al. 2021) showed that a small model trained on about 5M words of CDS can learn a lot of grammar.
- *"Starting small."* Elman (1993) argued that a limited working memory early on helps a learner find
  structure. Later replications were mixed (Rohde & Plaut 1999), but the compute version of the idea, short
  sequences first, reliably speeds up LM pretraining (sequence-length warmup, Li et al. 2022).
- *Babbling, plus contingent feedback.* Infants whose babbling gets a contingent caregiver response produce
  more mature sounds (Goldstein & Schwade 2008). Caregivers also *recast*: they repeat the child's flawed
  utterance in corrected form, and children pick up those corrections (Chouinard & Clark 2003).
- *Sleep consolidates.* Infants who nap after exposure to an artificial language generalize its structure
  better (Gómez et al. 2006).

**What works for small LMs on small data**
- *BabyLM Challenge* (2023 and 2024, 10M- and 100M-word tracks). The winners relied on architecture and
  objective changes (LTG-BERT/ELC-BERT, and the hybrid causal+masked GPT-BERT in 2024) and on distillation:
  Baby Llama's student, distilled from an ensemble of teachers trained on the same 10M words, beat those
  teachers. Curriculum learning gave small or no consistent gains in the challenge reports (for example, the
  CLIMB submission). The lesson is that curricula alone are not a free lunch.
- *TinyStories* (Eldan & Li 2023). With a vocabulary restricted to a small child's level, models under 10M
  parameters write fluent, mostly grammatical stories. Simple data matters more than size at this scale.
- *Self-improvement.* STaR, ReST, and rejection-sampling fine-tuning have a model generate, filter with a
  reward or verifier, and train on the survivors. Training on your own outputs alone collapses (Shumailov et
  al. 2024). Accumulating generated data on top of real data avoids that (Gerstgrasser et al. 2024).
- *Self-distillation.* Born-again networks (Furlanello et al. 2018) and mean-teacher / EMA teachers
  (Tarvainen & Valpola 2017) regularize a student with soft targets from itself, which is cheap and useful
  when data is reused for many epochs.
- *Predictive coding.* Next-token prediction already is a predictive-coding objective. Richer variants such
  as multi-token prediction help large models but hurt small ones (Gloeckle et al. 2024), so they are left
  out here.
- *Architecture and tokenizer at tiny scale.* Deep-and-thin models beat wide-and-shallow ones at a fixed
  small size (MobileLLM, 2024). RoPE, RMSNorm, and tied embeddings are standard. A small vocabulary (4k-8k)
  keeps the parameter budget in the transformer instead of the embedding table. Byte-level BPE has no
  out-of-vocabulary tokens, while character-level input makes sequences about 4x longer, which is too slow
  on a CPU.

## 2. The Cradle recipe

The model is a 6-layer, 192-dimensional, 6-head decoder (3.4M parameters, about 0.8M of them in the tied
embedding table) with RoPE, RMSNorm, and a 256-token context. The tokenizer is byte-level BPE with 4,096
tokens. All text is lowercased.

**Data** (about 2.3M words, 2.9M tokens, all public): AO-CHILDES (345k words of child-directed speech in age
order) and 12,000 TinyStories (2.0M words). Held out: 1,000 stories and 1/25 of the CHILDES transcripts. The
budget is about 12M trained tokens, roughly 4 epochs. Data is scarce and reused, which is the BabyLM regime.

The phases below are fractions of a fixed compute budget. Both recipes share one AdamW optimizer and one
cosine learning-rate schedule over the whole budget.

| # | Phase (share of budget) | Data | Objective | Infant analogue |
|---|---|---|---|---|
| 1 | **Listen: CDS** (15%) | AO-CHILDES, cumulative age quartiles (youngest first), context 64 | next-token CE mixed with soft targets from a corpus **bigram model** (weight α=0.3, decaying) | statistical learning of transitional probabilities; short memory |
| 2 | **Listen: stories** (45%) | stories ranked by difficulty (length + word rarity), cumulative thirds, plus CDS; context 128→192→256 | CE with the bigram prior fading to 0 | growing input complexity and working memory |
| 3 | **Babble + feedback** (25%) | 6 rounds. Each round the model continues 1,024 caregiver prompts (512 CHILDES openings, 512 story openings) | train on 80% replayed real data + 20% feedback buffer | babbling, contingent response, recasts |
| 4 | **Sleep** (15%) | replayed real data | 0.5·CE + 0.5·KL to an **EMA copy** of the model (decay 0.998, tracked since phase 2) | consolidation during sleep |

**The caregiver (phase 3) is cheap and non-neural.**
1. *Fluency*: mean interpolated trigram log-probability under corpus statistics ("does the caregiver
   understand it?").
2. *Adaptive discriminator*: a hashed n-gram logistic regression re-fit every round to tell the model's
   babble from the true continuations of the same prompts. Its feedback therefore tracks the learner's
   *current* mistakes. It is cross-fitted on two halves, so no sample is scored by a model that saw it.
3. A *repetition* penalty (repeated trigrams), an *ending* bonus (finished the sentence), and a *too short*
   penalty.

The top 25% of babbles are **praised**: the model trains on prompt + its own continuation. For the bottom
25%, the caregiver **recasts**: a TF-IDF lookup retrieves the closest real utterance or story window, and the
model trains on that instead. The buffer keeps the last two rounds. Self-generated text is under 3% of all
trained tokens, which avoids collapse.

**What is new here.** Each piece has precedent, but the combination is new as far as I know: (a) a free
n-gram "transitional probability" teacher as an annealed soft-label prior during early listening; (b) a
self-improvement loop whose critic adapts every round to the learner's current errors; (c) recasts, meaning
retrieval-based corrective data chosen by what the model was *trying* to say, so the model's own outputs
decide which data it sees next; and (d) all of it inside one fixed compute budget against a matched baseline.

**v2 variants, added after the first ablations.**
- *Mixture-matched* (`--cds-mix 0.07 --phases 0.1 0.5 0.25 0.15`): CDS-only listening shrinks to 10%, and
  every later batch is 7% CDS rows. Total CDS exposure then matches the baseline's 17%, so only the
  ordering and objectives differ. (v1 front-loads CDS and ends up at about 35% CDS.)
- *Best-of-k babbling* (`--babble-mode best-of-k`): sample k=4 continuations per caregiver prompt, praise
  the best one for the top half of prompts, and for the quarter of prompts the learner handles worst, recast
  with the *true* continuation of that same prompt. Scores are ranked within each domain.

**Fair compute.** Trained tokens cost 3 units and forward-only tokens cost 1 (sampling, EMA-teacher passes).
Every run stops at the same unit count (36M units, 12M trained-token equivalents). The n-gram, discriminator,
and retrieval work takes seconds and is not counted. Cradle's shorter early contexts are cheaper per token,
and that saving is not credited to Cradle, so the accounting slightly favors the baseline.

## 3. How progress is measured

- **Held-out perplexity** on TinyStories (the target domain) and on CHILDES, with the same tokenizer for
  every run, so per-token losses are comparable.
- **Zorro** grammatical minimal pairs (23 paradigms, 400 pairs each, child vocabulary). Accuracy is how
  often the grammatical sentence gets the higher log-probability. Chance is 50%.
- **Generation quality** on 100 held-out story openings (96 new tokens, T=0.8, top-k 40): the perplexity of
  the continuation under **TinyStories-33M**, an independent, 10x larger judge that was trained on different
  stories; distinct-2 (diversity); and the repeated-trigram rate. The judge never sees the caregiver
  signals, so it cannot be gamed by the critic.
- Learning curves (validation loss against compute), a "developmental diary" of samples at the end of each
  phase, and per-round babble statistics.

## 4. Results

**Setup.** One 4-core cloud VM CPU (no GPU). Every run has the same budget: 36M compute units (12M
trained-token equivalents, about 4 epochs of the 2.9M-token corpus). A run takes 13-15 minutes. Wall-clock times are approximate because a few runs shared the CPU with smoke tests. The model
has 3.44M parameters. Raw numbers are in `runs/results.json`, logs in `runs/*/log.json`, and the full table
in `runs/report.md`.

### 4.1 Headline: baseline vs. Cradle variants (same compute)

Perplexity is on held-out data, so lower is better. Δ is relative to the baseline's story perplexity. Rows
with two seeds show the mean, with seed 0 / seed 1 in brackets.

| Recipe | Stories ppl | Δ stories | CHILDES ppl | Zorro acc | Judge ppl ↓ | distinct-2 | rep-3 | Wall |
|---|---|---|---|---|---|---|---|---|
| Baseline (shuffled next-token) | **11.32** (11.30 / 11.34) | — | 37.8 | 64.9% | 14.5 (13.0 / 16.0) | .374 | .034 | 13.8 min |
| Cradle v1, all 4 phases | 11.60 | +2.6% | 34.2 | 65.4% | 15.0 | .337 | .052 | 14.9 min |
| **Listen-only, natural mix** (curriculum + prior) | **11.09** (11.08 / 11.10) | **−2.0%** | **34.3** | 64.9% | 14.6 (14.5 / 14.8) | .371 | .043 | 13.4 min |
| **Listen-only, matched mix** | **10.87** (10.83 / 10.90) | **−4.0%** | 43.9 | 64.4% (62.7 / 66.0) | 12.8 (13.1 / 12.6) | .363 | .040 | 13.3 min |
| v2: matched mix + best-of-k babble | 11.04 | −2.5% | 43.6 | 61.1% | 14.0 | .372 | .036 | 14.9 min |
| Baseline + bigram prior only | 11.20 | −1.1% | 36.9 | 64.9% | 13.5 | .353 | .039 | 13.7 min |
| *reference: interpolated trigram* | *≈32* | | *≈72* | | | | | seconds |

What the table says:
- **The listening curriculum works.** Two parts drive it: (1) age-ordered CDS first, then simple-to-complex
  stories with a growing context, and (2) the annealed bigram "transitional probability" prior. With the
  natural mix it improves *both* domains, stories by about 2% and CDS by about 9%, even though it sees about
  22% fewer story tokens (35% of its tokens are CDS, against 17% for the baseline). With the matched mix, so
  the same story tokens as the baseline, story perplexity improves by about 4%. CDS then gets *worse*,
  because CDS exposure is front-loaded and mostly forgotten.
- Seed noise on perplexity is small: the two baseline seeds differ by 0.3%. The 2-4% story gains are an
  order of magnitude larger than that.
- **The self-improvement loop did not help.** Both babble variants cost perplexity: v1 pooled 11.39 → 11.60,
  and v2 best-of-k 10.83 → 11.04. Neither improved the independent judge's view of generation quality.
- **Sleep (EMA self-distillation) hurt.** Removing it took perplexity from 11.60 to 11.29 and restored
  diversity (distinct-2 from .337 to .367).
- Generation quality per the TinyStories-33M judge is a tie within noise. The baseline's judge perplexity
  moves from 13.0 to 16.0 between seeds. Re-sampling the seed-0 models with 4 sampling seeds gives 13.6±0.5
  for the baseline and 14.2±0.2 for natural-mix listen-only, with identical repetition rates (0.040 vs
  0.041). The matched-mix model scored better than the baseline mean on both seeds (13.1 and 12.6 vs 14.5),
  which is suggestive but not conclusive.
- Zorro grammar is 61-66% for every model, and the differences are within noise. The same matched-mix
  recipe scores 62.7% and 66.0% on two seeds, and single paradigms swing by up to 50 points between runs.
  Zorro does not separate the recipes at this scale.

### 4.2 Ablations of Cradle v1 (remove one component at a time, seed 0)

| Variant | Stories ppl | Change vs full v1 | Reading |
|---|---|---|---|
| Full v1 | 11.60 | — | |
| − curriculum (shuffled data, keeps prior/babble/sleep) | 11.85 | +0.25 | curriculum helps |
| − bigram prior | 12.00 | +0.40 | prior helps (most of it during the CDS-only phase) |
| − babble | 11.39 | −0.21 | babble loop hurts |
| − sleep | 11.29 | −0.31 | EMA distillation hurts |
| − babble − sleep (= listen-only) | 11.08 | −0.52 | the two harmful parts add up |

Learning curves (story validation loss on a fixed 48-window subset vs. share of budget) show where the gains
come from. The curriculum runs start behind, because they spend phase 1 on CDS only. They overtake the
baseline by 60%: matched mix 2.295, natural mix 2.351, baseline 2.369. Plain replay keeps the lead, and
the babble and sleep phases erode it.

| Run | 15% | 30% | 60% | 85% | end |
|---|---|---|---|---|---|
| Baseline | 3.20 | 2.77 | 2.37 | 2.21 | 2.18 |
| Listen-only, natural mix | 4.54 | 2.90 | 2.35 | 2.19 | 2.16 |
| Listen-only, matched mix | 3.28 | 2.74 | **2.30** | **2.16** | **2.12** |
| Cradle v1 (babble + sleep) | 4.54 | 2.90 | 2.35 | 2.22 | 2.20 |

### 4.3 Developmental diary (Cradle v1, seed 0, same prompts at the end of each phase)

| After phase | "look at the…" | "once upon a time, there was a little…" |
|---|---|---|
| 1 listen (CDS) | look at the other kids and. the park in the road. oh? | …little boy named and there was it. |
| 2 listen (stories) | …saw a big butterfly. the butterfly was scared and cried the sound. the butterfly flew away… | …little girl called jane. she was only three years old and she was always looking for a great time. |
| 3 babble | look at the big truck and saw the fire truck in the fire truck? … | …little girl called anna. she loved to explore and explore the world around her… |
| 4 sleep | look at the big red ball. the purple ball is looking at it. it is not very fun. | …little girl named lily. she loved to play outside in the sun. one day, she saw a little bird on a tree… |

The listen-only model in chat (`python -m slm.chat --prompt "the girl was sad because"`):

> the girl was sad because she couldn't play with the toys. she had to go to the store and have to stay inside. the end.

A held-out story continuation (baseline vs. listen-only, same prompt, seed 0) is in `runs/report.md`. Both
are fluent at the sentence level and drift at the plot level. That is typical for a model of about 3M
parameters.

### 4.4 Why the "talking" phases did not help (honest diagnosis)

1. **The caregiver knows nothing the learner doesn't.** Every critic signal (trigram fluency, the
   discriminator, retrieval) comes from the same 2.9M tokens the learner already trains on. A
   self-improvement loop can only add information when the verifier knows something the learner does not.
   STaR has answer checking. A real baby has a far more competent adult. Here, praising its own samples
   just displaces real data with worse data.
2. **The critic is weak.** The adaptive discriminator separates babble from real speech about 80% of the
   time, but many "worst" samples are fine ("yeah she's a doctor."). Ranking noise means some good data is
   recast and some bad data is praised.
3. **Sleep used a lagging teacher.** With decay 0.998 (about 500 steps) in a 1,465-step run, the EMA teacher
   is *worse* than the student late in training. It also costs a third more compute per token in that
   phase.
4. The loop is cheap (generation used about 1% of the budget), so the cost is not compute. It is the
   quality of the replacement data.

### 4.5 Recommended recipe today

Use **listen-only Cradle** (`--recipe cradle --no-babble --no-sleep`). Add `--cds-mix 0.07 --phases 0.1 0.5
0.25 0.15` if story quality is all you care about. It takes the same 13-14 minutes on a laptop-class CPU,
is 2-4% better than the baseline on held-out stories, and needs no extra dependencies.

## 5. Next experiments

1. **A smarter caregiver.** Use a stronger frozen model (TinyStories-33M, or an LLM via API) as the critic
   for best-of-k and recasts, and judge with a *different* model or with human ratings. This is the
   cleanest test of whether the babble loop helps once the feedback carries new information.
2. **Preference learning instead of imitation.** Run DPO or unlikelihood on best-vs-worst pairs from the
   same prompt, instead of training on praised samples. This targets generation quality (repetition,
   coherence), which perplexity does not measure.
3. **Spaced repetition for CDS.** Replace the front-loaded CDS phase with a smoothly decaying CDS share
   (for example 100% → 10%) to keep matched-mix's story gain without forgetting CDS.
4. **Better priors.** Try a Kneser-Ney trigram prior, build the phase-1 prior from CDS only (this removes the
   current leak of story statistics into phase 1), and tune α and the decay.
5. **Cheaper "sleep."** Average the last N checkpoints' weights (LAWA) instead of KL distillation, which
   costs no extra compute, or use an EMA with a shorter horizon (decay 0.99).
6. **Scale checks.** Run 3+ seeds, a 50M-token budget, and a 10M-parameter model to see whether the 2-4%
   curriculum gain grows, holds, or vanishes, since BabyLM reports curricula fading at scale. Also try the
   GPT-BERT-style hybrid masked + causal objective as a richer "listening" objective.
