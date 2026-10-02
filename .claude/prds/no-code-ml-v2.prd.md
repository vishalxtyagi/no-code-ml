# pocketml — "Replace your LLM call with a model you own, trained for free"
<!-- Working title. Product name TBD (see Open Questions). Supersedes the earlier CSV-only AutoML draft. -->

## Problem
Developers and small product teams increasingly send **repetitive prediction work** to a frontier LLM API. Typical jobs:
- tagging support tickets
- extracting fields from documents
- scoring leads from a table
- routing messages
- classifying images or rows

It works on day one. At volume, they pay per call forever. Every prediction takes hundreds of milliseconds to seconds, sends their data to a third party, and can drift whenever the provider changes the model.

For most of these tasks a small specialized model would do as well or better, faster, cheaper, offline, and deterministically. But building one still needs ML skill, labeled data, a GPU, and an evaluation setup they don't have. So they keep paying the LLM, or never ship the feature.

The current no-code-ml repo doesn't address any of this. On `main` it is a single-user, regression-only Flask demo. The unmerged rewrite on `origin/copilot/add-best-model-insights-feature` is a generic tabular AutoML demo in a crowded category.

## Evidence
**Observed — desk research (market facts, not proof of user pain):**
- **Small fine-tuned models often beat zero-shot LLMs on narrow tasks at a fraction of the cost.** Classification and extraction studies and vendor write-ups claim 10–30× cheaper inference and 50–200 ms latency vs 300 ms–2 s.
  - These numbers are **unverified vendor/blog claims, not targets.**
  - Sources: [futureagi SLM vs LLM](https://futureagi.com/blog/comparison-slm-llm-language-models/), [Trained Miniatures (arXiv)](https://arxiv.org/pdf/2508.15617), [distil labs benchmarks](https://www.distillabs.ai/blog/distil-labs-benchmarking-the-platform/)
- **Foundation models now predict on many data types with no training at all:**
  - Tabular: TabPFN-3 handles up to ~1M rows and beats tuned AutoML on TabArena ([TabPFN-3 report](https://arxiv.org/pdf/2605.13986)).
  - Time series: Chronos-2 forecasts zero-shot ([MLM overview](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)).
  - Mixed tabular, text and image data: open-source multimodal AutoML now bundles these models ([AutoGluon 1.6](https://auto.gluon.ai/stable/whats_new/index.html)).
  - "Training is costly" is often no longer true. For many tasks the first good model needs zero training.
- **Free compute is real but user-owned:**
  - Google shipped an official open-source Colab MCP server (Mar 2026). It lets a *local* agent run code in the user's *own* browser Colab session ([Google blog](https://developers.googleblog.com/announcing-the-colab-mcp-server-connect-any-ai-agent-to-google-colab/), [repo](https://github.com/googlecolab/colab-mcp)).
  - Colab's free tier restricts workloads "operated mainly through another web interface" ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
  - Kaggle gives 30 GPU-h/week, scriptable via its API ([Kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage)).
  - Modal gives $30/mo of credits; HF ZeroGPU gives about 5 min/day ([Spheron roundup](https://www.spheron.network/blog/free-gpu-cloud-credits-2026/)).
  - Free LoRA fine-tuning of small LMs on a T4 is routine ([Unsloth notebooks](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/ModernBert.ipynb)).
- **Trained models can run inside a browser tab** (inference only) via WebGPU/WASM runtimes ([Transformers.js v4](https://www.developersdigest.tech/blog/transformers-js-guide)).
- **LLM data-analysis tools are non-reproducible and weak at real ML.** Re-running the same question can give different results ([Julius AI review](https://mcpanalytics.ai/articles/julius-ai-review)).
- **Competitors exist and the space is consolidating:**
  - [distil labs](https://www.distillabs.ai/docs/getting-started/overview/) turns a prompt plus a few dozen examples into a small LM (text tasks, hosted).
  - [Plexe](https://github.com/plexe-ai/plexe) (YC, open source) turns a prompt into an ML model (structured data).
  - OpenPipe was acquired by CoreWeave and Predibase by Rubrik, both in 2025 ([CoreWeave](https://www.coreweave.com/news/coreweave-to-acquire-openpipe-leader-in-reinforcement-learning), [Rubrik](https://www.rubrik.com/company/newsroom/press-releases/25/rubrik-to-acquire-predibase-to-accelerate-agentic-ai-adoption)).

**Observed — codebase audit:**
- `main` keeps all state in one process-global object, handles regression only, and has no validity checks or reproducible export. Dependencies date from 2023.

**Assumptions — need validation:**
- Assumption — needs validation via 10 interviews with developers who ship LLM-backed features: is per-call cost, latency or privacy of a *repetitive* LLM task a felt pain, and would they swap in an owned model?
- Assumption — needs validation via a competitive teardown (distil labs, Plexe, AutoGluon, Unsloth Studio): is "local-first + free bring-your-own compute + open source + any-data + browser-runnable output" actually unserved, or just unmarketed?
- Assumption — needs validation via a prototype benchmark: on real user tasks, does a small owned model come close enough to the teacher LLM to be worth switching?

## Users
- **Primary**: A developer, indie hacker, or engineer on a small product team who:
  - **is calling, or would call, an LLM API for a repetitive prediction task** on text, tables, or mixed rows (classify, extract, score, route)
  - is comfortable in a terminal or an AI coding agent
  - **Trigger:** an LLM bill, latency or privacy complaint, or a feature blocked by per-call cost.
- **Secondary**: Data-literate analysts with a labeled table who want a validated model they own. They're served by the same tabular path but not designed for first.
- **Not for**:
  - Teams that need a fully hosted, zero-install SaaS. The free-compute promise depends on the product running on the user's side.
  - Enterprise MLOps buyers (governance, feature stores, registries).
  - People who want a chatbot or a generative model. Unsloth and others already serve them well.
  - Researchers who need custom architectures.

## Hypothesis
We believe **a local-first, open-source tool that turns (a) the user's existing LLM prompt or (b) their own data of any supported type into a small owned model** will **let developers cut repetitive LLM calls without ML expertise or paid compute** for **developers shipping LLM-backed prediction features**. It works like this:
- It tries foundation models with **no training** first.
- It trains or distills only when that wins, on compute the user already has for free.
- It proves the result on the user's own evaluation set.
- It exports something that runs anywhere, including a browser tab.

We'll know we're right when:
1. **On TBD % of user tasks, the owned model is within TBD points of the teacher LLM** on the user's held-out set, at **≤ 1/TBD the cost per prediction**.
2. **TBD % of first-time users reach a validated, exported model within 30 minutes** without paying for compute.

(TBD values need a baseline from the prototype benchmark.)

**Why pick this over the alternatives (each must hold, or the pitch fails):**
- **Over the frontier LLM:** cheaper, faster and private per prediction, deterministic, and runs offline.
- **Over distil labs, Plexe and other hosted prompt-to-model services:**
  - it's free to run on your own compute
  - it's open source and local-first, so your data never leaves your machine or your Colab
  - it covers text *and* tabular *and* mixed data in one flow
  - the output is yours: a re-runnable notebook plus a portable model
- **Over a notebook with AutoML libraries:** no ML skill needed, validity checks built in, and the zero-training-first logic means most tasks need no GPU at all.

## Success Metrics
| Metric | Target | How measured |
|---|---|---|
| Teacher parity: owned-model score vs teacher LLM on the user's held-out set | Within TBD pts on TBD % of tasks | Built-in eval report, aggregated from opt-in telemetry plus an internal benchmark suite |
| Cost-per-prediction ratio (owned model vs teacher LLM) | ≤ 1/TBD | Measured per run: the LLM's metered cost vs the model's measured inference cost |
| Zero-training sufficiency: tasks where a no-training foundation-model baseline is accepted as final | TBD % (tracked, not targeted) | Run logs |
| Time from start to validated exported model (first-time user, median) | ≤ 30 min | Opt-in telemetry |
| Operator compute cost per user run | $0 | By construction: all compute is the user's. Verify via infra billing. |
| Reproducibility: exported notebooks that re-run and match the reported metric within tolerance | ≥ 95% | Automated re-run of a benchmark sample |
| Validity catch rate on seeded leakage or proxy problems | ≥ 90% | Internal benchmark with planted issues |
| Launch traction (milestone signal, not success) | TBD | GitHub stars, HN/PH rank, week-2 return of installers |

## Scope
**MVP**: one local-first tool, usable both on its own and from the user's AI coding agent, for **two input modes on two data types**.
- **Input mode A, "Replace my prompt":**
  - The user supplies an existing LLM prompt, a few examples and optionally unlabeled inputs.
  - The tool uses an LLM of the user's choice to label and synthesize training data, then produces a small model for the task.
- **Input mode B, "Learn from my data":** the user supplies labeled data.
- **Data types:** **tabular** (including columns of free text) and **plain text**, for classification, extraction into fixed fields, and regression.
- **Zero-training first:** foundation-model or few-shot baselines are evaluated before any training. Training or distillation runs only if it beats them.
- **Validity report before any score:** leakage, target-proxy columns, imbalance, contamination, too-small data, and teacher-label noise.
- **Proof:** a side-by-side report of owned model vs teacher LLM covering accuracy, cost per 1k predictions, and latency, on the user's held-out data.
- **Free compute, user-owned:**
  - runs on the user's machine (CPU first-class)
  - can use the user's own free notebook GPU session when needed
  - no operator-hosted compute
- **Own it:** a re-runnable notebook, a portable model artifact, and a local prediction endpoint. For text models, the artifact can also run in a browser tab.

**Out of scope**
- Images and documents/PDFs: next milestone, once the core loop is proven on tabular and text.
- Time-series forecasting: a later milestone. It needs its own validation semantics.
- Audio and video: deferred until demand is shown.
- Generative or chat-model fine-tuning: already well served (e.g. Unsloth). Out even if users ask.
- Hosted SaaS with operator-paid compute: it breaks the $0-compute promise and the local-first privacy story.
- Teams, auth, billing and a model marketplace: nothing to monetize before parity and activation are proven.
- MLOps (drift monitoring, scheduled retraining, registry): enterprise territory.
- Automating free GPU providers in ways their terms forbid, including multi-account juggling. This is a hard line.

## Delivery Milestones
<!-- Business outcomes, not engineering tasks. /plan turns each into a plan. -->
<!-- Status: pending | in-progress | complete -->

| # | Milestone | Outcome | Status | Plan |
|---|---|---|---|---|
| 1 | Trustworthy tabular+text core | A user points the tool at a table or text dataset and gets a validity report, a zero-training baseline, and (only if better) a trained model, all on their own machine and with several runs not interfering. | complete | `.claude/plans/m1-core.plan.md` |
| 2 | Replace my prompt | A user hands over an existing LLM prompt and a few examples and gets a small model plus a teacher-vs-owned report covering accuracy, cost and latency. | pending | — |
| 3 | Own and run anywhere | Every result exports as a re-runnable notebook, a portable model, a local endpoint, and (for text) a browser-runnable build with a shareable demo page. | pending | — |
| 4 | Free GPU, legitimately | Heavy steps can run on the user's own free notebook GPU session from their agent or the tool, within the providers' terms, with CPU fallback. | pending | — |
| 5 | More data types | Images and documents, then time series, through the same zero-training-first flow. | pending | — |
| 6 | Launch | Open-source release, a "your $X/month LLM call → a 30 MB model" demo on public datasets, and launches on HN, Product Hunt and others. Judged on parity and activation metrics, not rank. | pending | — |

## Open Questions
- [x] **Kaggle API for third-party tools.** *Resolved 2026-10-03.*
  - A **local** tool running with the user's own Kaggle credentials (`kaggle auth login` or an API token) is ordinary first-party API use.
  - A **hosted** product acting for users would need Kaggle's OAuth 2.0 + PKCE flow, with a client ID issued by the Kaggle team.
  - Scraping or crawling is prohibited ([Kaggle API docs](https://www.kaggle.com/docs/api), [kaggle-cli auth](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)).
  - → Milestone 4 stays local-first. Request an OAuth client only if a hosted mode is ever built.
- [x] **Teacher LLM terms.** *Resolved 2026-10-03; re-check before launch.*
  - **Anthropic** explicitly **permits** training non-competing task-specific models: classification, extraction, sentiment, categorization, semantic search, anomaly detection. It forbids chatbots and open-ended generation ([Claude support](https://support.claude.com/en/articles/12326764-can-i-use-my-outputs-to-train-an-ai-model)).
  - **OpenAI** forbids competing models, with a "Permitted Exception" for classifiers and embeddings *not distributed or made commercially available to third parties*. Per the [Services Agreement](https://openai.com/policies/services-agreement/); the page returned 403 to automated fetch, so verify the wording by hand.
  - **Gemini API** forbids using the Services "to develop models that compete with the Services" ([Gemini API terms](https://ai.google.dev/gemini-api/terms)). It's unclear whether a replacement classifier counts.
  - → Milestone 2 scope:
    1. Restrict "replace my prompt" to classification and extraction tasks. Never generative or chat.
    2. Default the teacher to an **open-weight model** (Apache-2.0-class licence), with no output restrictions.
    3. Show a per-provider terms notice when the user picks a hosted teacher. Models trained from OpenAI outputs are flagged "internal use only, don't redistribute".
- [ ] **Colab path** (blocks milestone 4; **needs:** a prototype): the official Colab MCP server needs a local agent and the user's open browser session, and supports specific agent clients. Is that UX acceptable for milestone 4, or is "open this notebook in Colab with one click" enough? Validate with a prototype.
- [x] **Licensing of the best no-training models.** *Resolved by milestone 1.* TabICLv2 is BSD-3 and free for commercial use ([repo](https://github.com/soda-inria/tabicl)), and it is used as tier 1. On Titanic it beat trained gradient boosting (balanced accuracy 0.808 vs 0.793). TabPFN stays excluded. The original question was: the strongest tabular foundation model's open weights are non-commercial ([Prior Labs](https://docs.priorlabs.ai/models)). Which permissively licensed alternatives are good enough? Does the product's own license allow commercial users? Validate via a license review and a benchmark.
- [x] **Teacher LLM cost:** the user brings their own key or runs an open-weight teacher locally, so the operator pays nothing. For the terms, see the resolved item above.
- [ ] **Differentiation vs distil labs and Plexe** (**needs:** user interviews, which can't be done from here): is local-first plus free compute plus any-data enough of an edge, or do we need a sharper wedge such as browser-runnable models? Validate via a competitive teardown and user interviews.
- [x] **Minimum data.** *Resolved for milestone 1:* the tool warns below 50 rows or below 10 per class, and refuses fewer than 2 per class (or fewer than 10 rows for regression). The upper size limit is still unmeasured: TabICL took about 2 min on CPU for 891 rows. Benchmark larger tables before launch.
- [ ] **Interface.** The CLI shipped in milestone 1. Agent (MCP) and local web UI are still open. **Needs:** prototype tests with 5+ target users.
- [x] **Product name.** *Resolved 2026-10-03:* **pocketml** (free on PyPI at the time). Package and CLI renamed. The repo name `no-code-ml` can be renamed on GitHub later.
- [x] **Baseline.** *Resolved:* started fresh as a local CLI. The Copilot branch was reference only.

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Seen as "distil labs / Plexe but worse": hosted competitors are funded and polished | High | High | Lead with what they structurally can't offer (free, local, open source, any-data, runs in a browser). Run a teardown before milestone 2. |
| Free GPU providers restrict or change free-tier use ([Colab FAQ](https://research.google.com/colaboratory/faq.html)) | Med | High | CPU-first and zero-training-first so the GPU is optional. Only use paths providers sanction. Never juggle accounts. |
| Owned models fail to match the teacher on real tasks, so the core claim fails | Med | High | Always show the honest parity report. Recommend "keep the LLM" when the gap is too large. |
| Best foundation models carry non-commercial licenses | High | Med | Default to permissive models. Make licensed models an explicit opt-in with a warning. |
| LLM provider terms restrict training on outputs | Med | High | Open question above. Allow open-weight teacher models as the default fallback. |
| Scope creep toward "any data, any task" before the core loop works | High | High | The MVP is two data types. The out-of-scope list is binding until milestone 2 metrics are hit. |
| Local install friction kills activation | Med | Med | One-command install and a sample task that completes in under 5 min on a laptop CPU. |

---
*Status: DRAFT — requirements only. Implementation planning pending via /plan.*
