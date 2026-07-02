# How to use Claude (Code + Cowork) to the max for this project

This is your operating manual for getting the most out of Claude across the 5 weeks.
Read it once now, then keep it open in Week 1.

---

## 1. The stack (and why)

| Layer | Choice | Why |
|-------|--------|-----|
| Language | **Python 3.11** | the ML ecosystem; you already know it |
| Env | **venv + `requirements.txt`** (or `conda`) | simple, reproducible, no surprises |
| Data | **pandas + pyarrow** | tidy frames, fast parquet I/O |
| Classical ML | **scikit-learn** | clustering, IsolationForest/OCSVM, metrics |
| Deep learning | **PyTorch (CPU)** | you know it; data is tiny so no GPU needed |
| Dim-reduction / viz | **UMAP / PCA + matplotlib** | latent visualisation, paper figures |
| Clustering extra | **hdbscan** | density clusters without fixing k |
| Notebooks | **JupyterLab** | exploration; keep heavy logic in `src/dga` |
| Paper | **LaTeX (IEEEtran) + Zotero/BibTeX** | required IEEE format; manage refs properly |
| Config | **YAML (`config/default.yaml`)** | run experiments by editing config, not code |
| Versioning | **git** (+ GitHub/GitLab) | history, tags at each milestone, backup |

Everything above is already wired into the scaffold and verified to run. Install the
PyTorch **CPU** wheel (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) —
it's much smaller and plenty for ~4.5k samples.

---

## 2. Three Claudes — when to use which

- **Claude Code (terminal/VS Code)** → your **main driver**. It sees the whole repo,
  edits files, runs the pipeline, debugs tracebacks, writes LaTeX. Use it for ~90% of work.
- **Cowork (this desktop app)** → file-level tasks and **document deliverables**: turning
  results into a polished report/slides, working with the raw Excel, quick visualisations.
- **claude.ai (chat)** → quick thinking on the move: explain a paper, sketch a proof,
  brainstorm an experiment — when you're away from the repo.

They share the same brain; the difference is how much of your project they can see and touch.
The `.md` files below make all three instantly productive.

---

## 3. Custom `.md` files — yes, and here's the system

This is the highest-leverage habit. Markdown files are how you give Claude durable memory
and context. Your repo already has them; keep them alive.

| File | Purpose | Who reads it | Update cadence |
|------|---------|--------------|----------------|
| `CLAUDE.md` | project context Claude Code auto-loads every session | Claude (auto) | when structure/conventions change |
| `docs/PROJECT_PLAN.md` | the 5-week plan, milestones, risks | you + Claude | weekly |
| `docs/methodology.md` | decisions & rationale (why log1p, why this latent dim) | you + Claude | when you make a methodological choice |
| `docs/literature.md` | annotated bibliography (1 line/paper) | you + Claude | as you read |
| `docs/lab_notebook.md` | daily research log (what/why/result/next) | you (your record) | daily, 5 min |
| `.claude/commands/*.md` | reusable slash commands (see §4) | Claude Code | once, then refine |

**Rules of thumb:**
- Keep `CLAUDE.md` short — it's loaded every time; bloat costs you context. Link out to the rest.
- Put *decisions and rationale* in `methodology.md` so "why did we do X?" always has an answer.
- When you start a session: tell Claude Code *"read CLAUDE.md and docs/PROJECT_PLAN.md, we're on Week N, today's goal is …"*.

---

## 4. Claude Code workflow essentials

1. **`/init`** — already effectively done (you have a `CLAUDE.md`). Re-run if structure changes a lot.
2. **Plan first, then code.** For anything non-trivial, ask for a plan and approve it before edits:
   *"Plan how to add per-sample gas-proportion normalisation as a config option. Don't write code yet."*
   Use plan/think mode for design; let it implement only once the plan is right.
3. **The tight loop:** ask → it edits + runs → read the traceback/result → correct. Always have it
   **run** what it writes (`python scripts/…`), don't trust code by eye.
4. **Custom slash commands** — drop a markdown file in `.claude/commands/` and it becomes `/name`.
   Three are pre-made for you (`/new-experiment`, `/review-paper`, `/lab-log`). Example body:
   ```md
   # .claude/commands/review-paper.md
   Act as a critical IEEE "Reviewer 2". Read paper/sections/*.tex. For each section list:
   weaknesses, unsupported claims, missing baselines/citations, and clarity fixes.
   Be specific and harsh; do NOT rewrite the text, just critique.
   ```
5. **Subagents** — parallelise read-heavy work. e.g. *"Spawn an agent to read all notebooks and
   summarise which experiments are done vs missing."* Great for literature triage and repo audits.
6. **MCP servers (optional)** — connect tools you actually use: a reference manager (Zotero),
   Google Drive/Slack for sharing, arXiv search. Add only what saves real time.
7. **Skills** — Cowork has document skills (docx/pptx/pdf/xlsx). Use them for the **report and
   slides** at the end, and for any spreadsheet wrangling.

---

## 5. The weekly loop with Claude

Each week, in this order:

1. **Monday kickoff:** *"Read CLAUDE.md + PROJECT_PLAN.md. We're starting Week N. List today's
   concrete tasks and the Definition of Done."*
2. **Build:** plan → implement → run → verify, one task at a time. Prefer config edits for experiments.
3. **After each real result:** *"Append a dated entry to docs/lab_notebook.md: what I ran, the
   numbers, and what it implies."* (You keep ownership; Claude just formats.)
4. **Friday wrap:** *"Summarise this week vs the plan's milestone, update the checklist in
   PROJECT_PLAN.md, and draft my 1-page progress note for Dr. Phadungthin."*

---

## 6. Research-specific prompting patterns

- **Literature triage:** paste an abstract → *"In 3 lines: problem, method, dataset, limitation.
  Which of my three buckets does it fit? Is it worth a full read?"* Then you read the keepers.
- **Math you must own:** *"Derive the VAE ELBO step by step and explain the KL term intuitively."*
  Learn it, then write it yourself — don't paste derivations you can't defend.
- **Code review:** *"Review src/dga/preprocessing.py for leakage and silent bugs; check the scaler
  is fit on train only when I add a split."*
- **Reviewer 2 (use weekly in W4–W5):** *"Attack my results section as a skeptical reviewer:
  what would you reject this for?"* Fix what it finds.
- **Figures:** *"Make a 300-dpi figure of the latent space coloured by Duval class and by cluster,
  side by side, IEEE single-column width."*
- **Writing:** draft in your own words first, then *"tighten this paragraph, keep my voice, no new
  claims, IEEE register."* Never let it invent results or citations.
- **Explaining to your supervisor:** *"Explain why ARI is low but silhouette high, in 4 sentences,
  for a power-engineering audience."*

---

## 7. Hard rules (academic integrity & correctness)

These are non-negotiable for a paper with your name on it:

1. **Every number in the paper comes from `results/`** — produced by code you can re-run. Never
   from a model's text output.
2. **Verify every citation by hand.** Claude *will* sometimes produce plausible-but-fake references
   or wrong page numbers. Open the actual source. The seeded `references.bib` has `\todo{}` markers
   for exactly this reason.
3. **You own the science.** Claude plans, codes, debugs, drafts, critiques. The hypotheses, the
   interpretation, and the final text are yours. Keep `lab_notebook.md` as your own record.
4. **Don't paste unverified prose** into the paper. Read, understand, rewrite.
5. **Check the AI-use policy** at CESI/KMUTNB; if a disclosure is required, write one honestly.
6. **Confidential data:** keep `data/raw/` out of any public repo unless your supervisor approves
   (the `.gitignore` has a commented switch for this).

---

## 8. Quick start (Day 1)

```bash
# in the repo root
python -m venv .venv && source .venv/bin/activate     # Win: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # if torch is slow

python scripts/prepare_data.py
python scripts/train_ae.py
python scripts/run_clustering.py        # you should see metrics + a saved figure
```
Then open Claude Code in the repo and say:
> "Read CLAUDE.md and docs/PROJECT_PLAN.md. We're on Week 1, Day 1. Walk me through verifying
> the pipeline output, then help me start the literature review in docs/literature.md."
