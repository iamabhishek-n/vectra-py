# Phase 5 — vectra-py OSS Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give vectra-py the standard OSS contributor-facing scaffolding it's missing: CONTRIBUTING.md, issue templates, a PR template, a lint tool + CI lint step, and a CHANGELOG.md with semver discipline going forward.

**Architecture:** Pure documentation/CI-config additions, plus adding one new dev dependency (a linter — vectra-py currently has NO lint tool configured at all, unlike vectra-js which already has ESLint). `CODE_OF_CONDUCT.md` already exists (confirmed by direct inspection) and is NOT touched by this plan.

**Tech Stack:** GitHub Actions (`.github/workflows/ci.yml` already exists, runs `pytest` + `pip-audit`). `ruff` (new dependency) for linting — fast, single-tool, increasingly the default choice for new Python projects, avoids needing to configure/reconcile multiple older tools (flake8 + black + isort).

## Global Constraints

- Do not touch `CODE_OF_CONDUCT.md` (already exists) — confirmed present, out of scope.
- Do not attempt to enable GitHub branch-protection / required-status-checks via any API — this requires authenticated, repo-admin-level GitHub access this session does not have. Document the exact manual steps instead (mirrors vectra-js's Phase 5 Task 3 approach).
- CHANGELOG content must be reconstructed from real `git log`/`pyproject.toml` version history — do not fabricate version numbers, dates, or feature descriptions not verifiable from actual commits.
- No direct-to-master commits, no force-push, no skipped hooks.
- Every task ends with `pytest` still passing (this plan doesn't touch runtime code, but confirm nothing broke regardless).

---

### Task 1: Add CONTRIBUTING.md and issue templates

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`

**Interfaces:**
- Produces: contributor-facing docs mirroring vectra-js's equivalents in spirit (read `C:\Users\shiny\OneDrive\Desktop\Github\vectra\vectra-js\CONTRIBUTING.md` and `C:\Users\shiny\OneDrive\Desktop\Github\vectra\vectra-js\.github\ISSUE_TEMPLATE\*.md` if they exist by the time this task runs — vectra-js's Phase 5 plan creates them — for tone/structure consistency across the two sibling SDKs), adapted to Python idiom (`pip install -e ".[dev]"`, `pytest`, not yet `ruff` since that's added in Task 3).

- [ ] **Step 1: Check for the sibling repo's equivalents**

Check if `C:\Users\shiny\OneDrive\Desktop\Github\vectra\vectra-js\CONTRIBUTING.md` and `.github/ISSUE_TEMPLATE/*.md` exist yet (they may or may not, depending on execution order across repos). If they exist, read them for tone/structure to mirror. If not, proceed with your own reasonable structure — don't block on it.

- [ ] **Step 2: Write CONTRIBUTING.md**

Cover: dev setup (`git clone`, `python -m venv .venv`, `pip install -e ".[dev]"`), running tests (`pytest`), making a PR (one logical change per PR, tests required, link to `CODE_OF_CONDUCT.md`), reporting bugs/features (link to the new issue templates). Keep it concise.

- [ ] **Step 3: Write issue templates**

Create `.github/ISSUE_TEMPLATE/bug_report.md` (sections: description, repro steps, expected vs actual behavior, environment — Python version, vectra-py version) and `.github/ISSUE_TEMPLATE/feature_request.md` (sections: problem description, proposed solution, alternatives considered). Use standard GitHub issue-template YAML frontmatter (`---\nname: ...\nabout: ...\ntitle: ...\nlabels: ...\n---`) matching GitHub's conventional format.

- [ ] **Step 4: Commit**

```bash
git add CONTRIBUTING.md .github/ISSUE_TEMPLATE/bug_report.md .github/ISSUE_TEMPLATE/feature_request.md
git commit -m "docs: add CONTRIBUTING.md and issue templates"
```

---

### Task 2: Add a Pull Request template

**Files:**
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

**Interfaces:**
- Produces: every new PR is pre-populated with a checklist (tests added/passing, description, linked issue).

- [ ] **Step 1: Write the PR template**

Create `.github/PULL_REQUEST_TEMPLATE.md` with: a "What does this PR do?" prompt, a checklist (`- [ ] Tests added/updated and passing (\`pytest\`)`, `- [ ] Linked issue (if applicable)` — do not reference lint yet, since Task 3 adds it; if executing after Task 3 lands, include a lint checklist item too), a "Related issue" line.

- [ ] **Step 2: Commit**

```bash
git add .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: add pull request template"
```

---

### Task 3: Add ruff for linting, wire it into CI

**Files:**
- Modify: `pyproject.toml` (add `ruff` to `[project.optional-dependencies].dev`, add a `[tool.ruff]` config section)
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/PULL_REQUEST_TEMPLATE.md` (add the lint checklist item, if Task 2 already landed without it)
- Create: `docs/BRANCH_PROTECTION_SETUP.md` (or fold into CONTRIBUTING.md — implementer's call)

**Interfaces:**
- Produces: `ruff check .` runs cleanly (or with justified, documented exclusions) and is invoked in CI after `pytest`.

- [ ] **Step 1: Read the current CI workflow and pyproject.toml**

Read `.github/workflows/ci.yml` and `pyproject.toml`'s `[project.optional-dependencies]` section to see the exact current structure before editing.

- [ ] **Step 2: Add ruff as a dev dependency**

Add `"ruff"` to the `dev` extras list in `pyproject.toml`. Add a minimal `[tool.ruff]` config section (e.g. `line-length = 120` to match this codebase's apparent style rather than ruff's stricter 88-char default — check a few existing files' typical line lengths first to pick a reasonable value; a lenient default that doesn't require reformatting every file is preferred for a first-time lint adoption).

- [ ] **Step 3: Run ruff locally, fix or suppress what it finds**

Install: `pip install -e ".[dev]"`
Run: `ruff check .`

If it reports a manageable number of issues, fix the auto-fixable ones (`ruff check . --fix`) and manually address the rest if they're small/mechanical (unused imports, obvious style issues). If it reports a large number of pre-existing issues that would be excessive to fix in this task, use a `[tool.ruff]` `ignore` list for specific noisy-but-harmless rule codes, or an initial narrower `select` list (e.g. just `["E", "F"]` — pyflakes + pycodestyle errors, the highest-signal baseline) rather than ruff's full default rule set, and document in your report why. The goal is a lint gate that's meaningful from day one, not one so strict it's immediately red with hundreds of pre-existing warnings unrelated to this task.

- [ ] **Step 4: Add the CI lint step**

Add a `run: ruff check .` step to the existing CI job (after `pip install -e ".[dev]"`, before or after `pytest` — implementer's call, before is more conventional).

- [ ] **Step 5: Write the branch-protection documentation**

Same content/reasoning as vectra-js's Phase 5 Task 3: document the exact manual GitHub Settings → Branches steps to require the `test` CI job to pass before merging to `master`. State plainly this requires manual, authenticated admin action this session cannot perform via API.

- [ ] **Step 6: Verify**

Run: `ruff check . && pytest`
Expected: both pass (or ruff passes against whatever `select`/`ignore` scope was chosen in Step 3).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml .github/PULL_REQUEST_TEMPLATE.md docs/BRANCH_PROTECTION_SETUP.md CONTRIBUTING.md
git commit -m "ci: add ruff linting, wire into CI, document manual branch-protection setup"
```
(Adjust file list based on where the branch-protection doc actually landed and whether the PR template needed updating.)

---

### Task 4: Add CHANGELOG.md, adopt Keep-a-Changelog format going forward

**Files:**
- Create: `CHANGELOG.md`
- Modify: `CONTRIBUTING.md` (add semver-discipline note)

**Interfaces:**
- Produces: same structure/reasoning as vectra-js's Phase 5 Task 4 — a `CHANGELOG.md` in Keep a Changelog format, real historical entries reconstructed from `git log`/`pyproject.toml` version history (currently at `1.0.0` per direct inspection), an `[Unreleased]` section, and a semver-discipline note in `CONTRIBUTING.md`.

- [ ] **Step 1: Reconstruct real version history**

Run `git log -p --all -- pyproject.toml | grep -B2 -A2 'version ='` (or equivalent) to find every point the version changed. Do NOT invent version numbers, dates, or feature descriptions not backed by real git history.

- [ ] **Step 2: Write CHANGELOG.md**

Same Keep a Changelog structure as vectra-js's (see that repo's `CHANGELOG.md` for the exact format/header text to mirror if it exists by the time this runs, for cross-SDK consistency; otherwise use the standard Keep a Changelog header). Real entries only; honest terse entries where specifics aren't recoverable from history.

- [ ] **Step 3: Add semver-discipline note to CONTRIBUTING.md**

Same content as vectra-js's equivalent note — link to semver.org, note that version bumps should be their own commit/PR separate from feature work.

- [ ] **Step 4: Verify tests still pass**

Run: `pytest`
Expected: all pass (this task touches no runtime code, but confirm nothing broke).

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md CONTRIBUTING.md
git commit -m "docs: add CHANGELOG.md, document semver discipline"
```

---

## Self-Review Notes

- **Spec coverage:** CONTRIBUTING.md + issue templates (Task 1, vectra-py had NEITHER before this plan, unlike vectra-js which already had issue templates), PR template (Task 2), lint tool + CI lint gate + documented manual branch-protection (Task 3), CHANGELOG.md + semver discipline (Task 4).
- **Placeholder scan:** no TBD/TODO; Task 3's lint-scope decision (full ruff ruleset vs. a narrower baseline) is an explicit, reasoned implementer judgment call, not a placeholder.
- **Cross-repo consistency:** Tasks explicitly reference checking the vectra-js sibling's Phase 5 output for tone/format consistency where order-of-execution allows it, without hard-blocking on it if vectra-js's plan hasn't run yet.
- **No CODE_OF_CONDUCT work**: confirmed already exists via direct file listing before writing this plan.
