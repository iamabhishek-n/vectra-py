# Branch Protection Setup

This repository's CI workflow (`.github/workflows/ci.yml`) runs a job named
**`test`** on every push and pull request targeting `master`. That job
installs the package, runs `ruff check .` (lint) and `pytest` (tests), and
runs `pip-audit` (dependency vulnerability scan).

Right now, CI running is purely informational: GitHub will show a green
check or a red X on a pull request, but nothing stops a maintainer from
merging a PR while that check is red, or before it has even finished
running. Turning that from "visible" into "enforced" requires a one-time,
manual configuration change in the repository's GitHub Settings — it cannot
be done by committing a file, and it cannot be done by an API call from an
unauthenticated or non-admin context. **This step requires a repository
admin to be signed in to GitHub and to perform it by hand.**

## Why this can't be automated here

Branch protection rules are a repository setting, not a file in the repo.
Configuring them requires the GitHub REST/GraphQL API's
`repos/{owner}/{repo}/branches/{branch}/protection` endpoint (or the
equivalent Settings UI), which requires an authenticated request from a
user with **admin** access to the repository. No agent or CI job running
inside this repository has those credentials, so this is intentionally left
as a manual step for a maintainer to complete.

## Steps (one-time, for a repository admin)

1. Go to the repository on GitHub: `https://github.com/iamabhishek-n/vectra-py`
2. Click **Settings** (requires admin access to the repository).
3. In the left sidebar, click **Branches**.
4. Under **Branch protection rules**, click **Add branch ruleset** (or
   **Add rule**, depending on the GitHub UI version you see).
5. Set the branch name pattern to `master`.
6. Enable **Require status checks to pass before merging**.
7. In the status checks search box, find and select **`test`** (the job
   name defined in `.github/workflows/ci.yml`). You may need to open a
   pull request first (even a throwaway one) so that GitHub has seen the
   `test` check run at least once and offers it in this list.
8. (Recommended) Also enable **Require branches to be up to date before
   merging**, so PRs are tested against the latest `master` before merge.
9. (Recommended) Enable **Require a pull request before merging** and
   **Do not allow bypassing the above settings** so the rule applies to
   everyone, including admins, unless there's a specific reason to exempt
   them.
10. Click **Create** (or **Save changes**).

Once saved, pull requests targeting `master` will show the `test` check as
required, and the merge button will be disabled/blocked until it passes.

## Verifying it worked

Open (or push a commit to) a pull request targeting `master` and confirm:

- The PR page shows a "Required" label next to the `test` check.
- The merge button is disabled while `test` is pending or failing.
- The merge button becomes available once `test` succeeds.

## Keeping this current

If the CI job is ever renamed (the `test:` key under `jobs:` in
`.github/workflows/ci.yml`), the required status check in the branch
protection rule must be updated to match the new job name, or the old name
will simply stop appearing and the rule will have no effect.
