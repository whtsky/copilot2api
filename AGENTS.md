# AGENTS.md

Agent instructions for this repository:

- When you change behavior, APIs, routing, workflows, or user-visible setup, update `CHANGELOG.md` in the same change.
- Base changelog entries on the actual commits or concrete diff, not guesses.
- Keep changelog entries focused on end-user-visible behavior, compatibility, setup, or operational changes.
- Do not include internal refactors, test-only work, or CI-only changes in `CHANGELOG.md` unless they directly affect end users.
- Keep new changelog content under `## [Unreleased]` until the project owner cuts a version.
- Prefer short, grouped bullets under headings like `Features`, `Bug Fixes`, `Tests`, `CI`, or `Docs`.
- Do not delete older release notes unless the user explicitly asks for a rewrite.

Repository note:

- `CLAUDE.md` should stay a symlink to this file so agent instructions have a single source of truth.
