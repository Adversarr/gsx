---
name: code-review
description: Review unstaged GSX code changes when the user asks for a code review, to review changes, or to check code. Applies to C, C++, CUDA, and Metal.
---

Review my unstaged GSX changes. Think carefully and revisit the patch more than once.

Start by inspecting the unstaged diff. Read surrounding code and any related headers, callers, callees, tests, or AGENTS.md conventions needed to understand the change in context. If the user points to specific files or asks for staged changes, follow that instead.

Do not use a rigid checklist. Infer what matters from the patch and the codebase. In general, pay attention to:

- correctness: whether the code does the right thing in normal and edge cases
- safety: whether it handles errors, lifetimes, bounds, and resources safely
- simplicity: whether the solution is simpler than it needs to be
- clarity: whether the intent, structure, and naming are easy to understand
- maintainability: whether future changes will be straightforward and low-risk
- extensibility: whether the design leaves room for future features or backends
- completeness: whether all required code paths, integrations, and follow-up changes are covered
- consistency with existing patterns: whether the change fits the codebase’s established structure and conventions
- tech debt: whether it introduces avoidable complexity, shortcuts, or deferred cleanup
- cross-platform compatibility: whether it behaves correctly across supported compilers, architectures, backends, and platforms
- testing coverage: whether the change is covered by adequate tests, and whether important cases are missing

Look for missed updates, incomplete error handling, edge cases, lifetime or cleanup problems, unnecessary complexity, weak abstractions, portability issues, test gaps, and unresolved design decisions.

Assume the patch may be close to merge-ready, but still challenge assumptions. Prefer simple and clear designs over clever ones. Be concrete: point to specific files, functions, or lines, explain why something matters, and suggest better alternatives when useful. Prioritize the highest-value issues rather than listing every possible nit.

Use this structure:

## Summary
## Strengths
## Issues
### Critical
### Important
### Minor
## Questions & Brainstorming
## Verdict