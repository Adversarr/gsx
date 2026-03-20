---
name: test-coverage
description: |
  Generate code coverage for C/C++ CMake projects using gcovr. Trigger when users ask to enable/measure coverage, run a subset of tests (regex like "^CoreRuntime\."), or produce TXT/Markdown/XML coverage reports. Works with Clang/GCC; uses llvm-cov wrapper when available, otherwise gcov.
compatibility:
  tools: [cmake>=3.20, ctest, gcovr>=5.0, llvm-cov or gcov]
  os: [Linux, macOS]
---

# Test Coverage Skill

What it does
- Configures a coverage build, runs tests (optionally filtered), generates gcovr TXT/Markdown/XML.
- Supports Clang (llvm-cov) and GCC (gcov) without platform lock-in.

Requirements
- cmake, ctest, gcovr; plus llvm-cov (Clang) or gcov (GCC).

Quick start (repo root)
- CoreRuntime only:
```bash
.agents/skills/test-coverage/scripts/generate_coverage.sh --test-regex '^CoreRuntime\.'
```
- Full suite + Markdown + XML:
```bash
.agents/skills/test-coverage/scripts/generate_coverage.sh \
  --md-out build-cov/coverage/full/coverage.md \
  --xml-out build-cov/coverage/full/coverage.xml
```

Options
- `--build-dir <dir>` default `build-cov`
- `--test-regex <regex>` only run matching tests
- `--md-out <file>` Markdown report output (default `<build-dir>/coverage/coverage.md`)
- `--filter-root <path>` gcovr filter root (default `<repo>/gsx`)
- `--no-exclude-stb`, `--no-exclude-happly` stop excluding bundled third-party code
- `--xml-out <file>` additionally write XML

Notes
- Non-`gsx` projects: adjust `--filter-root` and excludes.
