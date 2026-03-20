#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build-cov"
TEST_REGEX=""
HTML_DIR=""  # deprecated; kept for backward compatibility
MD_OUT=""
FILTER_ROOT=""
EXCLUDE_STB=1
EXCLUDE_HAPPLY=1
XML_OUT=""

usage() {
  cat <<EOF
Usage: $0 [options]
  --build-dir <path>       Build directory (default: build-cov)
  --test-regex <regex>     Only run tests matching this regex (ctest -R)
  --md-out <file>          Markdown report output file (default: <build-dir>/coverage/coverage.md)
  --html-dir <path>        [Deprecated] Directory for HTML details; if set, writes Markdown to <path>/coverage.md
  --filter-root <path>     gcovr --filter root (default: <repo>/gsx)
  --no-exclude-stb         Do not exclude gsx/src/extra/stb/.* from coverage
  --no-exclude-happly      Do not exclude gsx/src/extra/happly/.* from coverage
  --xml-out <file>         Also write gcovr XML to this path
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --test-regex) TEST_REGEX="$2"; shift 2;;
    --html-dir) HTML_DIR="$2"; shift 2;;
    --md-out) MD_OUT="$2"; shift 2;;
    --filter-root) FILTER_ROOT="$2"; shift 2;;
    --no-exclude-stb) EXCLUDE_STB=0; shift;;
    --no-exclude-happly) EXCLUDE_HAPPLY=0; shift;;
    --xml-out) XML_OUT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

REPO_ROOT="$(pwd)"
[[ -n "$FILTER_ROOT" ]] || FILTER_ROOT="$REPO_ROOT/gsx"

# Determine markdown output path with backward compatibility for --html-dir
if [[ -z "$MD_OUT" ]]; then
  if [[ -n "$HTML_DIR" ]]; then
    MD_OUT="$HTML_DIR/coverage.md"
  else
    MD_OUT="$BUILD_DIR/coverage/coverage.md"
  fi
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found" >&2; exit 1
fi
if ! command -v ctest >/dev/null 2>&1; then
  echo "ctest not found" >&2; exit 1
fi
if ! command -v gcovr >/dev/null 2>&1; then
  echo "gcovr not found; install it via brew/pipx" >&2; exit 1
fi

# Find llvm-cov on macOS if available
LLVM_COV=""
if command -v xcrun >/dev/null 2>&1; then
  set +e
  LLVM_COV="$(xcrun -f llvm-cov 2>/dev/null || true)"
  set -e
fi
if [[ -z "$LLVM_COV" ]] && command -v llvm-cov >/dev/null 2>&1; then
  LLVM_COV="$(command -v llvm-cov)"
fi

echo "[coverage] Configuring build in '$BUILD_DIR'"
CFG_ARGS=(
  -S .
  -B "$BUILD_DIR"
  -DGSX_ENABLE_COVERAGE=ON
  -DGSX_BUILD_TESTS=ON
  -DGSX_USE_CUDA=OFF
  -DGSX_USE_METAL=OFF
)
if [[ -n "$LLVM_COV" ]]; then
  CFG_ARGS+=( -DGSX_LLVM_COV_EXECUTABLE="$LLVM_COV" )
fi
cmake "${CFG_ARGS[@]}"

echo "[coverage] Building..."
cmake --build "$BUILD_DIR" -j

echo "[coverage] Resetting old coverage artifacts"
cmake -DGSX_COVERAGE_BUILD_DIR="$BUILD_DIR" -P cmake/gsx-reset-coverage.cmake

echo "[coverage] Running tests ${TEST_REGEX:+(regex: $TEST_REGEX)}"
if [[ -n "$TEST_REGEX" ]]; then
  ctest --test-dir "$BUILD_DIR" -R "$TEST_REGEX" --output-on-failure
else
  ctest --test-dir "$BUILD_DIR" --output-on-failure
fi

mkdir -p "$(dirname "$MD_OUT")"

GCOV_EXEC="$BUILD_DIR/gsx-llvm-gcov.sh"
if [[ ! -x "$GCOV_EXEC" ]]; then
  # Fallback to gcov if wrapper is not present (e.g., GCC toolchain)
  if command -v gcov >/dev/null 2>&1; then
    GCOV_EXEC="$(command -v gcov)"
  else
    echo "Neither $GCOV_EXEC nor gcov found" >&2; exit 1
  fi
fi

# Ensure gcov executable path is absolute (gcovr may change CWD per file)
if [[ -x "$GCOV_EXEC" && "$GCOV_EXEC" != /* ]]; then
  GCOV_EXEC="$(cd "$(dirname "$GCOV_EXEC")" && pwd)/$(basename "$GCOV_EXEC")"
fi

echo "[coverage] Generating gcovr reports"
GCOVR_ARGS=(
  --root "$REPO_ROOT"
  --object-directory "$BUILD_DIR"
  --gcov-executable "$GCOV_EXEC"
  --filter "$FILTER_ROOT"
  --txt
  --markdown "$MD_OUT"
)
if [[ "$EXCLUDE_STB" -eq 1 ]]; then
  GCOVR_ARGS+=( --exclude "$REPO_ROOT/gsx/src/extra/stb/.*" )
fi
if [[ "$EXCLUDE_HAPPLY" -eq 1 ]]; then
  GCOVR_ARGS+=( --exclude "$REPO_ROOT/gsx/src/extra/happly/.*" )
fi
if [[ -n "$XML_OUT" ]]; then
  mkdir -p "$(dirname "$XML_OUT")"
  GCOVR_ARGS+=( --xml "$XML_OUT" )
fi

gcovr "${GCOVR_ARGS[@]}"

echo "[coverage] Markdown report at: $MD_OUT"
