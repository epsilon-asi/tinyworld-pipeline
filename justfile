# Grounded pipeline developer commands. `just --list` shows them.

set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Create the venv, install all extras, install git hooks.
setup:
    uv sync --all-extras
    uv run pre-commit install

# Ruff lint + format check + import-layer contracts.
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run lint-imports

# Apply ruff fixes and formatting.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# mypy --strict over src and tests.
type:
    uv run mypy src tests tools

# Unit tests with the 80 % line/branch/function coverage gates. Excludes Blender and live Tripo suites.
test *ARGS:
    uv run pytest -n auto -m "not blender and not tripo_live" \
        --cov=grounded_pipeline --cov-branch --cov-report=term-missing --cov-report=json \
        --cov-fail-under=80 {{ARGS}}
    uv run python tools/check_function_coverage.py coverage.json --threshold 80

# Blender integration tests (needs GP_BLENDER_EXE or an auto-detectable Blender 4.5).
test-blender *ARGS:
    uv run pytest -m blender {{ARGS}}

# Live Tripo smoke test (needs TRIPO_API_KEY; spends credits).
test-tripo-live *ARGS:
    uv run pytest -m tripo_live {{ARGS}}

# Regenerate committed FBX fixtures with the real Blender.
fixtures:
    "${GP_BLENDER_EXE:-/Applications/Blender.app/Contents/MacOS/Blender}" -b --factory-startup \
        --python tools/make_fixtures.py -- tests/fixtures/fbx

# Everything CI runs.
ci: lint type test
