# 07 — Configuration Layering and Secrets

## 1. Sources and precedence (lowest → highest)

```
1. Built-in defaults          config/defaults.py  (the pydantic field defaults in 06 §2)
2. library/library.toml       whole-library settings
3. models/<id>/model.md       per-model partial overrides (same section names)
4. Environment variables      GP_<SECTION>__<KEY>   e.g. GP_TEXTURE__QUALITY=detailed
5. CLI flags                  only a documented subset (08 §3) — they map onto config keys
```

Resolution produces one immutable `LibraryConfig` **per model** (`ResolvedModel.config`). Nothing reads
`library.toml` or env after `discover`; stages consume the resolved object only. This is what makes
fingerprints stable and tests simple.

```python
# config/layering.py  # NORMATIVE
def resolve_config(
    library_toml: LibraryConfig,
    model_overrides: ModelFrontMatter,
    env: Mapping[str, str],
    cli: CliOverrides,
) -> LibraryConfig:
    """Deep-merge in precedence order. Partial models only override keys that were explicitly set
    (pydantic `model_fields_set`), never keys that merely have defaults."""
```

Rules:

- A key present in a higher layer with an explicit value wins even if equal to the default.
- Lists replace, they do not append (e.g. `bake.maps`).
- `texture.style_image` is relative to the library root in all layers; `model.md` MAY instead set
  `references/style.png`-style paths — these are detected as model-relative when the file exists under
  the model folder and NOT under the library root (ambiguity → `E_PATH_AMBIGUOUS`).
- Env var parsing is handled by `pydantic-settings` with `env_prefix="GP_"`, `env_nested_delimiter="__"`.
  Booleans accept `1/0/true/false`; lists are JSON (`GP_BAKE__MAPS='["basecolor","normal"]'`).
- Unknown env vars with the `GP_` prefix cause a **warning**, not an error (CI images often carry stray
  vars); unknown TOML/front-matter keys are errors.

## 2. Secrets

| Secret | Source | Notes |
|---|---|---|
| `TRIPO_API_KEY` | env var or `.env` in the current working directory or `~/.config/tinyworld-pipeline/.env` | Never in `library.toml`, never in `model.md`, never logged. The client redacts it from all log/exception text (`****` + last 4 chars). |

```python
# config/settings.py  # NORMATIVE
class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "~/.config/tinyworld-pipeline/.env"),
                                      env_file_encoding="utf-8", extra="ignore")
    tripo_api_key: SecretStr | None = Field(None, alias="TRIPO_API_KEY")
```

`gp doctor` reports whether a key is present (never its value) and performs `GET /v3/account/balance`
to prove it works. `gp validate` does not require a key; `gp run` requires it unless `--to-stage prepare`.

## 3. Fingerprint-relevant configuration

Each stage's input fingerprint includes exactly the config subset it consumes (see `13_…` §3). The
mapping is normative:

| Stage | Config sections hashed |
|---|---|
| discover | none (hashes `model.md`, `library.toml`, `style/style.md` contents) |
| prepare | `geometry`, `uv`, `export.fbx_scale_option` (import settings are derived from it) |
| texture | `texture.*`, composed prompt sha, style image sha, guidance image shas, `static.glb` sha |
| verify | `verify.*`, `result.glb` sha, `static.glb` sha |
| apply | `verify_report.route`, `result.glb` sha; bake route only: `bake.*`, `prepared.blend` sha, Blender-side code sha |
| export | `export.*`, `prepared.blend` sha, urp textures shas, animation fbx shas |
| handoff | `unity.*`, export outputs shas, `library.unity_import_dir` |

Changing `tripo.poll_interval_s` or `logging.*` therefore never invalidates any stage.

## 4. Blender executable resolution (`blender/locate.py`)

Order: `--blender` flag → `GP_BLENDER__EXECUTABLE` env → `library.toml [blender].executable` → `PATH`
(`blender`, `blender.exe`) → well-known locations (macOS and Linux implemented in v1; the Windows
entries are specified for the deferred Windows host port, ADR-0002):

- Windows: `%ProgramFiles%\Blender Foundation\Blender 4.*\blender.exe` (highest 4.x), Steam
  `steamapps\common\Blender\blender.exe`, Microsoft Store alias.
- macOS: `/Applications/Blender.app/Contents/MacOS/Blender`, `~/Applications/…`.
- Linux: `/usr/bin/blender`, `/snap/bin/blender`, `/opt/blender*/blender`, Flatpak
  `flatpak run org.blender.Blender` (wrapped by a shim script).

The chosen path and parsed version (`blender --version` → regex `Blender (\d+)\.(\d+)\.(\d+)`) are
cached in `library/.pipeline/blender.json` for the run and shown by `gp doctor`.

## 5. Example `.env`

```
TRIPO_API_KEY=sk-********************************
GP_BLENDER__EXECUTABLE=C:\Program Files\Blender Foundation\Blender 4.5\blender.exe
GP_LOGGING__LEVEL=DEBUG
```
