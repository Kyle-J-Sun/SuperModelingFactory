# Contributing to SuperModelingFactory

> This guide is for **maintainers and licensed contributors**. The repository
> is private; only people with push access (or holders of a commercial license
> and signed contributor agreement) should be using this workflow.

---

## TL;DR — daily edit / build / release loop

```bash
# Edit a closed-source module
vim Modeling_Tool/WOE/WOE_Master.py

# Recompile in place + smoke test
make compile && make test           # or: python tasks.py compile && python tasks.py test

# Run protection checks
make verify                         # or: python tasks.py verify

# Commit + push (CI rebuilds wheels)
git commit -am "fix(WOE): ..."
git push

# When ready to ship a version
make release VERSION=0.1.1
git push origin main && git push origin v0.1.1
```

---

## 1. Repository layout & what is closed-source

| Location | Status | Notes |
|---|---|---|
| `Modeling_Tool/WOE/*.py` | **closed-source** | Compiled to `.so` / `.pyd` via Cython |
| `Modeling_Tool/Feature/*.py` | **closed-source** | Same |
| `Modeling_Tool/Model/*.py` | **closed-source** | Same |
| `Modeling_Tool/Eval/*.py` | **closed-source** | Same |
| `Modeling_Tool/Sample/*.py` | **closed-source** | Same |
| `Modeling_Tool/Core/{Binning_Tool,kDataFrame,XOR_Encryptor,Slope_Tool}.py` | **closed-source** | Same |
| `Modeling_Tool/Core/{ODPS_Tool,Json_Data_Converter,utils,Check_DuckDB_Compatibility}.py` | open | Plumbing — users need to read these |
| `Modeling_Tool/UAT/`, `ExcelMaster/`, `Report/` | open | Reporting / QA tooling |
| `Modeling_Tool/**/backup/` | excluded entirely | Never compiled, never shipped |

**The canonical source of truth for the closed-source list is**
`setup.py :: CLOSED_SOURCE_MODULES`. Two other files mirror it and **must
stay in sync**:

- `gen_stubs.py :: MODULES` — drives `.pyi` generation
- `MANIFEST.in` — `exclude` lines drop closed-source `.py` from the sdist

`scripts/verify_protection.py` enforces this invariant in CI.

---

## 2. One-time setup

```bash
git clone git@github.com:Kyle-J-Sun/<repo>.git
cd <repo>

# Install Cython + editable build (creates .so in place)
make install                        # or: python tasks.py install
```

After this, every `import Modeling_Tool.WOE.WOE_Master` resolves to the
compiled extension in your working tree.

---

## 3. The four common change scenarios

### 3.1  Modify an existing closed-source module

Edit the `.py`, re-compile, test, commit. **Nothing else changes.**

```bash
vim Modeling_Tool/WOE/WOE_Master.py
make compile-one M=Modeling_Tool/WOE/WOE_Master.py   # fast — single module
make test                                            # imports + light smoke
make verify                                          # invariants still hold
git commit -am "fix(WOE): correct monotone constraint when bin has all NaN"
```

If the public API (class / function signatures) changed, regenerate the stub
so IDE users see the new signature:

```bash
make stub
git add Modeling_Tool/WOE/WOE_Master.pyi
```

### 3.2  Add a NEW closed-source module

The only scenario that touches configuration files. Steps **in order**:

1. Create `Modeling_Tool/<Subpkg>/<New_Tool>.py` and write the code.
2. Add an entry to `setup.py :: CLOSED_SOURCE_MODULES` (keep grouped by subpkg).
3. Add the **same** entry to `gen_stubs.py :: MODULES`.
4. Add an `exclude` line in `MANIFEST.in`.
5. Run `make stub` — generates the `.pyi` with a fresh fingerprint.
6. Run `make compile && make test && make verify` — all three must pass.
7. Commit:
   ```bash
   git add Modeling_Tool/<Subpkg>/<New_Tool>.py \
           Modeling_Tool/<Subpkg>/<New_Tool>.pyi \
           setup.py gen_stubs.py MANIFEST.in
   git commit -m "feat(<Subpkg>): add <New_Tool>"
   ```

> **Tip:** if you forget any of steps 2–4, `make verify` (also run in CI) will
> flag it and refuse to merge.

### 3.3  Modify an open-source module

No special handling. Edit, test, commit.

```bash
vim ExcelMaster/Template.py
git commit -am "feat(ExcelMaster): add PVA template variant"
```

### 3.4  Move a module between closed-source and open-source

**Closed → open** (you decide it doesn't need protection):
1. Remove its entry from `setup.py :: CLOSED_SOURCE_MODULES`
2. Remove its entry from `gen_stubs.py :: MODULES`
3. Remove its `exclude` line from `MANIFEST.in`
4. Delete the `.pyi` (otherwise it shadows the `.py` for IDEs)
5. `make verify`

**Open → closed** (you decide it needs protection):
1. Add to all three files (as in 3.2 step 2–4)
2. `make stub`
3. `make verify`

---

## 4. Releasing a new version

We use **semantic versioning**: `MAJOR.MINOR.PATCH`.

| Change | Bump |
|---|---|
| Bug fix, no API change | patch (`0.1.1 → 0.1.2`) |
| New feature, backward compatible | minor (`0.1.2 → 0.2.0`) |
| Breaking API change | major (`0.2.0 → 1.0.0`) |

### Release procedure

```bash
# On main, working tree clean, all CI green
git checkout main && git pull
make release VERSION=0.1.1          # bumps pyproject.toml, verifies, commits, tags
git push origin main
git push origin v0.1.1
```

The tag push triggers `.github/workflows/build.yml` which:

1. Builds wheels for Linux / macOS arm64 / macOS x86_64 / Windows
   × Python 3.10 / 3.11 / 3.12 = **12 wheels**
2. Builds an sdist
3. Creates a GitHub Release **v0.1.1** and attaches all 13 artefacts
4. Auto-generates release notes from commit history

### What users receive

```bash
pip install --upgrade \
  https://github.com/Kyle-J-Sun/<repo>/releases/download/v0.1.1/<wheel-name>.whl
```

Users do **not** need Cython, a C compiler, or push access to the repo —
the Release is consumable by anyone you grant Read access to.

---

## 5. Branch / PR conventions

- `main` is protected: every change must come in via PR.
- Branch names: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`.
- Commit messages: [Conventional Commits](https://www.conventionalcommits.org/),
  e.g. `feat(Model): add CatBoost wrapper`, `fix(WOE): ...`.
- PRs require: passing `pytest` workflow + passing `build` workflow (which
  runs `make verify` and builds at least one wheel).

---

## 6. The seven invariants CI checks

`scripts/verify_protection.py` (also invoked by `make verify`) enforces:

| # | Invariant | Failure mode if violated |
|---|---|---|
| 1 | `setup.py :: CLOSED_SOURCE_MODULES` == `gen_stubs.py :: MODULES` | Stubs drift from compiled set |
| 2 | Every closed-source `.py` has a sibling `.pyi` | IDE users lose autocomplete |
| 3 | Every closed-source `.py` is `exclude`-ed in `MANIFEST.in` | sdist leaks source |
| 4 | No closed-source `.py` exists in any built wheel | wheel leaks source |
| 5 | Each closed-source dotted module has a `.so` / `.pyd` in the wheel | import fails for users |
| 6 | (manual review) `.pyi` carries a `FINGERPRINT:` marker | plagiarism evidence lost |
| 7 | (manual review) `LICENSE` is BSL 1.1 and the Change Date is in the future | legal layer broken |

---

## 7. Troubleshooting

**`make compile` fails with `undeclared name not builtin: foo`**
The `.py` is missing an `import`. Cython is stricter than Python — fix the
import; pure-Python execution would have hit `NameError` at runtime anyway.

**Stub file doesn't show new method in IDE**
You probably forgot `make stub` after changing a class signature. Stubs are
regenerated from current `.py` AST.

**`pip install -e .` builds, but `import` finds the old `.so`**
Run `make clean && make install` — stale `.so` from a previous Python version
is being picked up.

**Wheel verification fails in CI but local `make verify` passes**
`make verify` without a wheel only runs invariants 1–3. CI also runs 4–5 on
the actually-built wheel. Run `make build && make verify` locally to reproduce.

---

## 8. Contact

Commercial licensing, security disclosures, and contributor agreements:
see `pyproject.toml :: authors[0].email`.
