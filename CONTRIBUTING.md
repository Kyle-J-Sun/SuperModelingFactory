# Contributing to SuperModelingFactory

Thanks for taking the time to look at SMF. The package is now public on PyPI
(`pip install supermodelingfactory`) under [BSL 1.1](LICENSE). This guide is
for **anyone who wants to file an issue, send a PR, or build / release SMF
from source**.

> **Users** who only want to install the released wheels do not need anything
> in this file — see the [README](README.md) for the install path.

---

## TL;DR — daily edit / build loop

```bash
# 1. Fork + clone
git clone git@github.com:<your-fork>/SuperModelingFactory.git
cd SuperModelingFactory

# 2. Editable install (compiles .so / .pyd in place via Cython)
make install

# 3. Edit any module
vim Modeling_Tool/WOE/WOE_Master.py

# 4. Recompile + light smoke test
make compile && make test

# 5. Run protection invariants (must pass before PR)
make verify

# 6. Send a PR
git checkout -b fix/<topic>
git commit -am "fix(WOE): ..."
git push origin fix/<topic>
# open a PR against main on GitHub
```

CI runs `tests` + `verify protection` on every PR. Wheel builds only run on
PRs that touch packaging files (`setup.py`, `pyproject.toml`, `MANIFEST.in`,
etc.) and on tag pushes — see [.github/workflows/build.yml](.github/workflows/build.yml).

---

## 1. Repository layout & what is closed-source

SMF is open-source but **the modeling algorithms are shipped as compiled
extensions** (`.so` / `.pyd`) rather than as plain `.py`. The original `.py`
sources of those modules live in this repo (so you can read and modify them
locally), but the **published wheel and sdist do not contain those `.py`
files** — only the compiled binaries.

| Location | In wheel | Notes |
|---|---|---|
| `Modeling_Tool/WOE/*.py` | as `.so` / `.pyd` | WOE binning, monotone constraints |
| `Modeling_Tool/Feature/*.py` | as `.so` / `.pyd` | Feature engineering |
| `Modeling_Tool/Model/*.py` | as `.so` / `.pyd` | LR / LGB / XGB / GBM wrappers |
| `Modeling_Tool/Eval/*.py` | as `.so` / `.pyd` | KS / AUC / PSI / Gini |
| `Modeling_Tool/Sample/*.py` | as `.so` / `.pyd` | Sampling utilities |
| `Modeling_Tool/Core/{Binning_Tool,kDataFrame,XOR_Encryptor,Slope_Tool}.py` | as `.so` / `.pyd` | Core algorithms |
| `Modeling_Tool/Core/{ODPS_Tool,Json_Data_Converter,utils,Check_DuckDB_Compatibility}.py` | plain `.py` | Pure-Python plumbing |
| `Modeling_Tool/UAT/`, `ExcelMaster/`, `Report/` | plain `.py` | Reporting / QA tooling |
| `Modeling_Tool/**/backup/` | excluded | Never compiled, never shipped |

**The canonical source of truth for the closed-source list is**
`setup.py :: CLOSED_SOURCE_MODULES`. Two other files mirror it and **must
stay in sync**:

- `gen_stubs.py :: MODULES` — drives `.pyi` generation
- `MANIFEST.in` — `exclude` lines drop the `.py` from the sdist

`scripts/verify_protection.py` enforces this invariant; CI runs it on every
PR via the `Verify protection` workflow.

---

## 2. Filing an issue

- **Bug report**: please include SMF version (`pip show supermodelingfactory`),
  Python version, OS, a minimal reproduction snippet, and the full traceback.
- **Feature request**: describe the use case first (what credit-risk modeling
  problem are you trying to solve?), then the proposed API.
- **Security disclosure**: do **not** file a public issue. Email the maintainer
  directly — see `pyproject.toml :: authors[0].email`.

---

## 3. Sending a pull request

Branch names: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`.

Commit messages: [Conventional Commits](https://www.conventionalcommits.org/),
e.g. `feat(Model): add CatBoost wrapper`, `fix(WOE): correct monotone
constraint when bin has all NaN`.

Required checks (all auto-run on the PR):

- `tests` — pytest suite on Py3.11 + Py3.12 across `legacy` / `modern` /
  `bleeding` dependency matrices (6 combinations total)
- `Verify protection` — runs `scripts/verify_protection.py`
- `Build wheels` — only triggered when the PR touches build-relevant files

PRs need at least one passing `tests` matrix + green `verify protection`
to be considered for merge.

---

## 4. The four common change scenarios

### 4.1  Modify an existing closed-source module

Edit the `.py`, re-compile, test, commit.

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

### 4.2  Add a NEW closed-source module

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

> If you forget any of steps 2–4, `make verify` (also in CI) will flag it.

### 4.3  Modify a pure-Python module

No special handling. Edit, test, commit.

```bash
vim ExcelMaster/Template.py
git commit -am "feat(ExcelMaster): add PVA template variant"
```

### 4.4  Move a module between compiled and pure-Python

**Compiled → pure-Python** (you decide it doesn't need compilation):

1. Remove its entry from `setup.py :: CLOSED_SOURCE_MODULES`
2. Remove its entry from `gen_stubs.py :: MODULES`
3. Remove its `exclude` line from `MANIFEST.in`
4. Delete the `.pyi` (otherwise it shadows the `.py` for IDEs)
5. `make verify`

**Pure-Python → compiled** (you decide it needs compilation):

1. Add to all three files (as in 4.2 step 2–4)
2. `make stub`
3. `make verify`

---

## 5. Releasing a new version (maintainers only)

SMF uses **semantic versioning**: `MAJOR.MINOR.PATCH`.

| Change | Bump |
|---|---|
| Bug fix, no API change | patch (`0.1.1 → 0.1.2`) |
| New feature, backward compatible | minor (`0.1.2 → 0.2.0`) |
| Breaking API change | major (`0.2.0 → 1.0.0`) |

### Release procedure

```bash
# On main, working tree clean, all CI green
git checkout main && git pull

# Bump version in BOTH files (these must agree):
#   pyproject.toml :: version
#   setup.py       :: os.environ.get("SMF_VERSION", "<version>") default
vim pyproject.toml setup.py
git commit -am "chore: bump version to 0.1.2"
git push origin main

# Wait for tests + verify to go green, then tag
git tag v0.1.2
git push origin v0.1.2
```

The tag push triggers `.github/workflows/build.yml` which:

1. Builds wheels for Linux x86_64 / macOS arm64 / Windows × Py3.10 / 3.11 /
   3.12 / 3.13 (skipping known-broken combinations — see `pyproject.toml ::
   tool.cibuildwheel`)
2. Builds an sdist
3. Creates a GitHub Release `v0.1.2` and attaches all artifacts
4. Publishes to PyPI via OIDC trusted publisher (no token needed)

### What users receive

```bash
pip install --upgrade supermodelingfactory
```

PyPI serves wheels for all supported platforms; users do **not** need
Cython, a C compiler, or anything else.

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

**`tests` workflow fails on a fresh PR for no obvious reason**
Check whether you're hitting a known dependency-matrix issue. The `bleeding`
matrix sometimes catches upstream regressions; verify against `modern` and
`legacy` first.

---

## 8. License & contact

- **License**: [BSL 1.1](LICENSE). Production use within a company is permitted;
  redistribution / SaaS / commercial competing offerings require a separate
  agreement. The Change Date converts the license to Apache 2.0 after the
  period stated in `LICENSE`.
- **Maintainer / commercial inquiries / security disclosures**: see
  `pyproject.toml :: authors[0].email`.
- **General questions**: open a GitHub Discussion or Issue.
