# Contributing to SuperModelingFactory

Thanks for taking the time to look at SMF. The package is public on PyPI
(`pip install supermodelingfactory`) under [BSL 1.1](LICENSE). This guide is
for **anyone who wants to file an issue, send a PR, or build / release SMF
from source**.

> **Users** who only want to install the released wheels do not need anything
> in this file — see the [README](README.md) for the install path.

---

## TL;DR — daily edit / test loop

```bash
# 1. Fork + clone
git clone git@github.com:<your-fork>/SuperModelingFactory.git
cd SuperModelingFactory

# 2. Editable install
make install
# Windows: python tasks.py install

# 3. Edit any module
vim Modeling_Tool/Pipeline/credit_model.py

# 4. Smoke import + packaging check
make test && make verify

# 5. Run pytest (clone SuperModelingFactory_pytest separately)
export PYTHONPATH="$(pwd):${PYTHONPATH}"
pytest /path/to/SuperModelingFactory_pytest -q

# 6. Send a PR
git checkout -b fix/<topic>
git commit -am "fix(Pipeline): ..."
git push origin fix/<topic>
# open a PR against main on GitHub
```

CI runs `tests` and `Verify source package` on every PR. Wheel builds run on
tag pushes — see [.github/workflows/build.yml](.github/workflows/build.yml).

---

## 1. Repository layout

SMF ships as a **Python source package**. All modeling code under
`Modeling_Tool/`, `ExcelMaster/`, and `Report/` is readable `.py` source in
both the repository and the published wheel / sdist.

| Location | Notes |
|---|---|
| `Modeling_Tool/Core/` | Binning, ODPS helpers, utilities |
| `Modeling_Tool/WOE/` | WOE binning, monotone constraints, plotting |
| `Modeling_Tool/Feature/` | PSI, IV, correlation, distribution |
| `Modeling_Tool/Model/` | LR / LGB / XGB / CatBoost wrappers |
| `Modeling_Tool/Eval/` | KS / AUC / PSI / Gini, performance tables |
| `Modeling_Tool/Sample/` | Splitting, reject inference, adaptation |
| `Modeling_Tool/Explainability/` | SHAP, LIME, PDP, ICE, ALE, Owen |
| `Modeling_Tool/Pipeline/` | High-level one-click pipelines |
| `Modeling_Tool/UAT/` | Online/offline score consistency checks |
| `ExcelMaster/`, `Report/` | Excel reporting templates |
| `Modeling_Tool/**/backup/` | Archived snapshots — not part of the public API |

Test cases live in the separate
[`SuperModelingFactory_pytest`](https://github.com/Kyle-J-Sun/SuperModelingFactory_pytest)
repository. The main repo's GitHub Actions workflow clones that repo with
`secrets.PYTEST_REPO_TOKEN`.

---

## 2. Filing an issue

- **Bug report**: include SMF version (`pip show supermodelingfactory`),
  Python version, OS, a minimal reproduction snippet, and the full traceback.
- **Feature request**: describe the use case first (what credit-risk modeling
  problem are you trying to solve?), then the proposed API.
- **Security disclosure**: do **not** file a public issue. Email the maintainer
  directly — see `pyproject.toml :: authors[0].email`.

---

## 3. Sending a pull request

Branch names: `feat/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`.

Commit messages: [Conventional Commits](https://www.conventionalcommits.org/),
e.g. `feat(Pipeline): add extra_eval_datasets`, `fix(WOE): correct monotone
constraint when bin has all NaN`.

Required checks (auto-run on the PR):

- `tests` — pytest on Py3.11 + Py3.12 across `legacy` / `modern` dependency
  matrices (4 combinations)
- `Verify source package` — `compileall` + `python -m build`
- `Build wheels` — triggered on tag pushes and packaging-file changes

PRs need a green `tests` matrix and `Verify source package` before merge.

---

## 4. Common change scenarios

### 4.1  Modify an existing module

Edit the `.py`, run smoke import + pytest, commit.

```bash
vim Modeling_Tool/WOE/WOE_Master.py
make test
pytest /path/to/SuperModelingFactory_pytest/test_woe.py -q
git commit -am "fix(WOE): correct monotone constraint when bin has all NaN"
```

### 4.2  Add a new public API

1. Implement the module under the appropriate `Modeling_Tool/<Subpkg>/` folder.
2. Export it from that subpackage's `__init__.py` if it should be part of the
   public surface.
3. Add or extend tests in `SuperModelingFactory_pytest`.
4. Update docs in [`SuperModelingFactory_doc`](https://github.com/Kyle-J-Sun/SuperModelingFactory_doc)
   when the API is user-facing.

### 4.3  Pipeline changes

`Modeling_Tool/Pipeline/` hosts the high-level business pipelines. Shared
helpers live in `_common.py`. When adding config fields:

- Keep defaults `None` / `False` for backward compatibility.
- Validate inputs in `_validate_input()` before expensive work.
- Add pytest coverage in `SuperModelingFactory_pytest/test_pipeline_api.py`.

---

## 5. Releasing a new version (maintainers only)

SMF uses **semantic versioning**: `MAJOR.MINOR.PATCH`.

| Change | Bump |
|---|---|
| Bug fix, no API change | patch (`0.3.5 → 0.3.6`) |
| New feature, backward compatible | minor (`0.3.6 → 0.4.0`) |
| Breaking API change | major (`0.4.0 → 1.0.0`) |

### Release procedure

Bump version in **all five** places (they must agree):

- `pyproject.toml :: version`
- `setup.py :: SMF_VERSION` default
- `Modeling_Tool/__init__.py :: __version__`
- `README.md` and `Modeling_Tool/README.md` version footers

```bash
# On main, working tree clean, all CI green
git checkout main && git pull

make release VERSION=0.3.6
# bumps package metadata + README footers, runs verify, commits, tags

git push origin main
git push origin v0.3.6
```

Recommended commit order across repos: **pytest → main → doc → tag**.

The tag push triggers `.github/workflows/build.yml` which builds wheels, an
sdist, creates a GitHub Release, and publishes to PyPI via OIDC trusted
publisher.

### What users receive

```bash
pip install --upgrade supermodelingfactory
```

Users do **not** need a C compiler or any compile step.

---

## 6. Troubleshooting

**`pip install -e .` succeeds but `import Modeling_Tool` fails**
Check that `PYTHONPATH` is not shadowing the editable install with an old
checkout. Run `python -c "import Modeling_Tool; print(Modeling_Tool.__file__)"`.

**`make verify` fails locally**
Run `python -m compileall -q Modeling_Tool ExcelMaster Report` to see which
file has a syntax error, then `python -m build --sdist --wheel` for packaging
issues.

**`tests` workflow fails on a fresh PR**
Check whether you're hitting a known dependency-matrix issue. Compare against
`modern` and `legacy` matrices in
[.github/workflows/tests.yml](.github/workflows/tests.yml).

---

## 7. License & contact

- **License**: [BSL 1.1](LICENSE). Production use within a company is permitted;
  redistribution / SaaS / commercial competing offerings require a separate
  agreement. The Change Date converts the license to Apache 2.0 after the
  period stated in `LICENSE`.
- **Maintainer / commercial inquiries / security disclosures**: see
  `pyproject.toml :: authors[0].email`.
- **General questions**: open a GitHub Discussion or Issue.
