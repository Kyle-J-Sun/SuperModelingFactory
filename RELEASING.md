# SMF Release Process

## TL;DR

```bash
# 1. Bump the version in all three places, open a PR, merge to main
python scripts/check_version_consistency.py    # run the check locally first

# 2. Tag from main; the format must be v{X}.{Y}.{Z}
git tag -a v0.4.3 -m "SMF 0.4.3 — <one-line summary>"
git push origin v0.4.3
```

After the tag is pushed, GitHub Actions automatically:
1. Builds the wheel and sdist
2. Smoke-tests the wheel: installs it and imports the five core modules
3. Attaches the artifacts to a GitHub Release and generates release notes
4. Publishes to PyPI through a Trusted Publisher (OIDC)

No local `twine`, `.pypirc`, or manual upload is needed.

## The three version numbers

Before a release, the version must be **identical** in these three files:

| File | Field |
|---|---|
| `Modeling_Tool/__init__.py` | `__version__ = "X.Y.Z"` |
| `setup.py` | The default in `os.environ.get("SMF_VERSION", "X.Y.Z")` |
| `pyproject.toml` | `[project] version = "X.Y.Z"` |

**Why all three matter:** `python -m build` follows PEP 517, and the PEP 621 metadata in `pyproject.toml` takes precedence over
`setup.py`. If any one of the three drifts, the built artifacts carry a version that disagrees with `__version__` and the tag,
and PyPI creates its directory from the artifact's version.

`scripts/check_version_consistency.py` runs automatically on every PR and every push to `main` through the
`version-consistency.yml` workflow. Drift fails CI immediately, so it cannot reach the release step.

## Choosing the version number

- **Patch** (`0.x.Y`): pure hardening and bug fixes, no API removals, zero numerical differences on healthy inputs.
- **Minor** (`0.X.0`): at least one behavior change (a flipped default, a changed required field, changed return-value
  semantics). Any breaking default **must first land behind an opt-in parameter in the previous patch release**, and the default
  is flipped only in the next minor. Examples:
  - 0.3.19 → 0.4.0: the default of `ScoreComparisonPipelineConfig.cross_vars` flipped from `["rating"]` to `[]`
  - 0.4.2 → a future 0.5.0: the default of `PSI_Tool.missing_policy` will flip from `"drop"` to `"include"`
- **Major** (`X.0.0`): a major architectural change. None is planned.

Documented exceptions:

- **0.9.1** is a patch release that changes numbers: the class-pure WOE bin merge default (`small_bin_policy="merge"`),
  equal-frequency binning with missing values in their own bin, and the reject-inference and score-comparison fixes. Each
  is listed with its legacy setting (or "bug fix, no legacy switch") in the "Behavior-changing fixes" box of
  `docs/changelog/v0.9.1.md` in the doc repository. 0.9.0 was prepared (commit `e1ef627`) but never tagged or published,
  so 0.9.1 is the release after 0.8.2 on PyPI.

## Coordinating the repositories

A code change is a coordinated change across the main, doc, and pytest repositories (and the agent repository when needed):

1. Update the pytest repository: commit, push, open a PR
2. Update the doc repository: commit, push, open a PR
3. Update the `_agent` repository if `known_gotchas.md` or `pipeline_catalog.md` needs it: commit, push, open a PR
4. Update the main repository **last**: commit, push, open a PR

The PRs are also **merged** in the order `pytest → doc → _agent → main`. Merging the main repository triggers the tag, and the tag
triggers the PyPI release.

For details, see the top-level `SMF Coordinated Push Workflow` instruction of the Space.

## Pre-release checklist

Go through this by hand before tagging:

- [ ] The PRs of all repositories are merged into their default branches
- [ ] `python scripts/check_version_consistency.py` passes locally
- [ ] The tests, verify, and build workflows are green on the latest commit of `main`
- [ ] The full pytest suite passes locally: 0 skipped, 0 failed
- [ ] The doc repository has `docs/changelog/vX.Y.Z.md`
- [ ] The `_agent` repository's `known_gotchas.md` has this release's batch appended (if there were fixes)

## Post-release checklist

About three minutes after the tag is pushed, GitHub Actions should have finished:

- [ ] The `Build distributions` workflow run shows all three jobs green
- [ ] `https://github.com/Kyle-J-Sun/SuperModelingFactory/releases/tag/vX.Y.Z` exists, with the wheel and sdist attached
- [ ] `curl -s https://pypi.org/pypi/supermodelingfactory/X.Y.Z/json` returns 200
- [ ] `pip install SuperModelingFactory==X.Y.Z` works in a clean virtual environment

## Manual fallback

If the Trusted Publisher (OIDC) stops working for some reason (for example, the trusted-publisher configuration on PyPI was
cleared), you can publish once from your machine:

```bash
cd SuperModelingFactory
git checkout main && git pull
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/*   # needs a local ~/.pypirc or TWINE_PASSWORD
```

Afterwards, reconfigure the Trusted Publisher in the project settings on pypi.org; the next tag will then publish automatically again.
