# =============================================================================
# SuperModelingFactory — developer Makefile
# -----------------------------------------------------------------------------
# Conveniences for the daily edit -> compile -> test -> release loop.
# All targets are thin wrappers around the equivalent Python commands so
# Windows users (who don't get `make`) can use tasks.py instead.
# =============================================================================

PYTHON ?= python
VERSION ?=

.PHONY: help install stub compile build sdist verify clean release \
        compile-one test info

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Editable install (compiles .so in-place + installs runtime deps).
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install "cython>=3.0" build
	$(PYTHON) -m pip install -e .

stub:  ## Regenerate every .pyi stub from the current closed-source .py sources.
	$(PYTHON) gen_stubs.py

compile:  ## Compile all closed-source modules in-place (creates .so / .pyd).
	$(PYTHON) setup.py build_ext --inplace

compile-one:  ## Compile a single module, e.g. `make compile-one M=Modeling_Tool/WOE/WOE_Master.py`
	@test -n "$(M)" || (echo "Usage: make compile-one M=<path/to/file.py>" && exit 1)
	$(PYTHON) -c "from Cython.Build import cythonize; from setuptools.extension import Extension; \
		cythonize([Extension('$(subst /,.,$(basename $(M)))', ['$(M)'])], \
		compiler_directives={'language_level':'3','embedsignature':False})"
	$(PYTHON) setup.py build_ext --inplace

build:  ## Build a wheel for the current platform into ./dist/.
	$(PYTHON) -m build --wheel

sdist:  ## Build a source distribution (.tar.gz) into ./dist/.
	$(PYTHON) -m build --sdist

verify:  ## Run protection checks (parity + .pyi + MANIFEST + wheel if present).
	$(PYTHON) scripts/verify_protection.py --wheelhouse

test:  ## Smoke-test that closed-source modules import after compile.
	$(PYTHON) -c "import Modeling_Tool; \
import Modeling_Tool.WOE.WOE_Master; \
import Modeling_Tool.Model.LRM_Tool; \
import Modeling_Tool.Feature.PSI_Tool; \
print('imports OK')"

info:  ## List closed-source modules and their fingerprints.
	@$(PYTHON) -c "import gen_stubs; \
[print(f'{m:55s}  {gen_stubs.fingerprint(m)}') for m,_ in gen_stubs.MODULES]"

clean:  ## Delete build artefacts. Source .py / .pyi are preserved.
	find Modeling_Tool -name '*.c' -delete
	find Modeling_Tool -name '*.so' -delete
	find Modeling_Tool -name '*.pyd' -delete
	find Modeling_Tool -name '*.html' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ wheelhouse/ *.egg-info/

release:  ## Tag + push a new release. Usage: `make release VERSION=0.1.1`
	@test -n "$(VERSION)" || (echo "Usage: make release VERSION=0.1.1" && exit 1)
	@echo "==> bumping version to $(VERSION)"
	$(PYTHON) -c "import re,pathlib; \
p=pathlib.Path('pyproject.toml'); \
p.write_text(re.sub(r'^version\s*=.*', 'version = \"$(VERSION)\"', p.read_text(), count=1, flags=re.M))"
	@echo "==> verifying protection"
	$(MAKE) verify
	@echo "==> committing version bump"
	git add pyproject.toml
	git commit -m "chore: bump version to $(VERSION)"
	@echo "==> tagging v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo ""
	@echo "Ready to push. Run:"
	@echo "    git push origin main"
	@echo "    git push origin v$(VERSION)"
	@echo ""
	@echo "GitHub Actions will then build wheels and create the Release."
