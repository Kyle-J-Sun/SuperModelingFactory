# =============================================================================
# SuperModelingFactory — developer Makefile
# -----------------------------------------------------------------------------
# Conveniences for the daily edit -> test -> release loop.
# Windows users (who don't get `make`) can use tasks.py instead.
# =============================================================================

PYTHON ?= python
VERSION ?=

.PHONY: help install build sdist verify clean test release

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Editable install (runtime deps only).
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

build:  ## Build a wheel for the current platform into ./dist/.
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build --wheel

sdist:  ## Build a source distribution (.tar.gz) into ./dist/.
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build --sdist

verify:  ## Compile sources and build sdist + wheel (matches CI verify workflow).
	$(PYTHON) -m compileall -q Modeling_Tool ExcelMaster Report
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build --sdist --wheel

test:  ## Smoke-test that core modules import.
	$(PYTHON) -c "import Modeling_Tool; \
import Modeling_Tool.WOE.WOE_Master; \
import Modeling_Tool.Model.LRM_Tool; \
import Modeling_Tool.Feature.PSI_Tool; \
import Modeling_Tool.Pipeline; \
print('imports OK')"

clean:  ## Delete build artefacts.
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ wheelhouse/ *.egg-info/

release:  ## Tag + push a new release. Usage: `make release VERSION=0.3.6`
	@test -n "$(VERSION)" || (echo "Usage: make release VERSION=0.3.6" && exit 1)
	@echo "==> bumping version to $(VERSION)"
	$(PYTHON) -c "import re,pathlib; \
files=[('pyproject.toml',r'^version\s*=.*',f'version = \"$(VERSION)\"'), \
('setup.py',r'SMF_VERSION\", \"[^\"]+\"',f'SMF_VERSION\", \"$(VERSION)\"'), \
('Modeling_Tool/__init__.py',r'__version__ = \"[^\"]+\"',f'__version__ = \"$(VERSION)\"'), \
('README.md',r'^-\s+\*\*Version\*\*:\s*\S+',f'- **Version**: $(VERSION)'), \
('Modeling_Tool/README.md',r'^-\s+\*\*Version\*\*:\s*\S+',f'- **Version**: $(VERSION)')]; \
[(lambda p,pat,rep: p.write_text(re.sub(pat,rep,p.read_text(encoding='utf-8'),count=1,flags=re.M)))(pathlib.Path(f),pat,rep) for f,pat,rep in files]"
	@echo "==> verifying package"
	$(MAKE) verify
	@echo "==> committing version bump"
	git add pyproject.toml setup.py Modeling_Tool/__init__.py README.md Modeling_Tool/README.md
	git commit -m "chore: bump version to $(VERSION)"
	@echo "==> tagging v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo ""
	@echo "Ready to push. Run:"
	@echo "    git push origin main"
	@echo "    git push origin v$(VERSION)"
	@echo ""
	@echo "Also bump README version footers and sync SuperModelingFactory_doc / pytest repos."
	@echo "GitHub Actions will then build wheels and create the Release."
