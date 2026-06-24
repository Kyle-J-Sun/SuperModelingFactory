# Installing SuperModelingFactory

## Quick Install

```bash
pip install supermodelingfactory
```

That's it.

> macOS users: install OpenMP runtime once (required by lightgbm, a transitive
> dependency):
>
> ```bash
> brew install libomp
> ```

## Supported Environments

| OS | Architecture | Python |
|---|---|---|
| macOS 11+ | arm64 (Apple Silicon) | 3.10 / 3.11 / 3.12 / 3.13 |
| Linux | x86_64 (manylinux_2_28) | 3.10 / 3.11 / 3.12 / 3.13 |
| Windows | x86_64 | 3.10 / 3.11 / 3.12 / 3.13 |

> ⚠️ Intel Mac users: `pip install` will fall back to the sdist and require a
> local Cython build (`pip install cython` first). Pre-built wheels are
> Apple Silicon only.

## Verification

```bash
python -c "
from Modeling_Tool import WOE_Master, LRMaster, PSICalculator
import Modeling_Tool
print('SMF version:', Modeling_Tool.__version__)
print('OK')
"
```

## Optional Dependencies

Some sub-modules pull in heavyweight dependencies that we keep behind extras:

```bash
# Alibaba Cloud ODPS / MaxCompute integration
pip install 'supermodelingfactory[odps]'
```

## Upgrading

```bash
pip install --upgrade supermodelingfactory
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: Library not loaded: @rpath/libomp.dylib` (macOS) | `brew install libomp` |
| `Bad CPU type in executable` on Apple Silicon | You have an Intel Python at `/usr/local/bin/python3.x`. Use Homebrew or conda Python (arm64). Verify with `python -c "import platform; print(platform.machine())"` — expect `arm64`. |
| `ModuleNotFoundError: No module named 'odps'` | Install with the odps extra: `pip install 'supermodelingfactory[odps]'` |
| `ImportError` from a closed-source module on import | You're on an unsupported Python or platform — verify with `pip debug --verbose` and check the supported environments table above. |

## For Maintainers

If you're a maintainer building wheels locally instead of consuming pre-built
ones, see [CONTRIBUTING.md](CONTRIBUTING.md).
