# SMF 发版流程

## TL;DR

```bash
# 1. 三处版本号同步升,推 PR,merge 到 main
python scripts/check_version_consistency.py    # 本地先自查

# 2. 从 main 打 tag,格式必须 v{X}.{Y}.{Z}
git tag -a v0.4.3 -m "SMF 0.4.3 — <one-line summary>"
git push origin v0.4.3
```

Tag 推送后 GitHub Actions 自动:
1. 构建 wheel + sdist
2. Smoke-test 装 wheel 并 import 五个核心模块
3. 附加 artifact 到 GitHub Release 并自动生成 release notes
4. 通过 Trusted Publisher (OIDC) 发到 PyPI

无需任何本地 twine、pypirc、或手动上传。

## 三处版本号

发版前 **必须** 保证以下三个文件版本号一致:

| 文件 | 字段 |
|---|---|
| `Modeling_Tool/__init__.py` | `__version__ = "X.Y.Z"` |
| `setup.py` | `os.environ.get("SMF_VERSION", "X.Y.Z")` 里的默认值 |
| `pyproject.toml` | `[project] version = "X.Y.Z"` |

**为什么三处都要盯**:`python -m build` 走 PEP 517,PEP 621 元数据(`pyproject.toml`)优先于 `setup.py`。三者中任何一处漂移都会导致构建产物版本号与 `__version__` / tag 对不上,而 PyPI 是根据构建产物版本号建目录的。

`scripts/check_version_consistency.py` 会在每次 PR / push 到 main 时通过 `version-consistency.yml` workflow 自动跑,漂移会直接 CI 挂掉,不会漏到发版环节。

## 版本号选取规则

- **Patch** (`0.x.Y`):纯硬化 / bug fix,零 API 移除,健康输入零数值差异。
- **Minor** (`0.X.0`):至少一处行为变更(默认值翻转、必填字段变化、返回值语义调整)。任何 breaking default **必须先在上一 patch 通过 opt-in 参数落地**,再在下一 minor 翻默认。参考:
  - 0.3.19 → 0.4.0:`ScoreComparisonPipelineConfig.cross_vars` 默认从 `["rating"]` 翻到 `[]`
  - 0.4.2 → 未来 0.5.0:`PSI_Tool.missing_policy` 默认将从 `"drop"` 翻到 `"include"`
- **Major** (`X.0.0`):重大架构调整。目前无计划。

## 三仓协同规则

代码变更是主仓、doc、pytest 三仓协同事务:

1. 修 pytest 仓,commit + push + open PR
2. 修 doc 仓,commit + push + open PR
3. 修 _agent 仓(如需更新 known_gotchas 或 pipeline_catalog),commit + push + open PR
4. **最后**修主仓,commit + push + open PR

主仓 PR **合并顺序** 也遵循同样的 `pytest → doc → _agent → main`。主仓合并触发 tag,tag 触发 PyPI 发版。

详情见 Space 顶层 `SMF Coordinated Push Workflow` instruction。

## 事前 checklist

发 tag 之前手动过一遍:

- [ ] 三仓的 PR 都已 merge 到各自 default 分支
- [ ] `python scripts/check_version_consistency.py` 本地跑通
- [ ] `main` 最新 commit 上的 tests / verify / build workflow 都是绿的
- [ ] 全量 pytest 本地跑通:0 skip 0 fail
- [ ] doc 仓 `docs/changelog/vX.Y.Z.md` 已就位
- [ ] `_agent` 仓 `known_gotchas.md` 已 append 本轮批次(若有修复)

## 事后 checklist

Tag 推送后大约 3 分钟内 GitHub Actions 应完成:

- [ ] `Build distributions` workflow run 显示 3 个 job 全绿
- [ ] `https://github.com/Kyle-J-Sun/SuperModelingFactory/releases/tag/vX.Y.Z` 存在,附带 wheel + sdist
- [ ] `curl -s https://pypi.org/pypi/supermodelingfactory/X.Y.Z/json` 返回 200
- [ ] `pip install SuperModelingFactory==X.Y.Z` 从干净虚拟环境能装

## 手动 fallback

如果 Trusted Publisher OIDC 因某种原因失效(比如 PyPI 上的 trusted publisher 配置被清了),你可以本地手动发一次:

```bash
cd SuperModelingFactory
git checkout main && git pull
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/*   # 需要本地 ~/.pypirc 或 TWINE_PASSWORD
```

之后重新配 Trusted Publisher(在 pypi.org project settings 里),下次 tag 会自动恢复。
