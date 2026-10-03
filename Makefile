# SeqForge · Makefile
# 作者：晨星 (CJX0712)
#
# Windows 下用 `make`（若已装）或直接用等价的 python 命令。

PY ?= python

.DEFAULT_GOAL := help
.PHONY: help install install-dev test test-invariants test-backends \
        lint format check demo demo-quick ablation selftest clean coverage

help: ## 显示可用目标
	@grep -E "^[a-zA-Z_-]+:.*?## .*$$" $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## 安装运行时依赖（最小集：numpy + scipy）
	$(PY) -m pip install -r requirements.txt

install-dev: ## 安装开发依赖（含可选后端 + lint/test 工具）
	$(PY) -m pip install -r requirements.lock.txt

test: ## 跑全部测试
	$(PY) -m pytest -q -W ignore::UserWarning

test-invariants: ## 只跑 22 条确定性不变量（CI 门禁）
	$(PY) -m pytest tests/test_invariants.py -v -W ignore::UserWarning

test-backends: ## 只跑后端与降级测试
	$(PY) -m pytest tests/test_backends.py -v -W ignore::UserWarning

lint: ## ruff check（硬门禁）
	$(PY) -m ruff check .

format: ## ruff format
	$(PY) -m ruff format .

format-check: ## ruff format --check
	$(PY) -m ruff format --check .

check: lint format-check test ## CI 等价三档：lint + format + pytest

coverage: ## 测试 + 覆盖率报告
	$(PY) -m pytest -q -W ignore::UserWarning --cov=. --cov-report=term-missing

demo: ## 端到端 demo（含消融）
	$(PY) examples/run_demo.py --ablation

demo-quick: ## 快速 demo
	$(PY) examples/run_demo.py --quick

ablation: ## 只跑消融对照
	$(PY) examples/run_demo.py --quick --ablation

selftest: ## CLI 快速自检
	$(PY) cli.py selftest

clean: ## 清理本地产物
	rm -rf artifacts .pytest_cache .ruff_cache .coverage htmlcov *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
