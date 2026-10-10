.PHONY: install dev lint format test data models jobs

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

data:     ## Semantic Drone Dataset (TU Graz) + Ultralytics layout at 1536 x 1024
	bash scripts/download_data.sh

models:   ## Cityscapes-pretrained YOLO26-sem weights
	bash scripts/download_models.sh

jobs:
	python3 slurm/generate.py --list
