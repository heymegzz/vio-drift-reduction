.PHONY: help test data train benchmark clean all dl-test dl-train dl-evaluate dl-report dl-smoke

help:
	@echo "VIO Drift Reduction - Makefile Commands"
	@echo "  make test        Run Phase 1 unit tests"
	@echo "  make dl-test     Run all tests (requires Phase 2 dependencies)"
	@echo "  make dl-smoke    Run a small end-to-end Phase 2 check"
	@echo "  make dl-train/dl-evaluate/dl-report  Run Phase 2 stages"
	@echo "  make data        Run feature engineering & split verification"
	@echo "  make train       Train all ML models (Ridge, RF, GBM)"
	@echo "  make benchmark   Run multi-model comparison & generate report"
	@echo "  make all         Run full pipeline (data → train → benchmark)"
	@echo "  make clean       Remove python caches and temporary artifacts"

test:
	python3 -W ignore -m unittest tests.test_feature_engineering tests.test_model_inference

data:
	python3 models/00_data_preparation.py

train:
	python3 models/01_ridge_regression.py
	python3 models/02_random_forest.py
	python3 models/03_gradient_boosting.py

benchmark:
	python3 models/04_comparison.py

all: data train benchmark

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Phase 2 writes only to its own results/checkpoint directories.
dl-test:
	python3 -W ignore -m unittest discover -s tests -p "test_*.py"

dl-train:
	python3 -m src.dl.experiment --stage train

dl-evaluate:
	python3 -m src.dl.experiment --stage evaluate

dl-report:
	python3 -m src.dl.explain
	python3 -m src.dl.report

dl-smoke:
	python3 -m src.dl.experiment --smoke --stage all
