.PHONY: help test data train benchmark clean

help:
	@echo "VIO Drift Reduction - Makefile Commands"
	@echo "  make test        Run unit test suite"
	@echo "  make data        Run feature engineering & split verification"
	@echo "  make train       Train all ML models (Ridge, RF, GBM)"
	@echo "  make benchmark   Run multi-model comparison & generate report"
	@echo "  make clean       Remove python caches and temporary artifacts"

test:
	python3 -m unittest discover -s tests -p "test_*.py"

data:
	python3 models/00_data_preparation.py

train:
	python3 models/01_ridge_regression.py
	python3 models/02_random_forest.py
	python3 models/03_gradient_boosting.py

benchmark:
	python3 models/04_comparison.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
