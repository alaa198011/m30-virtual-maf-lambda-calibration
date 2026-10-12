.PHONY: venv lock install data splits train calibrate evaluate stats figures \
        pipeline pipeline-quick test clean distclean freeze-env

PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

lock: venv
	$(PIP) install pip-tools
	$(VENV)/bin/pip-compile requirements.in -o requirements-lock.txt

install: venv
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e .

freeze-env:
	$(PY) src/run_pipeline.py --stage environment

data:
	$(PY) src/generate_dataset.py --config configs/data_generation.yaml

splits: data
	$(PY) src/prepare_splits.py --config configs/data_generation.yaml

train: splits
	$(PY) src/train_pure_ai.py --config configs/model_pure_ai.yaml
	$(PY) src/train_residual_ai.py --config configs/model_residual_ai.yaml

calibrate: train
	$(PY) src/online_calibration.py --config configs/calibration.yaml

evaluate: calibrate
	$(PY) src/evaluate.py --config configs/experiments.yaml
	$(PY) src/statistical_analysis.py --config configs/experiments.yaml

figures: evaluate
	$(PY) src/make_publication_figures.py

pipeline:
	$(PY) src/run_pipeline.py --mode full

pipeline-quick:
	$(PY) src/run_pipeline.py --mode quick

test:
	$(VENV)/bin/pytest tests/ -v

clean:
	rm -rf data/raw/* data/splits/* models/*.joblib results/*.csv results/*.parquet \
	       results/*.json results/figures/*.png
	find . -name "__pycache__" -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV) *.egg-info
