.PHONY: test eval-routing eval-full eval-e2e

PYTHON ?= .venv/bin/python

# Testes unitários
test:
	$(PYTHON) -m pytest tests/unit/ -v

# Eval de roteamento (offline, sem API externa) — < 5 segundos
eval-routing:
	$(PYTHON) -m tests.eval.run_eval --mode routing

# Eval com LLM real (OpenRouter) — ~90 segundos, requer .env
eval-full:
	$(PYTHON) -m tests.eval.run_eval --mode full

# Eval end-to-end contra serviço deployado — requer EVAL_SERVICE_URL
eval-e2e:
	$(PYTHON) -m tests.eval.run_eval --mode e2e --url $(EVAL_SERVICE_URL)
