.PHONY: test eval-routing eval-full eval-e2e

# Testes unitários
test:
	python -m pytest tests/unit/ -v

# Eval de roteamento (offline, sem API externa) — < 5 segundos
eval-routing:
	python -m tests.eval.run_eval --mode routing

# Eval com LLM real (OpenRouter) — ~90 segundos, requer .env
eval-full:
	python -m tests.eval.run_eval --mode full

# Eval end-to-end contra serviço deployado — requer EVAL_SERVICE_URL
eval-e2e:
	python -m tests.eval.run_eval --mode e2e --url $(EVAL_SERVICE_URL)
