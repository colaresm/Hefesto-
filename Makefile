.DEFAULT_GOAL := help
IMAGE ?= hefesto:1.0.0
PORT  ?= 8501

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- Docker
build: ## Build the Docker image
	docker compose build

up: ## Start the web interface (http://localhost:$(PORT))
	docker compose up -d && echo "Hefesto running at http://localhost:$(PORT)"

down: ## Stop the web interface
	docker compose down

logs: ## Follow container logs
	docker compose logs -f hefesto

cli: ## Run the CLI in Docker, e.g. make cli ARGS="list-models"
	docker compose run --rm hefesto-cli $(ARGS)

docker-test: ## Run the test suite inside a throw-away container
	docker run --rm -v "$(PWD)/tests:/app/tests:ro" --entrypoint sh $(IMAGE) \
	  -c "pip install -q pytest && python -m pytest -q /app/tests -p no:cacheprovider"

# ----------------------------------------------------------------- Local
install: ## Install locally (editable) with UI and dev extras
	pip install -r requirements-dev.txt && pip install -e ".[ui,dev]"

run: ## Run the web interface locally
	streamlit run app/streamlit_app.py --server.port $(PORT)

test: ## Run tests
	python -m pytest

lint: ## Lint and check formatting
	ruff check src app tests && ruff format --check src app tests

format: ## Auto-format the code
	ruff format src app tests && ruff check --fix src app tests

demo-data: ## Regenerate the synthetic demo dataset
	python examples/generate_demo_dataset.py
