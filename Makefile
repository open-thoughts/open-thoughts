PYTHON = python3
CLEANUP_DIRS = ~/.cache/curator __pycache__ .pytest_cache .tox .coverage .nox *.egg-info dist build 

lint: 
	@echo "Running Linter (Ruff)..."
	isort tests/ src/ examples
	poetry run ruff format src tests examples

test:
	@echo "Running tests with pytest..."
	poetry run pytest tests/ --maxfail=1 --disable-warnings -q

check: 
	@echo "Checking Linter (Ruff)..."
	poetry run ruff check src/ tests/ examples --output-format=github
	poetry run ruff format src/ tests/ examples --check

clean:
	@echo "Cleaning up build artifacts and cache..."
	rm -rf $(CLEANUP_DIRS)

install:
	@echo "Installing dependencies..."
	poetry install --with dev
	poetry run pre-commit install

all: lint test clean
