# Format source code automatically
style:
	black --line-length 119 --target-version py37 astred examples
	isort astred examples

# Control quality
quality:
	black --check --line-length 119 --target-version py37 astred examples
	isort --check-only astred examples
	flake8 astred --exclude __pycache__,__init__.py

