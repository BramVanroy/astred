# Format source code automatically
style:
	black --line-length 96 --target-version py36 astred examples
	isort --recursive astred examples

# Control quality
quality:
	black --check --line-length 96 --target-version py36 astred examples
	isort --check-only --recursive astred examples
	flake8 astred --exclude __pycache__,__init__.py

