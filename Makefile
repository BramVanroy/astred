# Format source code automatically
style:
	black --line-length 96 --target-version py36 astred
	isort --recursive astred

# Control quality
quality:
	black --check --line-length 96 --target-version py36 astred
	isort --check-only --recursive astred
	flake8 astred --exclude __pycache__,__init__.py

