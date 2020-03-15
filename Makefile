# Control quality
quality:
	black --check --line-length 96 --target-version py36 sacr
	isort --check-only --recursive sacr
	flake8 sac

# Format source code automatically
style:
	black --line-length 96 --target-version py36 sacr
	isort --recursive sacr
