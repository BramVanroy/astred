# Control quality
quality:
	black --check --line-length 119 --target-version py37 sac
	isort --check-only --recursive sac
	flake8 sac

# Format source code automatically
style:
	black --line-length 119 --target-version py37 sac
	isort --recursive sac
