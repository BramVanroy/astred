# Control quality
quality:
	black --check --line-length 96 --target-version py36 sac
	isort --check-only --recursive sac
	flake8 sac

# Format source code automatically
style:
	black --line-length 96 --target-version py36 sac
	isort --recursive sac
