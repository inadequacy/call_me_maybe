# Makefile for python code
# 
# > make help
#
# The following commands can be used.
#
# install:  Installs development requirements
# run:  Executes the logic
# debug: ((need to add isn't on that link))
# clean:  Remove build and cache files
# lint:  Runs flake8 on src, exit if critical rules are broken
# lint-strict: 
# init:  sets up environment and installs requirements
# env:  Source venv and environment files for testing
# leave:  Cleanup and deactivate venv

VENV_PATH='test_area/bin/activate'
ENVIRONMENT_VARIABLE_FILE='.env'
DOCKER_NAME=$DOCKER_NAME
DOCKER_TAG=$DOCKER_TAG

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo 'The following commands can be used.'
	@echo ''
	$(call find.functions)

env: ## Source venv and environment files for testing
env:
	python3 -m venv test_area
	source $(VENV_PATH)
	source $(ENVIRONMENT_VARIABLE_FILE)

install: ## Install
install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	