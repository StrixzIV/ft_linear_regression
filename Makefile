SHELL := /bin/bash

setup:
	python3 -m venv evaluation-env
	source ./evaluation-env/bin/activate
	pip install -r requirements.txt

source:
	source ./evaluation-env/bin/activate

clean:
	rm -rf evaluation-env
