SHELL := /bin/bash

.PHONY: test
test:
	python3 -m doctest src/datagen.py
