#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$PWD"
pytest tests
