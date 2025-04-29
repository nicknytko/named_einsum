#!/bin/bash

echo "... flake8 ..."
python3 -m flake8 --docstring-convention numpy --statistics named_einsum --exclude named_einsum/lark_parser.py,named_einsum/version.py && echo "flake8 passed."
echo
echo "... pylint ..."
pylint --rcfile setup.cfg named_einsum
