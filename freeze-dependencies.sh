#!/bin/bash

pipreqs --savepath=requirements.in --ignore bin,etc,include,lib,lib64 && pip-compile --resolver=backtracking
