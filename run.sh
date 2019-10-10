#!/bin/bash
nvidia-docker run --rm -it -v `pwd`:/work -w /work tbpp_crnn:gpu python "$@"
