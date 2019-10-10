#!/bin/bash
nvidia-docker run --rm -it -v `pwd`:/work -w /work tbpp:gpu python "$@"
