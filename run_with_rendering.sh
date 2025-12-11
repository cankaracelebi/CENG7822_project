#!/bin/bash
# Quick wrapper to run Python programs with proper OpenGL support in conda
# This uses the system's libstdc++ instead of conda's version


# export this to make it work with miniconda + ubuntu opengl
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Run whatever command was passed
"$@"
