#!/usr/bin/env bash

bk2file=$1
mp4file="${bk2file%.bk2}.mp4"

python -m retro.scripts.playback_movie $bk2file
open $mp4file