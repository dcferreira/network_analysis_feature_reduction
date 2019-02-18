#!/bin/sh
if [ "$1" = "tfm" ]; then
    cd bokeh_stream && PYTHONPATH=.. bokeh serve . --address 0.0.0.0 --allow-websocket-origin '*'
elif [ "$1" = "unsw15" ]; then
    python -u run_unsw15.py $@
elif [ "$1" = "csv" ]; then
    python -u run_csv.py $@
fi
