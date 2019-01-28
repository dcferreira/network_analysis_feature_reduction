#!/bin/sh
if [ "$1" = "tfm" ]; then
    cd bokeh_stream && PYTHONPATH=.. bokeh serve . --address 0.0.0.0 --allow-websocket-origin '*'
else
    python -u run.py $@
fi
