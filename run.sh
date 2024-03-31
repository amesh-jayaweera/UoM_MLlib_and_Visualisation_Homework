#!/bin/bash
if [ ! -d "venv" ]; then
    python3 -m venv web/venv
    source web/venv/bin/activate
    pip install -r web/requirements.txt
else
    source web/venv/bin/activate
fi

flask run &
sleep 3

# Check if OS is Linux or macOS
if [ "$(uname)" == "Linux" ]; then
    xdg-open http://127.0.0.1:5000
elif [ "$(uname)" == "Darwin" ]; then
    open http://127.0.0.1:5000
fi
