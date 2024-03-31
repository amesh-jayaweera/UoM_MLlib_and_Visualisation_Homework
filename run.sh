#!/bin/bash
if [ ! -d "venv" ]; then
    python3 -m venv web/venv
    source web/venv/bin/activate
    pip install -r web/requirements.txt
else
    source web/venv/bin/activate
fi

flask run --port 8000 &
sleep 10

# Check if OS is Linux or macOS
if [ "$(uname)" == "Linux" ]; then
    xdg-open http://127.0.0.1:8000
elif [ "$(uname)" == "Darwin" ]; then
    open http://127.0.0.1:8000
fi
