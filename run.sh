#!/bin/bash

# Check if venv exists or else create it
if [ ! -d "venv" ]; then

    # Check if Python has been installed
    if ! command -v python &> /dev/null; then
        echo "Python is not found. Please install Python and ensure it's added to the PATH."
        exit 1
    fi

    python -m venv venv

fi

# Activate the virtual environment
source venv/bin/activate

pip install -q --disable-pip-version-check -r front-end/requirements.txt


# @ehco "Starting the application..."

cd front-end/src

streamlit run app.py 