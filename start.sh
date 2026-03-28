#!/bin/bash
# Aether Application Unified Startup Script

# 1. Ensure we are in the script's directory (repository root)
cd "$(dirname "$0")" || exit

echo "======================================"
echo "    Starting Aether Unified App"
echo "======================================"

# 2. Check if the React frontend is built
if [ ! -d "aether-dashboard/dist" ]; then
    echo "--> React build not found. Building frontend..."
    cd aether-dashboard || exit
    npm install
    npm run build
    cd ..
else
    echo "--> React frontend build found."
fi

# 3. Ensure Python requirements are installed
echo "--> Installing Python dependencies..."
pip install -r requirements.txt > /dev/null

# 4. Start the Unified Server (FastAPI + Gradio + React)
echo "--> Starting Aether Server on port 7860"
echo "    Access the dashboard at: http://localhost:7860/"
python3 app.py
