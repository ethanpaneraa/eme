#!/bin/bash

# Script to start both frontend and server services
# Usage: ./start_services.sh

set -e  # Exit on any error

echo "starting GroupMe Vector Bot Services..."
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "shutting down services..."
    kill $FRONTEND_PID $SERVER_PID 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the server
echo "📡 Starting FastAPI server..."
cd server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!
cd ..

# Wait a moment for server to start
sleep 2

# Start the frontend
echo "🎨 Starting React frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "services started successfully!"
echo "server running at: http://localhost:8000"
echo "frontend running at: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for both processes
wait $FRONTEND_PID $SERVER_PID
