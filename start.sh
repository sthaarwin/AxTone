#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        🎸 AxTone Startup Script 🎸        ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo ""

# Check if Python dependencies are installed
echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"
if python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo -e "${GREEN}✓ Python dependencies OK${NC}"
else
    echo -e "${YELLOW}⚠ Installing Python dependencies...${NC}"
    pip install -r requirements.txt
fi

echo ""
echo -e "${YELLOW}🚀 Starting AxTone servers...${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping servers...${NC}"
    kill $PYTHON_PID 2>/dev/null
    kill $NEXT_PID 2>/dev/null
    wait $PYTHON_PID 2>/dev/null
    wait $NEXT_PID 2>/dev/null
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Python API backend
echo -e "${BLUE}Starting Python API backend on port 8000...${NC}"
python api.py &
PYTHON_PID=$!
sleep 3

# Start Next.js frontend
echo -e "${BLUE}Starting Next.js frontend on port 3000...${NC}"
cd frontend/axtone
npm run dev &
NEXT_PID=$!

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         ✅ Both servers are running!       ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Frontend: http://localhost:3000          ║${NC}"
echo -e "${GREEN}║  API Docs: http://localhost:8000/docs     ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop both servers        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""

# Wait for both processes
wait
