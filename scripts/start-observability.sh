#!/bin/bash

# Quick start observability stack for AMK Agent
# Usage: bash scripts/start-observability.sh

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   AMK Agent Observability Stack - Quick Start              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check docker-compose
if ! docker-compose --version > /dev/null 2>&1; then
    echo -e "${RED}❌ docker-compose is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ docker-compose is available${NC}"
echo ""

echo "Starting observability stack..."
echo ""

# Start the observability services
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d

echo ""
echo -e "${GREEN}✓ Observability stack is starting${NC}"
echo ""
echo "Waiting for services to be ready (60s)..."
sleep 10

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Services are up! Access them at:                         ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  📊 Grafana Dashboards:                                    ║"
echo "║     http://localhost:3000 (admin/admin)                    ║"
echo "║                                                            ║"
echo "║  📈 Prometheus Metrics:                                    ║"
echo "║     http://localhost:9090                                  ║"
echo "║     Django Metrics: http://localhost:8000/api/metrics/     ║"
echo "║                                                            ║"
echo "║  🔍 Tempo Traces:                                          ║"
echo "║     http://localhost:3200                                  ║"
echo "║     (View via Grafana Explore > Tempo)                     ║"
echo "║                                                            ║"
echo "║  📝 Loki Logs:                                             ║"
echo "║     http://localhost:3100                                  ║"
echo "║     (View via Grafana Explore > Loki)                      ║"
echo "║                                                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                               ║"
echo "║  1. Open Grafana: http://localhost:3000                    ║"
echo "║  2. Add data sources (Prometheus, Tempo, Loki)             ║"
echo "║  3. View dashboards: System Metrics, RAG Pipeline          ║"
echo "║                                                            ║"
echo "║  📖 See OBSERVABILITY.md for full documentation            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "View logs with:"
echo "  docker-compose logs -f grafana"
echo "  docker-compose logs -f prometheus"
echo "  docker-compose logs -f tempo"
echo "  docker-compose logs -f loki"
echo ""

echo "Stop observability stack with:"
echo "  docker-compose -f docker-compose.yml -f docker-compose.observability.yml down"
