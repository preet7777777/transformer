#!/usr/bin/env bash
# tfs-server.sh — start/stop/restart/status the TFS API and web demo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pids"
API_PID="$PID_DIR/api.pid"
WEB_PID="$PID_DIR/web.pid"
API_LOG="$SCRIPT_DIR/.pids/api.log"
WEB_LOG="$SCRIPT_DIR/.pids/web.log"

API_PORT="${TFS_API_PORT:-8000}"
WEB_PORT="${TFS_WEB_PORT:-8080}"
REGISTRY_DIR="${TFS_REGISTRY_DIR:-$SCRIPT_DIR/registry}"
RUNS_DIR="${TFS_RUNS_DIR:-$SCRIPT_DIR/runs}"

mkdir -p "$PID_DIR"

_is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

_stop() {
    local name="$1" pid_file="$2"
    if _is_running "$pid_file"; then
        kill "$(cat "$pid_file")" && rm -f "$pid_file"
        echo "  stopped $name"
    else
        rm -f "$pid_file"
        echo "  $name was not running"
    fi
}

_start_api() {
    if _is_running "$API_PID"; then
        echo "  API already running (pid $(cat "$API_PID"))"
        return
    fi
    tfs-serve \
        --registry-dir "$REGISTRY_DIR" \
        --runs-dir "$RUNS_DIR" \
        --port "$API_PORT" \
        > "$API_LOG" 2>&1 &
    echo $! > "$API_PID"
    echo "  API started  →  http://localhost:$API_PORT  (pid $!, log: $API_LOG)"
}

_start_web() {
    if _is_running "$WEB_PID"; then
        echo "  Web already running (pid $(cat "$WEB_PID"))"
        return
    fi
    tfs-web \
        --port "$WEB_PORT" \
        > "$WEB_LOG" 2>&1 &
    echo $! > "$WEB_PID"
    echo "  Web started  →  http://localhost:$WEB_PORT  (pid $!, log: $WEB_LOG)"
}

cmd="${1:-help}"

case "$cmd" in
start)
    echo "Starting TFS servers..."
    _start_api
    _start_web
    ;;
stop)
    echo "Stopping TFS servers..."
    _stop "API" "$API_PID"
    _stop "Web" "$WEB_PID"
    ;;
restart)
    echo "Restarting TFS servers..."
    _stop "API" "$API_PID"
    _stop "Web" "$WEB_PID"
    sleep 1
    _start_api
    _start_web
    ;;
status)
    if _is_running "$API_PID"; then
        echo "  API  running  pid=$(cat "$API_PID")  http://localhost:$API_PORT"
    else
        echo "  API  stopped"
    fi
    if _is_running "$WEB_PID"; then
        echo "  Web  running  pid=$(cat "$WEB_PID")  http://localhost:$WEB_PORT"
    else
        echo "  Web  stopped"
    fi
    ;;
logs)
    target="${2:-all}"
    case "$target" in
        api) tail -f "$API_LOG" ;;
        web) tail -f "$WEB_LOG" ;;
        *)   tail -f "$API_LOG" "$WEB_LOG" ;;
    esac
    ;;
help|*)
    cat <<'EOF'
Usage: ./tfs-server.sh <command> [target]

Commands:
  start           Start API server and web demo
  stop            Stop both servers
  restart         Stop then start both servers
  status          Show running status and URLs
  logs [api|web]  Tail logs (default: both)

Environment variables:
  TFS_API_PORT      API port (default: 8000)
  TFS_WEB_PORT      Web port (default: 8080)
  TFS_REGISTRY_DIR  Registry directory (default: ./registry)
  TFS_RUNS_DIR      Runs directory to auto-scan (default: ./runs)

Examples:
  ./tfs-server.sh start
  ./tfs-server.sh status
  ./tfs-server.sh logs api
  ./tfs-server.sh stop
  TFS_API_PORT=9000 ./tfs-server.sh start
EOF
    ;;
esac
