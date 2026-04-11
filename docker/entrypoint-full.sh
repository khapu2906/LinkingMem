#!/bin/sh
# entrypoint for the `full` Docker image.
# Starts the Python text plugin on a unix socket, waits for it to be ready,
# then starts the Rust engine in the foreground.
set -e

SOCKET="${PLUGIN_SOCKET:-/tmp/ai-graph-plugin.sock}"
PLUGIN_DIR="/app/plugins"
UVICORN="${PLUGIN_DIR}/text/.venv/bin/uvicorn"

# Remove a stale socket left over from a previous (crashed) container.
rm -f "$SOCKET"

echo "[entrypoint] starting text plugin on unix socket: $SOCKET"

# Run uvicorn from the plugins/ directory so the `text` package is importable.
cd "$PLUGIN_DIR"
"$UVICORN" text.main:app --uds "$SOCKET" --workers 1 &
PLUGIN_PID=$!

# Wait up to 60 s for the socket file to appear.
i=0
while [ $i -lt 60 ]; do
    [ -S "$SOCKET" ] && break
    sleep 1
    i=$((i + 1))
done

if [ ! -S "$SOCKET" ]; then
    echo "[entrypoint] ERROR: plugin did not create socket after 60 s" >&2
    kill "$PLUGIN_PID" 2>/dev/null
    exit 1
fi

echo "[entrypoint] plugin ready — starting engine"

# Forward SIGTERM/SIGINT to the plugin process so both shut down cleanly.
trap 'kill "$PLUGIN_PID" 2>/dev/null' EXIT INT TERM

# Run the engine in the foreground (PID 1 semantics handled by exec).
exec /usr/local/bin/server
