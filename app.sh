#!/usr/bin/env bash
set -euo pipefail

# to use, run PORT=8501 bash app.sh
# run this if needed pkill -f streamlit

# Default to prod port 8888, but allow override via ENV or CLI arg
PORT="${PORT:-${1:-8888}}"

# Try to kill any existing Streamlit processes (ignore errors)
if ! pkill -f streamlit 2>/dev/null; then
  echo "No existing Streamlit process found."
else
  echo "Previous Streamlit process killed."
fi

mkdir -p .streamlit
cat > .streamlit/config.toml <<EOF
[browser]
gatherUsageStats = true
[server]
address = "0.0.0.0"
port = $PORT
enableCORS = false
enableXsrfProtection = false
[theme]
primaryColor = "#543FDD"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#FAFAFA"
textColor = "#2E2E38"
EOF


# Generate and display the Streamlit URL
if [ -n "${DOMINO_RUN_HOST_PATH:-}" ]; then
    CLEAN_PATH=$(echo "$DOMINO_RUN_HOST_PATH" | sed 's|/r||g')
    STREAMLIT_URL="https://fitch.domino-eval.com${CLEAN_PATH}proxy/${PORT}/"
    echo "========================================="
    echo "Streamlit URL: $STREAMLIT_URL"
    echo "========================================="
else
    echo "DOMINO_RUN_HOST_PATH not found - running locally"
fi

# Run the app
streamlit run Agents.py
