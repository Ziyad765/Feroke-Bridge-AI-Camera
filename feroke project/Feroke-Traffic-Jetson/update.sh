#!/bin/bash
# ============================================================
# update.sh — Pull latest code from GitHub and restart Feroke
# Usage (on Jetson): bash ~/feroke/Feroke-Traffic-Jetson/update.sh
# ============================================================

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"   # root of the git repo
LITE_DIR="$REPO_DIR/feroke project/Feroke-Traffic-Lite"
JETSON_DIR="$REPO_DIR/feroke project/Feroke-Traffic-Jetson"

echo "🔄 Pulling latest code from GitHub..."
cd "$REPO_DIR" || { echo "❌ Cannot find repo at $REPO_DIR"; exit 1; }

# Stash any local-only changes (e.g. feroke_settings.json tweaks) so pull never fails
git stash push -m "auto-stash before update $(date +%Y%m%d-%H%M%S)" --include-untracked

git pull origin main
PULL_STATUS=$?

# Restore local settings if they were stashed
git stash pop 2>/dev/null

if [ $PULL_STATUS -ne 0 ]; then
    echo "❌ Git pull failed. Check your network or SSH key."
    exit 1
fi

echo "✅ Code updated successfully."
echo ""
echo "👉 To apply and run the update now:"
echo "   bash \"$JETSON_DIR/run.sh\""
echo ""
read -p "Launch Feroke now? [Y/n] " LAUNCH
LAUNCH="${LAUNCH:-Y}"
if [[ "$LAUNCH" =~ ^[Yy]$ ]]; then
    bash "$JETSON_DIR/run.sh"
fi
