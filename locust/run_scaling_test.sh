#!/usr/bin/env bash
# Automated scaling test: 1, 2 and 4 containers sequentially.
# Run from project root after docker compose is available.
set -euo pipefail

RESULTS_DIR="locust/results"
mkdir -p "$RESULTS_DIR"

for SCALE in 1 2 4; do
  echo "=========================================="
  echo "Scaling to $SCALE container(s)..."
  docker compose up -d --scale app=$SCALE
  sleep 15   # give containers time to become healthy

  echo "Running Locust (100 users, 20 spawn/s, 60 s) against scale=$SCALE"
  locust -f locust/locustfile.py \
         --host=http://localhost:8080 \
         --users 100 \
         --spawn-rate 20 \
         --run-time 60s \
         --headless \
         --csv="$RESULTS_DIR/scale_${SCALE}" \
         --html="$RESULTS_DIR/scale_${SCALE}_report.html" \
         2>&1 | tee "$RESULTS_DIR/scale_${SCALE}_output.txt"

  echo "Finished scale=$SCALE"
done

echo "All scaling tests complete. Results in $RESULTS_DIR/"
docker compose down
