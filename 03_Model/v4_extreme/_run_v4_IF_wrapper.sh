#!/bin/bash
cd "$(dirname "$0")/.."
python src/run_v4_postprocess_IF.py > docs/v4_IF_run.log 2>&1
echo "EXIT_CODE=$?" >> docs/v4_IF_run.log
echo "DONE" >> docs/v4_IF_run.log
