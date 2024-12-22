#!/bin/bash

# Colors for pretty printing
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Log functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Clean memory function
cleanup() {
    log "Cleaning memory..."
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    python -c "import torch; torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None"
    python -c "import gc; gc.collect()"
}

# Set paths
CONFIG_FILE="configs/smt_small_224.yaml"
CHECKPOINT_FILE="work_dirs/smt_segmentation/smt_small_224/binary_seg/model_best.pth"
OUTPUT_DIR="visualizations"

# Print start message
log "Starting visualization..."

# Verify files exist
if [ ! -f "$CONFIG_FILE" ]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CHECKPOINT_FILE" ]; then
    log_error "Checkpoint file not found: $CHECKPOINT_FILE"
    exit 1
fi

# Create output directory
log "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Initial cleanup
cleanup

# Run visualization
log "Running visualization..."
log "Using checkpoint: $CHECKPOINT_FILE"

PYTHONPATH=$(pwd):$PYTHONPATH python unified_visualization.py \
    --cfg "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT_FILE" \
    --output "$OUTPUT_DIR" \
    --local_rank 0 \
    --world_size 1

if [ $? -eq 0 ]; then
    log "Visualization completed successfully"
    log "Results saved to: $OUTPUT_DIR"
    ls -l "$OUTPUT_DIR"
else
    log_error "Visualization failed"
    exit 1
fi

# Final cleanup
cleanup

log "Script completed successfully"