#!/bin/bash

# Colors for pretty printing
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Error log function
log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Clean memory function
cleanup() {
    log "Cleaning memory..."
    if python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        python -c "import gc; gc.collect()"
    fi
}

# Print start message
log "Starting SMT-UPerNet training..."

# Set environment variables for MPS optimization
# These are the key changes to fix the watermark ratio error
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Changed from 0.8
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0   # Added this
export PYTORCH_MPS_ALLOCATOR_POLICY=default  # Added this

# Create output directory
mkdir -p work_dirs/smt_segmentation

# Clean memory before starting
cleanup

# Run training
log "Starting training..."

# Added PYTHONPATH
PYTHONPATH=$(pwd):$PYTHONPATH python main.py \
    --cfg configs/smt_small_224.yaml \
    --output work_dirs/smt_segmentation \
    --tag binary_seg \
    --local_rank 0 \
    --seed 42

# Check if training completed successfully
if [ $? -eq 0 ]; then
    log "Training completed successfully"
else
    log_error "Training failed"
    exit 1
fi

# Final cleanup
cleanup

log "Script completed"