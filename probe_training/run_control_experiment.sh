#!/bin/bash

# Randomized Weights Control Experiment Runner
# This script automates the comparison between learned and randomized model representations

set -e  # Exit on error

# Determine project root and config path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config.json"

# Function to extract python_path from config.json
get_configured_python() {
    if [ -f "$CONFIG_FILE" ]; then
        # Try to read python_path from config.json using python3
        # Fallback to 'python' if key missing or parsing fails
        python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('python_path', 'python'))" 2>/dev/null || echo "python"
    else
        echo "python"
    fi
}

PYTHON_PATH=$(get_configured_python)

# Default parameters
MODEL_NAME="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
DATASET_NAME="${2:-cnn_dailymail}"
CONFIG="${3:-small}"  # small, full, or comprehensive
# If no 4th arg, leave empty so per-dataset defaults in extraction script take over
MAX_SAMPLES="${4:-}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   RANDOMIZED WEIGHTS CONTROL EXPERIMENT                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script runs the RANDOMIZED control only (baseline already computed)."
echo ""
echo "Configuration:"
echo "  Model:      $MODEL_NAME"
echo "  Dataset:    $DATASET_NAME"
echo "  Config:     $CONFIG"
if [ -n "$MAX_SAMPLES" ]; then
    samples_display="$MAX_SAMPLES"
else
    samples_display="auto (dataset default)"
fi
echo "  Samples:    $samples_display"
echo ""
echo "═══════════════════════════════════════════════════════════════════"

# Check if Python is available
if ! command -v "$PYTHON_PATH" &> /dev/null; then
    echo "❌ Python command not found: $PYTHON_PATH"
    echo "Please ensure python3 is in your PATH"
    exit 1
fi

# Step 1: Extract hidden states with RANDOMIZED weights
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/2: Extracting hidden states with RANDOMIZED weights (control)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -d "saved_features/hidden_states/$MODEL_NAME/$DATASET_NAME/article_with_zeros_RANDOMIZED" ]; then
    echo "⚠️  Randomized hidden states already exist"
    read -p "Delete and re-extract? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "saved_features/hidden_states/$MODEL_NAME/$DATASET_NAME/article_with_zeros_RANDOMIZED"
        echo "→ Extracting randomized hidden states..."
        if [ -n "$MAX_SAMPLES" ]; then
            max_samples_flag=(--max_samples "$MAX_SAMPLES")
        else
            max_samples_flag=()
        fi

        $PYTHON_PATH _0_targeted_hidden_extraction.py \
            --model_name "$MODEL_NAME" \
            --dataset_name "$DATASET_NAME" \
            "${max_samples_flag[@]}" \
            --randomize_weights
        echo "✓ Randomized extraction complete"
    else
        echo "✓ Using existing randomized hidden states"
    fi
else
    echo "→ Extracting randomized hidden states..."
    if [ -n "$MAX_SAMPLES" ]; then
        max_samples_flag=(--max_samples "$MAX_SAMPLES")
    else
        max_samples_flag=()
    fi

    $PYTHON_PATH _0_targeted_hidden_extraction.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET_NAME" \
        "${max_samples_flag[@]}" \
        --randomize_weights
    echo "✓ Randomized extraction complete"
fi

# Step 2: Train probes on RANDOMIZED representations
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/2: Training probes on RANDOMIZED representations (control)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

start_time=$(date +%s)
$PYTHON_PATH _3_train_article_level_word_importance_probe.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_NAME" \
    --use_randomized_weights \
    "$CONFIG"
end_time=$(date +%s)
control_time=$((end_time - start_time))

echo "✓ Control probe training complete (${control_time}s)"

# Display results summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   RANDOMIZED CONTROL COMPLETE                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Training time: ${control_time}s"
echo "Results saved to: results/$MODEL_NAME/$DATASET_NAME/article_level_probe_RANDOMIZED/"
echo "Insights: cat results/$MODEL_NAME/$DATASET_NAME/article_level_probe_RANDOMIZED/insights/latest_article_level_insights_$CONFIG.txt"
echo ""
echo "(Baseline paths are untouched; this run only handles randomized controls.)"
