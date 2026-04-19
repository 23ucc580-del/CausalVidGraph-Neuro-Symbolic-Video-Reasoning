#!/bin/bash

# --- CONFIGURATION ---
RAW_DIR="data/raw_videos"
JSON_DIR="data/processed_json"
GRAPH_DIR="data/processed_graphs"

export CUDA_VISIBLE_DEVICES=1
python src/inference_llava.py --video_dir $RAW_DIR --output_dir $JSON_DIR

if [ $? -ne 0 ]; then
    echo "? LLaVA Crashed! Stopping to preserve data."
    exit 1
fi

python src/graph_builder.py

if [ $? -ne 0 ]; then
    echo "? Graph Builder Crashed!"
    exit 1
fi

rm -rf $RAW_DIR/*
