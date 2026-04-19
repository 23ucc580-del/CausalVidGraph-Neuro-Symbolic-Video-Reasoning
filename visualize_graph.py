# -*- coding: utf-8 -*-
import os
import argparse
import torch
import av
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig

# --- CONFIGURATION ---
LLAVA_MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
NUM_FRAMES = 16  # Reduced to 16 for cleaner graph visualization (32 is too crowded)

def get_event_timeline(video_path):
    print(f"[INFO] Analyzing video for Event Extraction: {os.path.basename(video_path)}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    processor = LlavaNextVideoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    
    # --- STABILIZATION FIX (Same as Inference) ---
    if not hasattr(processor, "patch_size"):
        processor.patch_size = model.config.vision_config.patch_size
    if not hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

    # Extract Frames (Resized to 336x336 for stability)
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0: total_frames = 100
    indices = np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int)
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_ndarray(format="rgb24")
            img_resized = cv2.resize(img, (336, 336))
            frames.append(img_resized)
            if len(frames) >= NUM_FRAMES: break
    
    clip = np.stack(frames)
    
    # --- PROMPT FOR STRUCTURED OUTPUT ---
    # We ask LLaVA specifically for a list of events to map to nodes
    prompt = "USER: <video>\nList the 5 main distinct events happening in this video in chronological order. Use a bullet point for each event.\nASSISTANT:"
    
    inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    raw_text = processor.batch_decode(out, skip_special_tokens=True)[0]
    
    if "ASSISTANT:" in raw_text:
        response = raw_text.split("ASSISTANT:")[-1].strip()
    else:
        response = raw_text.split("USER:")[-1].strip()

    print("[INFO] Event Timeline Extracted.")
    del model, processor
    torch.cuda.empty_cache()
    return response

def parse_events_to_nodes(event_text):
    # Cleaning the text to get a list of short node labels
    lines = event_text.split('\n')
    nodes = []
    for line in lines:
        clean_line = line.strip().replace('*', '').replace('-', '').strip()
        if clean_line:
            # Keep labels short for the graph (max 40 chars)
            short_label = (clean_line[:37] + '..') if len(clean_line) > 37 else clean_line
            nodes.append(short_label)
    return nodes

def draw_causal_graph(nodes, output_file="causal_graph.png"):
    print(f"[INFO] drawing graph with {len(nodes)} nodes...")
    
    G = nx.DiGraph() # Directed Graph
    
    # Add Nodes
    for i, label in enumerate(nodes):
        G.add_node(i, label=f"T{i+1}\n{label}")
        
    # Add Sequential Edges (Temporal Flow)
    edges = []
    for i in range(len(nodes) - 1):
        edges.append((i, i+1))
    G.add_edges_from(edges)
    
    # --- VISUALIZATION STYLE ---
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42, k=0.8)  # Layout algorithm
    
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, arrowsize=20, edge_color='gray')
    
    # Draw Labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    plt.title(f"Neuro-Symbolic Causal Graph (Visual Evidence)", fontsize=14)
    plt.axis('off')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Graph saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help="Path to raw video")
    parser.add_argument('--output', type=str, default="causal_event_graph.png", help="Output image filename")
    args = parser.parse_args()
    
    # 1. Get Text Events
    timeline_text = get_event_timeline(args.video_path)
    print(f"\n--- Extracted Events ---\n{timeline_text}\n------------------------")
    
    # 2. Parse into Graph Nodes
    node_labels = parse_events_to_nodes(timeline_text)
    
    # 3. Draw & Save
    if node_labels:
        draw_causal_graph(node_labels, args.output)
    else:
        print("[ERROR] Could not extract valid events to plot.")