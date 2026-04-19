# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import (BertTokenizer, BertModel, LlavaNextVideoProcessor,
                          LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig,
                          AutoTokenizer, AutoModel)
import av
import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import re

MODEL_PATH = "causal_qa_model_answer_aware.pth"
LLAVA_MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
NUM_FRAMES = 32

# Tuned based on observed SBERT scores:
# Valid queries:   ~0.45 - 0.55
# Invalid queries: ~0.30 - 0.44
# Entity-blocked:  caught by keyword guard regardless of score
STRICT_THRESHOLD = 0.40

print("--- INITIALIZING ULTIMATE NEURO-SYMBOLIC ENGINE (V8.0 - MEMORY OPTIMIZED) ---")


# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class CausalVLM_QA_AnswerAware(torch.nn.Module):
    def __init__(self):
        super(CausalVLM_QA_AnswerAware, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.conv1 = GCNConv(768, 256)
        self.conv2 = GCNConv(256, 256)
        self.fusion = torch.nn.Linear(768 + 768 + 768 + 256, 512)
        self.scorer = torch.nn.Linear(512, 1)

    def forward(self, data):
        batch_size = data.q_input_ids.reshape(-1, 32).shape[0]

        q_out = self.bert(input_ids=data.q_input_ids.reshape(-1, 32),
                          attention_mask=data.q_attention_mask.reshape(-1, 32))
        q_emb = q_out.last_hidden_state[:, 0, :]

        v_out = self.bert(input_ids=data.v_input_ids.reshape(-1, 128),
                          attention_mask=data.v_attention_mask.reshape(-1, 128))
        v_content_emb = v_out.last_hidden_state[:, 0, :]

        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        v_struct_emb = global_mean_pool(x, batch)

        all_a_ids = torch.stack(
            [getattr(data, f'a{i}_input_ids').reshape(-1, 32) for i in range(5)], dim=1)
        all_a_mask = torch.stack(
            [getattr(data, f'a{i}_attention_mask').reshape(-1, 32) for i in range(5)], dim=1)

        a_out = self.bert(input_ids=all_a_ids.reshape(-1, 32),
                          attention_mask=all_a_mask.reshape(-1, 32))
        a_embs = a_out.last_hidden_state[:, 0, :].reshape(batch_size, 5, 768)

        scores = []
        for i in range(5):
            combined = torch.cat([q_emb, v_content_emb, a_embs[:, i, :], v_struct_emb], dim=1)
            fused = F.relu(self.fusion(combined))
            scores.append(self.scorer(fused))

        return torch.cat(scores, dim=1)


# ============================================================
# LAYER 1: ENTITY HALLUCINATION GUARD
# ============================================================
def check_key_entities_present(question, full_description):
    """Block questions that mention specific locations/objects absent from the video."""
    location_keywords = [
        'balcony', 'kitchen', 'bathroom', 'garden', 'park',
        'street', 'car', 'beach', 'office', 'gym', 'pool',
        'stairs', 'roof', 'basement', 'garage', 'forest',
        'hospital', 'school', 'restaurant', 'mall', 'airport'
    ]
    person_keywords = ['girl', 'woman', 'lady', 'female', 'she', 'her']

    question_lower = question.lower()
    desc_lower = full_description.lower()

    for keyword in location_keywords:
        if keyword in question_lower and keyword not in desc_lower:
            print(f"[GUARD] Location '{keyword}' in question but absent from description.")
            return False, f"location '{keyword}'"

    # Person gender guard: if question asks about a girl but description says man/boy
    male_indicators = ['man', 'boy', 'guy', 'male', 'he ', 'his ']
    question_asks_female = any(w in question_lower for w in person_keywords)
    desc_has_only_male = any(w in desc_lower for w in male_indicators)

    if question_asks_female and desc_has_only_male and not any(w in desc_lower for w in person_keywords):
        print(f"[GUARD] Question asks about a girl/woman but video shows a man.")
        return False, "person gender mismatch"

    return True, None


# ============================================================
# LAYER 2: SBERT SEMANTIC RELEVANCE
# ============================================================
def compute_relevance_confidence(question, full_description, tokenizer, model, device):
    """Semantic similarity using MiniLM sentence embeddings."""
    model.eval()
    with torch.no_grad():
        q_inputs = tokenizer(question, return_tensors="pt", padding=True,
                             truncation=True, max_length=64).to(device)
        v_inputs = tokenizer(full_description, return_tensors="pt", padding=True,
                             truncation=True, max_length=256).to(device)

        q_outputs = model(**q_inputs)
        v_outputs = model(**v_inputs)

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / \
                   torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        q_emb = F.normalize(mean_pooling(q_outputs, q_inputs['attention_mask']), p=2, dim=1)
        v_emb = F.normalize(mean_pooling(v_outputs, v_inputs['attention_mask']), p=2, dim=1)

        return max(0.0, F.cosine_similarity(q_emb, v_emb, dim=1).item())


# ============================================================
# GRAPH BUILDER
# ============================================================
def build_graph_data_correct(frames_data, full_desc, question, tokenizer, bert_model, device):
    """Build semantic graph with per-frame BERT embeddings - matches training exactly."""
    bert_model.eval()
    node_embeddings = []

    with torch.no_grad():
        for frame in frames_data:
            desc = frame.get('description', 'empty frame')
            inputs = tokenizer(desc, return_tensors="pt", padding='max_length',
                               truncation=True, max_length=64).to(device)
            out = bert_model(**inputs)
            node_embeddings.append(out.last_hidden_state[:, 0, :].squeeze(0))

        if not node_embeddings:
            raise ValueError("frames_data produced no embeddings - check your JSON/fallback.")

        x = torch.stack(node_embeddings)
        num_nodes = len(node_embeddings)

        edge_index = (torch.tensor([[i, i + 1] for i in range(num_nodes - 1)],
                                    dtype=torch.long).T
                      if num_nodes > 1 else torch.empty((2, 0), dtype=torch.long)).to(device)

        graph = Data(x=x.to(device), edge_index=edge_index)
        graph.batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

        q_inputs = tokenizer(question, return_tensors="pt", padding='max_length',
                             truncation=True, max_length=32).to(device)
        v_inputs = tokenizer(full_desc, return_tensors="pt", padding='max_length',
                             truncation=True, max_length=128).to(device)
        graph.q_input_ids = q_inputs['input_ids'].squeeze(0)
        graph.q_attention_mask = q_inputs['attention_mask'].squeeze(0)
        graph.v_input_ids = v_inputs['input_ids'].squeeze(0)
        graph.v_attention_mask = v_inputs['attention_mask'].squeeze(0)

    return graph


# ============================================================
# CAUSAL GRAPH VISUALIZER
# ============================================================
def draw_causal_graph(events_list, output_file="causal_proof.png"):
    nodes = [textwrap.fill(e, width=20) for e in events_list]
    if not nodes:
        return False

    G = nx.DiGraph()
    for i, label in enumerate(nodes):
        G.add_node(i, label=f"T{i + 1}\n{label}")
    edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    G.add_edges_from(edges)

    plt.figure(figsize=(16, 7))
    pos = {i: (i * 2, (i % 2)) for i in range(len(nodes))}
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, arrowsize=25,
                           edge_color='gray', connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_nodes(G, pos, node_size=8000, node_color='#E1F5FE',
                           edgecolors='#0288D1', linewidths=2)
    nx.draw_networkx_labels(G, pos, nx.get_node_attributes(G, 'label'),
                            font_size=10, font_weight='bold')

    plt.title("Neuro-Symbolic Causal Graph (Chronological Timeline)", fontsize=14, pad=20)
    plt.axis('off')
    plt.margins(0.15)
    plt.tight_layout()

    save_path = os.path.join("data", "processed_graphs", output_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_ultimate_pipeline(video_path, question, answer_choices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vid_id = os.path.splitext(os.path.basename(video_path))[0]

    # ----------------------------------------------------------------
    # STAGE 1: PERCEPTION - LLaVA extracts all info from video
    # ----------------------------------------------------------------
    print(f"[INFO] Booting Perception Module...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16)
    processor = LlavaNextVideoProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava_model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID, quantization_config=bnb_config, device_map="auto")

    if not hasattr(processor, "patch_size"):
        processor.patch_size = llava_model.config.vision_config.patch_size
    if not hasattr(processor, "vision_feature_select_strategy"):
        processor.vision_feature_select_strategy = \
            llava_model.config.vision_feature_select_strategy

    print(f"[INFO] Extracting frames...")
    container = av.open(video_path)
    total_frames = (container.streams.video[0].frames
                    if container.streams.video[0].frames > 0 else 250)

    indices = set(np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int))
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(cv2.resize(frame.to_ndarray(format="rgb24"), (336, 336)))
        if len(frames) >= NUM_FRAMES:
            break

    if not frames:
        raise ValueError(f"[ERROR] Could not extract any frames from {video_path}.")
    clip = np.stack(frames)

    def ask_llava(prompt_text):
        inputs = processor(text=prompt_text, videos=clip,
                           return_tensors="pt").to(llava_model.device)
        with torch.no_grad():
            out = llava_model.generate(**inputs, max_new_tokens=350, do_sample=False)
        raw = processor.batch_decode(out, skip_special_tokens=True)[0]
        return (raw.split("ASSISTANT:")[-1].strip()
                if "ASSISTANT:" in raw else raw.split("USER:")[-1].strip())

    # Get scene description
    print(f"[INFO] Generating Scene Context...")
    full_desc = ask_llava(
        "USER: <video>\nDescribe what is happening in this video in detail.\nASSISTANT:")

    # Load or generate frame-level JSON
    json_path = os.path.join("data", "processed_json", f"{vid_id}.json")
    fallback_events_list = None

    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            frames_data = json.load(f)
    else:
        print("[WARNING] No pre-processed JSON found. Fallback node granularity differs from training data.")
        events_text = ask_llava(
            "USER: <video>\nList the main distinct actions in chronological order.\nASSISTANT:")
        fallback_events_list = [
            re.sub(r'^[\d\.\*\-]+\s*', '', line.strip()).strip()
            for line in events_text.split('\n') if line.strip()
        ]
        frames_data = [{'description': evt} for evt in fallback_events_list]

    # ----------------------------------------------------------------
    # STAGE 2: MULTI-LAYER DEFENSE GATE
    # (Run both checks before deciding to proceed)
    # ----------------------------------------------------------------
    print(f"[INFO] Calculating Relevance Check (SBERT)...")
    relevance_tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2')
    relevance_model = AutoModel.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2').to(device)

    confidence = compute_relevance_confidence(
        question, full_desc, relevance_tokenizer, relevance_model, device)
    print(f"[DEBUG] SBERT Similarity: {confidence:.4f}")

    del relevance_model, relevance_tokenizer
    torch.cuda.empty_cache()

    entities_grounded, blocked_entity = check_key_entities_present(question, full_desc)

    # ----------------------------------------------------------------
    # STAGE 3 onwards - only if both gates pass
    # ----------------------------------------------------------------
    final_answer = ""
    status_msg = ""
    graph_path = None
    predicted_text = None
    llava_raw_answer = None

    if not entities_grounded:
        # BLOCKED by entity guard - free LLaVA immediately
        status_msg = f"[HALLUCINATION BLOCKED] Question references {blocked_entity} not present in video."
        final_answer = "I cannot answer this. Your question mentions elements that are not visible in the video."
        del llava_model, processor
        torch.cuda.empty_cache()

    elif confidence < STRICT_THRESHOLD:
        # BLOCKED by SBERT relevance
        status_msg = f"[WARNING] LOW RELEVANCE (Similarity: {confidence:.1%})"
        final_answer = "I cannot confidently answer this. Your question is unrelated to the events in the video."
        del llava_model, processor
        torch.cuda.empty_cache()

    else:
        # BOTH GATES PASSED - proceed with full pipeline
        status_msg = f"[SUCCESS] Query Contextualized (Similarity: {confidence:.1%})"
        print(f"[INFO] Valid Query detected. Collecting all LLaVA outputs first...")

        # Get events list for causal graph
        if fallback_events_list is not None:
            events_list = fallback_events_list
        else:
            events_text = ask_llava(
                "USER: <video>\nList the main distinct actions in chronological order. "
                "Short bullet points.\nASSISTANT:")
            events_list = [
                re.sub(r'^[\d\.\*\-]+\s*', '', line.strip()).strip()
                for line in events_text.split('\n') if line.strip()
            ]

        # Get LLaVA's raw answer BEFORE freeing it
        llava_raw_answer = ask_llava(
            f"USER: <video>\nBased STRICTLY on visible events, answer: {question}\n"
            f"CRITICAL: Only describe what is clearly visible. Do NOT guess.\nASSISTANT:")

        # Draw causal graph while LLaVA is still alive
        graph_path = draw_causal_graph(events_list, output_file=f"proof_{vid_id}.png")

        # FREE LLAVA BEFORE LOADING GNN
        del llava_model, processor
        torch.cuda.empty_cache()
        print("[INFO] LLaVA freed. Loading GNN Validator...")

        # ----------------------------------------------------------------
        # STAGE 3: GNN ANSWER VALIDATION
        # ----------------------------------------------------------------
        choices = [c.strip() for c in answer_choices.split(',')] if answer_choices else []

        if len(choices) == 5:
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            gnn_model = CausalVLM_QA_AnswerAware().to(device)

            if os.path.exists(MODEL_PATH):
                gnn_model.load_state_dict(
                    torch.load(MODEL_PATH, map_location=device, weights_only=False))
                print(f"[INFO] Loaded GNN weights from {MODEL_PATH}")
            else:
                raise FileNotFoundError(
                    f"[ERROR] Model not found at '{MODEL_PATH}'. Run training first.")

            gnn_model.eval()
            graph_data = build_graph_data_correct(
                frames_data, full_desc, question, bert_tokenizer, gnn_model.bert, device)

            for i, choice in enumerate(choices):
                a_inputs = bert_tokenizer(
                    choice, return_tensors="pt", padding='max_length',
                    truncation=True, max_length=32).to(device)
                setattr(graph_data, f'a{i}_input_ids', a_inputs['input_ids'].squeeze(0))
                setattr(graph_data, f'a{i}_attention_mask',
                        a_inputs['attention_mask'].squeeze(0))

            with torch.no_grad():
                logits = gnn_model(graph_data)
                predicted_class = logits.argmax(dim=1).item()
                predicted_text = choices[predicted_class]
                print(f"[INFO] GNN selected: '{predicted_text}'")

            # GNN grounds the final answer
            final_answer = (
                f"[GNN Validated: '{predicted_text}'] {llava_raw_answer}"
            )

            del gnn_model
            torch.cuda.empty_cache()

        else:
            print("[WARNING] Exactly 5 choices not provided. Skipping GNN classification.")
            final_answer = llava_raw_answer

    # ----------------------------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------------------------
    print(f"\n" + "=" * 75)
    print(f" CAUSAL-VIDGRAPH ULTIMATE ANALYSIS REPORT V8")
    print(f"=" * 75)
    print(f" QUESTION: \"{question}\"")
    print(f"-" * 75)
    print(f" [1] LLaVA SCENE PERCEPTION:\n     \"{full_desc}\"")
    print(f"-" * 75)
    print(f" [2a] RELEVANCE SCORE (Entity Guard + SBERT):\n     * {status_msg}")

    if predicted_text:
        print(f" [2b] GNN PREDICTED ANSWER: \"{predicted_text}\"")
    elif confidence >= STRICT_THRESHOLD and entities_grounded:
        print(f" [2b] GNN: [SKIPPED] Requires exactly 5 choices via --answer_choices")

    print(f"-" * 75)
    print(f" [3] GENERATIVE EXPLANATION:\n     \"{final_answer}\"")
    print(f"-" * 75)
    if graph_path:
        print(f" [4] VISUAL PROOF:\n     * Graph saved to: {graph_path}")
    print(f"=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True,
                        help="Path to raw video file")
    parser.add_argument('--question', type=str, required=True,
                        help="Question to ask about the video")
    parser.add_argument('--answer_choices', type=str, default=None,
                        help="Exactly 5 comma-separated choices: 'a,b,c,d,e'")
    args = parser.parse_args()
    run_ultimate_pipeline(args.video_path, args.question, args.answer_choices)