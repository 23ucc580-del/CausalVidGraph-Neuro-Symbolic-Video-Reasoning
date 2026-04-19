# -*- coding: utf-8 -*-
# live_inference.py — Fixed version
# Bug fixed: Node features now use BERT CLS embeddings per frame description
# instead of torch.randn (which gave identical random noise to every video)

import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import (BertTokenizer, BertModel,
                          LlavaNextVideoProcessor,
                          LlavaNextVideoForConditionalGeneration,
                          BitsAndBytesConfig)
import av
import numpy as np
import re

# --- CONFIGURATION ---
MODEL_PATH    = "causal_qa_model_answer_aware.pth"
LLAVA_MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
NUM_FRAMES    = 32

print("--- INITIALIZING LIVE NEURO-SYMBOLIC ENGINE (FIXED) ---")


# ============================================================
# 1. MODEL ARCHITECTURE
#    Must match the architecture used during training exactly.
#    This uses the Answer-Aware scorer, not the old 5-class
#    classifier.  If your .pth file was trained with
#    causal_qa_model_answer_aware.pth, use this class.
# ============================================================
class CausalVLM_QA_AnswerAware(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert   = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        self.conv1  = GCNConv(768, 256)
        self.conv2  = GCNConv(256, 256)
        self.fusion = torch.nn.Linear(768 + 768 + 768 + 256, 512)
        self.scorer = torch.nn.Linear(512, 1)

    def forward(self, data):
        bs = data.q_input_ids.reshape(-1, 32).shape[0]

        q_emb = self.bert(
            input_ids=data.q_input_ids.reshape(-1, 32),
            attention_mask=data.q_attention_mask.reshape(-1, 32)
        ).last_hidden_state[:, 0, :]

        v_emb = self.bert(
            input_ids=data.v_input_ids.reshape(-1, 128),
            attention_mask=data.v_attention_mask.reshape(-1, 128)
        ).last_hidden_state[:, 0, :]

        x, ei, b = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, ei))
        x = self.conv2(x, ei)
        g_emb = global_mean_pool(x, b)

        all_ids  = torch.stack([getattr(data, f'a{i}_input_ids').reshape(-1, 32)
                                for i in range(5)], dim=1)
        all_mask = torch.stack([getattr(data, f'a{i}_attention_mask').reshape(-1, 32)
                                for i in range(5)], dim=1)
        a_embs = self.bert(
            input_ids=all_ids.reshape(-1, 32),
            attention_mask=all_mask.reshape(-1, 32)
        ).last_hidden_state[:, 0, :].reshape(bs, 5, 768)

        scores = []
        for i in range(5):
            c = torch.cat([q_emb, v_emb, a_embs[:, i, :], g_emb], dim=1)
            scores.append(self.scorer(F.relu(self.fusion(c))))
        return torch.cat(scores, dim=1)


# ============================================================
# 2. PERCEPTION MODULE — LLaVA Frame Captioning
# ============================================================
def get_video_description(video_path):
    """
    Runs LLaVA on the video and returns:
      - description : full scene description string
      - frames_data : list of {description: ...} dicts per frame/segment
    """
    print(f"[INFO] Watching video: {os.path.basename(video_path)}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    processor = LlavaNextVideoProcessor.from_pretrained(LLAVA_MODEL_ID)
    model     = LlavaNextVideoForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Frame extraction
    container   = av.open(video_path)
    total_frames = container.streams.video[0].frames or 250
    indices     = set(np.linspace(0, total_frames - 1, NUM_FRAMES).astype(int))

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > max(indices):
            break
        if i in indices:
            img = frame.to_image().resize((336, 336))
            frames.append(img)
    container.close()

    if not frames:
        raise ValueError("No frames extracted from video.")

    # LLaVA generation
    prompt = ("USER: <video>\n"
              "Describe this video scene by scene. "
              "For each part, mention what people are doing and what objects are visible. "
              "Be specific about actions and items.\n"
              "ASSISTANT:")

    inputs = processor(text=prompt, videos=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=350,
                             do_sample=False, num_beams=1, use_cache=True,
                             repetition_penalty=1.3)

    generated = processor.batch_decode(out, skip_special_tokens=True)[0]
    description = generated.split("ASSISTANT:")[-1].strip()
    if len(description) < 10:
        description = "The video shows various scenes with people and objects in motion."

    # Free LLaVA from VRAM before loading GNN
    del model, processor, inputs
    torch.cuda.empty_cache()
    print("[INFO] LLaVA freed. Building frame data...")

    # Split description into per-frame entries for graph nodes
    sentences = re.split(r'(?<=[.!?])\s+', description.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
    if not sentences:
        sentences = [description]

    frames_data = [{'description': s} for s in sentences]

    print(f"[INFO] Scene description: {description[:120]}...")
    return description, frames_data


# ============================================================
# 3. GRAPH BUILDER — BERT Node Features
#    FIX: Replaced torch.randn (Bug #1) with BERT CLS embeddings
#    Each node now represents the semantic meaning of one
#    frame/sentence — exactly matching training data format.
# ============================================================
def build_graph_data_correct(frames_data, full_desc, question,
                              answer_choices, tokenizer, bert_model, device):
    """
    Builds a PyG Data object with:
      - x           : BERT CLS embeddings per frame  [N, 768]
      - edge_index  : sequential edges i -> i+1
      - q/v tokens  : for question and video description
      - a0..a4 tokens: for the 5 answer choices
    """
    bert_model.eval()
    node_embeddings = []

    with torch.no_grad():
        for frame in frames_data:
            desc   = frame.get('description', 'empty frame')
            inputs = tokenizer(desc, return_tensors="pt",
                               padding='max_length', truncation=True,
                               max_length=64).to(device)
            out    = bert_model(**inputs)
            # CLS token = semantic summary of this frame
            node_embeddings.append(out.last_hidden_state[:, 0, :].squeeze(0))

    if not node_embeddings:
        raise ValueError("No frame embeddings produced.")

    x          = torch.stack(node_embeddings)                      # [N, 768]
    num_nodes  = len(node_embeddings)
    edge_index = (torch.tensor([[i, i + 1] for i in range(num_nodes - 1)],
                                dtype=torch.long).T
                  if num_nodes > 1
                  else torch.empty((2, 0), dtype=torch.long)).to(device)

    # Tokenize question and video description
    q_enc = tokenizer(question,  return_tensors="pt",
                      padding='max_length', truncation=True, max_length=32)
    v_enc = tokenizer(full_desc, return_tensors="pt",
                      padding='max_length', truncation=True, max_length=128)

    graph = Data(x=x, edge_index=edge_index)
    graph.q_input_ids      = q_enc['input_ids'].squeeze(0)
    graph.q_attention_mask = q_enc['attention_mask'].squeeze(0)
    graph.v_input_ids      = v_enc['input_ids'].squeeze(0)
    graph.v_attention_mask = v_enc['attention_mask'].squeeze(0)
    graph.batch            = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Tokenize all 5 answer choices
    for i, ans_text in enumerate(answer_choices):
        a_enc = tokenizer(str(ans_text), return_tensors="pt",
                          padding='max_length', truncation=True, max_length=32)
        setattr(graph, f'a{i}_input_ids',      a_enc['input_ids'].squeeze(0))
        setattr(graph, f'a{i}_attention_mask', a_enc['attention_mask'].squeeze(0))

    return graph.to(device)


# ============================================================
# 4. MAIN INFERENCE PIPELINE
# ============================================================
def run_live_inference(video_path, question, answer_choices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # -- Step A: Visual Perception ------------------------------
    try:
        description, frames_data = get_video_description(video_path)
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback; traceback.print_exc()
        return

    # -- Step B: Load GNN ---------------------------------------
    print("[INFO] Loading GNN model...")
    gnn_model = CausalVLM_QA_AnswerAware().to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] {MODEL_PATH} not found. Using untrained weights.")
    else:
        gnn_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=False))
        print(f"[INFO] Loaded weights from {MODEL_PATH}")

    gnn_model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # -- Step C: Build Semantic Graph ---------------------------
    # This now uses BERT CLS embeddings — NOT random noise.
    # Each graph node = semantic meaning of one video frame/event.
    print("[INFO] Building semantic causal graph...")
    try:
        graph_data = build_graph_data_correct(
            frames_data, description, question,
            answer_choices, tokenizer, gnn_model.bert, device
        )
    except Exception as e:
        print(f"[ERROR] Graph building failed: {e}")
        return

    # -- Step D: GNN Inference ----------------------------------
    print("[INFO] Running causal reasoning...")
    with torch.no_grad():
        logits     = gnn_model(graph_data)              # [1, 5]
        probs      = F.softmax(logits, dim=1)
        pred_idx   = logits.argmax(dim=1).item()
        pred_text  = answer_choices[pred_idx]
        confidence = probs[0][pred_idx].item()

    # Free GNN from VRAM
    del gnn_model
    torch.cuda.empty_cache()

    # -- Step E: Final Report -----------------------------------
    print(f"\n{'='*70}")
    print(f"  CAUSAL-VIDGRAPH ANALYSIS REPORT")
    print(f"{'='*70}")
    print(f"  QUESTION  : {question}")
    print(f"  CHOICES   : {', '.join(answer_choices)}")
    print(f"{'-'*70}")
    print(f"  [1] SCENE PERCEPTION (LLaVA):")
    print(f"      \"{description[:300]}{'...' if len(description)>300 else ''}\"")
    print(f"{'-'*70}")
    print(f"  [2] GNN CAUSAL REASONING:")
    print(f"      Graph nodes : {len(frames_data)} semantic frame embeddings")
    print(f"      Predicted   : '{pred_text}' (Choice {pred_idx})")
    print(f"      Confidence  : {confidence:.2%}")
    print(f"{'-'*70}")
    print(f"  [3] ANSWER SCORES (all 5 choices):")
    for i, (choice, score) in enumerate(zip(answer_choices, probs[0].tolist())):
        marker = " <-- PREDICTED" if i == pred_idx else ""
        print(f"      [{i}] {choice:<30} {score:.2%}{marker}")
    print(f"{'='*70}\n")


# ============================================================
# 5. ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CausalVidGraph Live Inference (Fixed)")
    parser.add_argument('--video_path',     required=True,
                        help="Path to input video file (.mp4, .avi, etc.)")
    parser.add_argument('--question',       required=True,
                        help="Question to ask about the video")
    parser.add_argument('--answer_choices', required=True,
                        help="Comma-separated list of exactly 5 answer choices")
    args = parser.parse_args()

    choices = [c.strip() for c in args.answer_choices.split(',')]
    if len(choices) != 5:
        print(f"[ERROR] Exactly 5 answer choices required. Got {len(choices)}.")
        exit(1)

    run_live_inference(args.video_path, args.question, choices)