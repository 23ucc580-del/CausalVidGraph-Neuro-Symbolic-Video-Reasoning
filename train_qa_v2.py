# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# --- CONFIGURATION ---
CSV_PATH = "data/next_qa/train.csv"
JSON_DIR = "data/processed_json"
GRAPH_DIR = "data/processed_graphs"
BATCH_SIZE = 16
EPOCHS = 50          # More epochs since LR scheduler will control overfitting
SAVE_PATH = "causal_qa_model_v2.pth"

print("--- INITIALIZING IMPROVED ANSWER-AWARE TRAINING (Dropout + LR Schedule) ---")


class NextQADataset(Dataset):
    def __init__(self, csv_path, json_dir, graph_dir):
        print(f"Loading QA Data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.graph_dir = graph_dir

        self.json_map = {}
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith(".json"):
                    vid_id = os.path.splitext(file)[0]
                    self.json_map[vid_id] = os.path.join(root, file)

        self.df['video'] = self.df['video'].astype(str)
        self.valid_indices = []

        for idx, row in self.df.iterrows():
            vid_id = row['video']
            graph_path = os.path.join(graph_dir, f"{vid_id}.pt")
            if os.path.exists(graph_path) and vid_id in self.json_map:
                self.valid_indices.append(idx)

        print(f"Found {len(self.valid_indices)} complete samples (Graph + JSON).")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.df.iloc[real_idx]

        vid_id = row['video']
        question = row['question']
        label = int(row['answer'])

        try:
            graph = torch.load(os.path.join(self.graph_dir, f"{vid_id}.pt"),
                               weights_only=False)
            json_path = self.json_map[vid_id]

            with open(json_path, 'r') as f:
                frames_data = json.load(f)

            full_desc = " ".join([f.get('description', '') for f in frames_data])
            if not full_desc.strip():
                full_desc = "Events occurring in video."

            v_inputs = self.tokenizer(full_desc, return_tensors="pt",
                                      padding='max_length', truncation=True, max_length=128)
            q_inputs = self.tokenizer(question, return_tensors="pt",
                                      padding='max_length', truncation=True, max_length=32)

            graph.q_input_ids      = q_inputs['input_ids'].squeeze(0)
            graph.q_attention_mask = q_inputs['attention_mask'].squeeze(0)
            graph.v_input_ids      = v_inputs['input_ids'].squeeze(0)
            graph.v_attention_mask = v_inputs['attention_mask'].squeeze(0)
            graph.y = torch.tensor(label, dtype=torch.long)

            ans_cols = ['a0', 'a1', 'a2', 'a3', 'a4']
            for i, col in enumerate(ans_cols):
                ans_text = str(row[col])
                a_inputs = self.tokenizer(ans_text, return_tensors="pt",
                                          padding='max_length', truncation=True, max_length=32)
                setattr(graph, f'a{i}_input_ids',      a_inputs['input_ids'].squeeze(0))
                setattr(graph, f'a{i}_attention_mask', a_inputs['attention_mask'].squeeze(0))

            return graph

        except Exception as e:
            print(f"CRITICAL ERROR loading {vid_id}: {e}")
            for offset in range(1, 6):
                try:
                    return self.__getitem__((idx + offset) % len(self))
                except:
                    continue
            raise RuntimeError(f"Could not load any valid sample near index {idx}")


class CausalVLM_QA_AnswerAware(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CausalVLM_QA_AnswerAware, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # IMPROVEMENT 1: Unfreeze last 3 BERT layers instead of just 1
        # Why: More BERT layers fine-tuned = embeddings better suited to video QA domain
        # Old: only layer[-1] was trained ? BERT barely adapted to your task
        # New: layers[-3], [-2], [-1] all trained ? BERT adapts more to video descriptions
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        self.conv1 = GCNConv(768, 256)
        self.conv2 = GCNConv(256, 256)

        # IMPROVEMENT 2: Batch Normalization after GCN layers
        # Why: Stabilizes training, prevents exploding/vanishing gradients in the graph
        # Old: raw GCN output fed directly to fusion ? unstable gradients
        # New: normalized output ? stable, faster convergence
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)

        self.fusion = torch.nn.Linear(768 + 768 + 768 + 256, 512)

        # IMPROVEMENT 3: Dropout layers
        # Why: Prevents overfitting (your 88% train vs 37% val gap is classic overfitting)
        # How: During training, randomly zeroes 30% of neurons each forward pass
        # Effect: Model cannot memorize training data ? forces generalizable features
        self.dropout_bert = torch.nn.Dropout(dropout_rate)   # after BERT embeddings
        self.dropout_gcn  = torch.nn.Dropout(dropout_rate)   # after GCN
        self.dropout_fuse = torch.nn.Dropout(dropout_rate)   # after fusion layer

        # IMPROVEMENT 4: Deeper scorer with intermediate layer
        # Why: Single linear 512->1 is too shallow to learn complex Q+V+A relationships
        # Old: fusion(2560->512) -> scorer(512->1)   [1 decision layer]
        # New: fusion(2560->512) -> hidden(512->128) -> scorer(128->1)  [2 decision layers]
        self.hidden  = torch.nn.Linear(512, 128)
        self.scorer  = torch.nn.Linear(128, 1)

    def forward(self, data):
        batch_size = data.q_input_ids.reshape(-1, 32).shape[0]

        # Encode question with dropout
        q_out = self.bert(input_ids=data.q_input_ids.reshape(-1, 32),
                          attention_mask=data.q_attention_mask.reshape(-1, 32))
        q_emb = self.dropout_bert(q_out.last_hidden_state[:, 0, :])

        # Encode video description with dropout
        v_out = self.bert(input_ids=data.v_input_ids.reshape(-1, 128),
                          attention_mask=data.v_attention_mask.reshape(-1, 128))
        v_content_emb = self.dropout_bert(v_out.last_hidden_state[:, 0, :])

        # GCN with batch norm + dropout
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        v_struct_emb = self.dropout_gcn(global_mean_pool(x, batch))

        # Encode all 5 answers in one parallel BERT pass
        all_a_ids  = torch.stack([getattr(data, f'a{i}_input_ids').reshape(-1, 32)
                                   for i in range(5)], dim=1)
        all_a_mask = torch.stack([getattr(data, f'a{i}_attention_mask').reshape(-1, 32)
                                   for i in range(5)], dim=1)
        a_out  = self.bert(input_ids=all_a_ids.reshape(-1, 32),
                           attention_mask=all_a_mask.reshape(-1, 32))
        a_embs = self.dropout_bert(
            a_out.last_hidden_state[:, 0, :]).reshape(batch_size, 5, 768)

        # Score all 5 answers
        scores = []
        for i in range(5):
            combined = torch.cat([q_emb, v_content_emb, a_embs[:, i, :], v_struct_emb], dim=1)
            fused  = self.dropout_fuse(F.relu(self.fusion(combined)))
            hidden = self.dropout_fuse(F.relu(self.hidden(fused)))
            scores.append(self.scorer(hidden))

        return torch.cat(scores, dim=1)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    dataset = NextQADataset(CSV_PATH, JSON_DIR, GRAPH_DIR)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = CausalVLM_QA_AnswerAware(dropout_rate=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,
                                  weight_decay=1e-4)   # L2 regularization
    criterion = torch.nn.CrossEntropyLoss()

    # IMPROVEMENT 5: Learning Rate Scheduler
    # Why: Fixed lr=0.0001 causes two problems:
    #   - Too large in late epochs ? model oscillates around minimum, never settles
    #   - Never adapts to loss plateau ? wastes compute
    # How ReduceLROnPlateau works:
    #   - Monitors training loss every epoch
    #   - If loss doesn't improve for 'patience=4' epochs ? multiply lr by 0.5
    #   - Keeps halving until lr hits min_lr=1e-6
    # Effect: Fast learning early, fine-grained tuning late ? better final weights
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # minimize loss
        factor=0.5,        # halve the lr when triggered
        patience=4,        # wait 4 epochs of no improvement before reducing
        min_lr=1e-6,       # never go below this
        verbose=True       # print when lr changes
    )

    best_loss = float('inf')
    best_acc  = 0.0

    print(f"\nSTARTING IMPROVED TRAINING ({EPOCHS} epochs)...")
    print(f"Improvements active:")
    print(f"  - Dropout (0.3) on BERT, GCN, Fusion")
    print(f"  - 3 BERT layers fine-tuned (was 1)")
    print(f"  - BatchNorm after GCN layers")
    print(f"  - Deeper scorer (512->128->1)")
    print(f"  - ReduceLROnPlateau scheduler")
    print(f"  - L2 weight decay (1e-4)\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        progress = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)

        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch)
            loss    = criterion(outputs, batch.y)
            loss.backward()

            # IMPROVEMENT 6: Gradient clipping
            # Why: Prevents exploding gradients (occasional NaN loss you may have seen)
            # Clips gradient norm to max 1.0 if it exceeds that
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            preds   = outputs.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total   += len(batch.y)

            progress.set_postfix({"Loss": f"{loss.item():.4f}",
                                   "Acc":  f"{correct/total:.2%}",
                                   "LR":   f"{optimizer.param_groups[0]['lr']:.2e}"})

        epoch_loss = total_loss / len(loader)
        epoch_acc  = correct / total

        print(f"Epoch {epoch:02d} | Loss={epoch_loss:.4f} | "
              f"Acc={epoch_acc:.2%} | LR={optimizer.param_groups[0]['lr']:.2e}")

        # Step the scheduler based on epoch loss
        scheduler.step(epoch_loss)

        # Save best model by accuracy
        if epoch_acc > best_acc:
            best_acc  = epoch_acc
            best_loss = epoch_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  >>> New best model saved (Acc={best_acc:.2%})")

    print(f"\nTraining complete.")
    print(f"Best Training Accuracy : {best_acc:.2%}")
    print(f"Model saved to         : {SAVE_PATH}")
    print(f"\nNOTE: Update MODEL_PATH in ultimate_demo.py to '{SAVE_PATH}'")


if __name__ == "__main__":
    train()