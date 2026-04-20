# CausalVidGraph: Neuro-Symbolic Video Reasoning

**CausalVidGraph** is a neuro-symbolic framework designed for Explainable Video Question Answering (Video QA). It addresses a major flaw in current vision-language models: the tendency to confidently hallucinate answers when asked about events or objects that do not actually exist in the video.

This system acts as a verifier, not just a text predictor, grounding its answers in actual video content and explicitly mapping out the causal chain of events.

## 🚀 Key Features

* **5-Layer Hallucination Defense:** Actively intercepts and blocks queries with false premises before inference (achieving a 99% block rate on invalid queries).
* **Explicit Causal Reasoning:** Uses a directed Temporal Graph Convolutional Network (GCN) to model how events unfold over time.
* **Open-Vocabulary Perception:** Leverages LLaVA-NeXT-Video-7B to generate rich, natural-language scene descriptions without being limited to a fixed object vocabulary.
* **Explainability:** Every generated answer is accompanied by a directed causal proof graph showing the exact reasoning chain.

## 🧠 Architecture Overview

The system operates in three main phases:
1. **Perception (LLaVA):** Extracts frames and generates text descriptions of the video content.
2. **Graph Construction (BERT & GCN):** Converts descriptions into BERT-encoded nodes connected by directed sequential edges, representing the flow of time.
3. **Answer-Aware Fusion:** A multi-path scoring mechanism that ranks five multiple-choice answers based on the generated temporal graph.

## 📊 Results

Tested on a subset of the **NExT-QA** benchmark (3,558 training samples, 300 test samples):
* **Accuracy:** 45.67% (25.67 points above the random baseline).
* **Hallucination Defense:** 99.0% True Negative Rate (TNR) with only a 1.0% False Positive Rate (FPR).

## 🛠️ Tech Stack

* **Vision-Language Model:** LLaVA-NeXT-Video-7B (Quantized to 4-bit NF4 via QLoRA)
* **Embeddings:** BERT (`bert-base-uncased`), SBERT (`all-MiniLM-L6-v2`)
* **Graph Neural Networks:** PyTorch Geometric (PyG)
* **Causal Graphs:** NetworkX

## 📝 Authors

* **Noor** (23UCC580)
* **Vidhi Jain** (23UCC613)
* **Druti Jain** (23UCC539)

*Department of Communication and Computer Engineering* *The LNM Institute of Information Technology, Jaipur (2026-2027)*
