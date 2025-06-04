## K-Paths: Reasoning over Graph Paths for Drug Repurposing and Drug Interaction Prediction.
K-Paths is a retrieval framework that extracts structured, diverse, and biologically meaningful paths from knowledge graphs (KGs). These extracted paths enables large language models (LLMs) and graph neural networks (GNNs) to predict unobserved drug-drug and drug-disease interactions more effectively.
Beyond its scalability and efficiency, K-Paths uniquely bridges the gap between KGs and LLMs, providing explainable rationales for predictions.

![Overview of K-Paths Framework](assets/K-Paths-overview.png)
K-Paths Overview: (1) Given a query about the effect of an entity ($u$) on another entity ($v$), (2) K-Paths extracts reasoning paths from an augmented Knowledge graph. (3) These paths are filtered for diversity and (4a) transformed into natural language descriptions for LLM inference. (4b) The retrieved paths can also be used to build a subgraph for GNN-based predictions.

[📖 Paper](https://arxiv.org/abs/2502.13344) | [🤗 Hugging Face Dataset](https://huggingface.co/Tassy24)

---
# News 
- May'25: K-Paths has been accepted as a conference paper at [KDD 2025](https://kdd2025.kdd.org/), Toronto, Canada! 🎉
- Feb'25: Feb 2025: Read the K-Paths manuscript on [arXiv](https://arxiv.org/abs/2502.13344)
---
## Features
- Extract multi-hop reasoning paths between entity pairs from a knowledge graph.
- Generate subgraphs via pruning, suitable for GNN training.
- Supports zero-shot LLM inference and automatic evaluation (exact-match using regex & BERTScore).

---
## Supported Datasets
- Drugbank (Drug–drug interaction type classification),
- PharmacotherapyDB (Drug repurposing), 
- DDinter (Drug–drug interaction severity classification)

---

## Installation
- Requires Python 3.10+
### Create and activate a virtual environment
```bash
python3.10 -m venv .kpaths-env
source .kpaths-env/bin/activate  # On macOS/Linux
# .\kpaths-env\Scripts\activate  # On Windows
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Use Pre-Extracted Reasoning Paths
- To use the multihop paths directly for inference, download via [🤗 Hugging Face Dataset](https://huggingface.co/Tassy24)

## Path and Subgraph Extraction from scratch

- **Step 1:** Download data
  GET the dataset bundle from:  
  [📦 data.zip (Google Drive)](https://drive.google.com/file/d/1_6meo_nB2RqHrVM9pqCBA67FQ6PR4QiI/view?usp=drive_link)
  - Extract the `data.zip` file so that the structure looks like:
    ```
    K-Paths/
    ├── data/
    │   └── subgraphs/
    │   └── paths/
    │       └── ...
    ```

- **Step 2:** Create Augmented KG:  
  ```python
  python k-paths/create_augmented_network.py
  ```

- **Step 3:** Extract `K` reasoning paths:  
  ```python
  python k-paths/get-Kpaths.py \
    --dataset ddinter \
    --split test \
    --mode K-paths \
    --add_reverse_edges
  ```

- **Step 4:** Create subgraphs for GNN input:  
  ```python
  python k-paths/get-subgraph.py
  ```

## Zero-Shot LLM Inference and Evaluation

- Inference
  *(Use `llm/llm_inference_v2.py` for Tx-Gemma models)*  
  ```python
  python llm/llm_inference.py \
    --dataset_path data/paths/drugbank_test_add_reverse.json \
    --dataset_name drugbank \
    --output_dir outputs/drugbank \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --use_kg
  ```
> Use `--help` to see flags like `--use_options`, `--option_style`.

- Evaluation
  ```python
  python llm/evaluate_llm_regex.py \
    --prediction_path output/google-txgemma-27b-chat-outputs-paths-drugbank_test_add_reverse-json-predictions.csv \
    --dataset drugbank \
    --model_style tx_gemma
  ```

---
## GNN Training & Evaluation

- Train RGCN
  ```python
  python gnn/train.py \
    --seed "${SEED}" \
    --train_file_path path_to_your_train_set.csv \
    --hetionet_triplet_file path_to_hetionet.txt \
    --node_file path_to_node2id.json \
    --entity_drug_file path_to_BKG_entity2Id.json \
    --use_text_embeddings \
    --model_save_path "trained_model_seed${SEED}.pt"

  ```

- Evaluate Trained Model
  ```python
  python gnn/eval.py \
    --model_path "trained_model_seed${SEED}.pt" \
    --train_file_path path_to_your_train_set.csv \
    --test_file path_to_your_test_set.csv \
    --hetionet_triplet_file path_to_hetionet.txt \
    --node_file path_to_node2id.json \
    --use_text_embeddings \
  ```
> Note: Most optional arguments (e.g., `--embedding_dim`, `--log_file`, `--output_predictions`) have sensible defaults.

## Coming Soon
- Custom Augmented Networks: Generate and modify augmented knowledge graphs by combining Hetionet with your own training data.
- LLM Fine-Tuning: Add support to fine-tune large language models using path-based data.