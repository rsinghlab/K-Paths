## K-Paths: Reasoning over Graph Paths for Drug Repurposing and Drug Interaction Prediction.
K-Paths is a retrieval framework that extracts structured, diverse, and biologically meaningful paths from knowledge graphs (KGs). These extracted paths enables large language models (LLMs) and graph neural networks (GNNs) to predict unobserved drug-drug and drug-disease interactions more effectively.
Beyond its scalability and efficiency, K-Paths uniquely bridges the gap between KGs and LLMs, providing explainable rationales for predictions.

![Overview of K-Paths Framework](assets/K-Paths-overview.png)
K-Paths Overview: (1) Given a query about the effect of an entity ($u$) on another entity ($v$), (2) K-Paths extracts reasoning paths from an augmented Knowledge graph. (3) These paths are filtered for diversity and (4a) transformed into natural language descriptions for LLM inference. (4b) The retrieved paths can also be used to build a subgraph for GNN-based predictions.

[📖 Paper](https://arxiv.org/abs/2502.13344) | [🤗 Hugging Face Dataset](https://huggingface.co/Tassy24)

---
# News 🎉
- K-Paths has been accepted as a conference paper at KDD 2025, Toronto, Canada.
- The repo is currently under active development — stay tuned for new features!

---
## Features
- Extract multi-hop reasoning paths between entity pairs from a knowledge graph.
- Generate subgraphs via pruning, suitable for downstream GNN training.
- Supports zero-shot LLM inference and automatic evaluation (exact-match using regex & BERTScore).

---
## Coming Soon
- Custom Augmented Networks: Generate and modify augmented knowledge graphs by combining Hetionet with your own training data.
- LLM Fine-Tuning: Add support to fine-tune large language models using path-based data.
- GNN Training & Inference: Add support for fine-tuning and running GNN models on the extracted subgraphs.

---
## Dataset support:
- Drugbank (Drug–drug interaction type classification),
- PharmacotherapyDB (Drug repurposing), 
- DDinter (Drug–drug interaction severity classification)

## Usage
- Requires Python 3.10+
- Install dependencies (pip install -r requirements.txt)

## Quick Start: Use Reasoning Paths:
- To use the multihop paths directly for inference, download via [🤗 Hugging Face Dataset](https://huggingface.co/Tassy24)

## To Reproduce/Extract Paths and Subgraphs from Scratch

- **Step 1:** Download the required data (Hetionet, etc.)  
  Download the dataset bundle from:  
  [📦 data.zip (Google Drive)](https://drive.google.com/file/d/1_6meo_nB2RqHrVM9pqCBA67FQ6PR4QiI/view?usp=drive_link)
  - Extract the `data.zip` file so that the structure looks like:
    ```
    K-Paths/
    ├── data/
    │   └── subgraphs/
    │   └── paths/
    │       └── ...
    ```

- **Step 2:** Create Augmented network for the supported datasets:  
  - Example:
    ```python
    python k-paths/src/create_augmented_network.py
    ```

- **Step 3:** Extract K reasoning paths:  
  - Example:
    ```python
    python k-paths/src/get-Kpaths.py \
      --dataset ddinter \
      --split test \
      --mode K-paths \
      --add_reverse_edges
    ```

- **Step 4:** Create subgraphs for GNN input:  
  - Example:
    ```python
    python k-paths/src/get-subgraph.py
    ```

## 🔍 To Run Zero-Shot Inference and Evaluation

- Step 1:Run zero-shot inference using a supported LLM:  
  *(Use `llm/llm_inference_v2.py` for Tx-Gemma models)*  
  - ⚠️ **Tip:** Use `--help` to view all supported flags and options (e.g., `--use_kg`, `--use_options`, or `--option_style`)
  - Example:
    ```python
    python llm/llm_inference.py \
      --dataset_path data/paths/drugbank_test_add_reverse.json \
      --dataset_name drugbank \
      --output_dir outputs/drugbank \
      --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct \
      --use_kg
    ```

- **Step 2:** Evaluate the model's predictions using regex matching:  
  - Example:
    ```python
    python llm/evaluate_llm_regex.py \
      --prediction_path output/google-txgemma-27b-chat-outputs-paths-drugbank_test_add_reverse-json-predictions.csv \
      --dataset drugbank \
      --model_style tx_gemma
    ```
