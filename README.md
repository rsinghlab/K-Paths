## K-Paths: Reasoning over Graph Paths for Drug Repurposing and Drug Interaction Prediction.
K-Paths is a retrieval framework that extracts structured, diverse, and biologically meaningful paths from knowledge graphs (KGs). Integrating these paths enables large language models (LLMs) and graph neural networks (GNNs) to predict unobserved drug-drug and drug-disease interactions effectively.
Beyond its scalability and efficiency, K-Paths uniquely bridges the gap between KGs and LLMs, providing explainable rationales for predicted interactions.

![Overview of K-Paths Framework](assets/K-Paths-overview.png)
K-Paths Overview: (1) Given a query about the effect of an entity ($u$) on another entity ($v$), (2) K-Paths extracts reasoning paths from an augmented KG connecting ($u$) and ($v$). (3) These paths are filtered for diversity and (4a) transformed into natural language descriptions for LLM inference. (4b) The retrieved paths can also be used to construct a subgraph, enabling GNNs to leverage more manageable information for training and prediction.

[📖 Paper](https://arxiv.org/abs/2502.13344) | [🤗 Hugging Face Dataset](https://huggingface.co/Tassy24)

# News 🎉
- K-Paths has been accepted as a conference paper at KDD 2025, Toronto, Canada.
- The K-Paths repo is currently undergoing careful construction.
  - Stay tuned for updates!
