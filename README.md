
# LRAGE: Legal Retrieval Augmented Generation Evaluation Tool

LRAGE (Legal Retrieval Augmented Generation Evaluation) is an open-source toolkit designed to evaluate Large Language Models (LLMs) in a Retrieval-Augmented Generation (RAG) setting, specifically tailored for the legal domain. LRAGE is developed to address the unique challenges that Legal AI researchers face, such as building and evaluating retrieval-augmented systems effectively. It seamlessly integrates datasets and tools to help researchers in evaluating LLM performance on legal tasks without cumbersome engineering overhead.

## Features

- **Legal Domain Focused Evaluation**: LRAGE is specifically developed for evaluating LLMs in a RAG setting with datasets and document collections from the legal domain, such as Pile-of-law and LegalBench.
- **Retriever & Reranker Integration**: Easily integrate and evaluate different retrievers and rerankers. LRAGE modularizes retrieval and reranking components, allowing for flexible experimentation.
- **Out-of-the-box Indexing**: Comes with pre-generated BM25 indices and embeddings for Pile-of-law, reducing the setup effort for researchers.
- [ ] (In progress)**LLM-as-a-Judge**: Experimental feature where LLMs can be used to evaluate the output quality based on rubrics or prompts, aiming to reduce manual evaluation effort.
- [ ] (In progress)**Graphical User Interface**: A GUI demo for intuitive usage, making the tool accessible even for those who are not deeply familiar with command-line interfaces. 

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/LRAGE.git
    cd LRAGE
    ```
2. Install:
    ```bash
    pip install -e .
    ```

## Quick Start

To evaluate a model on a sample dataset using the RAG setting, follow these steps:

1. Prepare your dataset in the supported format.
2. Run the evaluation script:
    ```bash
    lrage \
    --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks abercrombie \
    --batch_size 8 --trust_remote_code --device cuda \
    --retrieve_docs --top_k 3 --retriever pyserini --retriever_args retriever_type=bm25,bm25_index_path=YOUR_INDEX_PATH \
    --rerank --reranker rerankers --reranker_args reranker_type=colbert \
    ```
