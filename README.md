
# LRAGE: Legal Retrieval Augmented Generation Evaluation Tool

LRAGE (Legal Retrieval Augmented Generation Evaluation, pronounced as 'large') is an open-source toolkit designed to evaluate Large Language Models (LLMs) in a Retrieval-Augmented Generation (RAG) setting, specifically tailored for the legal domain.  

LRAGE is developed to address the unique challenges that Legal AI researchers face, such as building and evaluating retrieval-augmented systems effectively. It seamlessly integrates datasets and tools to help researchers in evaluating LLM performance on legal tasks without cumbersome engineering overhead.

## Features

- **Legal Domain Focused Evaluation**: LRAGE is specifically developed for evaluating LLMs in a RAG setting with datasets and document collections from the legal domain, such as [Pile-of-law](https://huggingface.co/datasets/pile-of-law/pile-of-law) and [LegalBench](https://github.com/HazyResearch/legalbench).  

- **Retriever & Reranker Integration**: Easily integrate and evaluate different retrievers and rerankers. LRAGE modularizes retrieval and reranking components, allowing for flexible experimentation.  

- **Pre-compiled indexes for the legal domain**: Comes with pre-generated BM25 indices and embeddings for Pile-of-law, reducing the setup effort for researchers.  

- (In progress)**LLM-as-a-Judge**: A feature where LLMs are used to evaluate the quality of LLM responses on an instance-by-instance basis, using customizable rubrics within the RAG setting.  

- (In progress)**Graphical User Interface**: A GUI demo for intuitive usage, making the tool accessible even for those who are not deeply familiar with command-line interfaces.  

## Extensions for RAG Evaluation from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

1.	**Addition of Retriever and Reranker abstract classes**: LRAGE introduces [retriever](https://github.com/hoorangyee/LRAGE/blob/main/lrage/api/retriever.py) and [reranker](https://github.com/hoorangyee/LRAGE/blob/main/lrage/api/reranker.py) abstract classes in the [lrage/api/](https://github.com/hoorangyee/LRAGE/tree/main/lrage/api). These additions allow the process of building requests in the [api.task.Task](https://github.com/hoorangyee/LRAGE/blob/b24b7dc253fdfaa82cd926d1d1147f8a18ec69bf/lrage/api/task.py#L179) class’s [build_all_requests()](https://github.com/hoorangyee/LRAGE/blob/b24b7dc253fdfaa82cd926d1d1147f8a18ec69bf/lrage/api/task.py#L376) method to go through both retrieval and reranking steps, enhancing the evaluation process for RAG.  


2.	**Extensible Retriever and Reranker implementations**: While maintaining the same structure as lm-evaluation-harness, LRAGE allows for the flexible integration of different retriever and reranker implementations. Just as lm-evaluation-harness provides an abstract [LM class](https://github.com/hoorangyee/LRAGE/blob/b24b7dc253fdfaa82cd926d1d1147f8a18ec69bf/lrage/api/model.py#L20) with implementations for libraries like HuggingFace (hf) and vLLM, LRAGE provides [pyserini_retriever](https://github.com/hoorangyee/LRAGE/blob/main/lrage/retrievers/pyserini_retriever.py) (powered by [Pyserini](https://github.com/castorini/pyserini)) in [lrage/retrievers/](https://github.com/hoorangyee/LRAGE/tree/main/lrage/retrievers) and [rerankers_reranker](https://github.com/hoorangyee/LRAGE/blob/main/lrage/rerankers/rerankers_reranker.py) (powered by [rerankers](https://github.com/AnswerDotAI/rerankers)) in [lrage/rerankers/](https://github.com/hoorangyee/LRAGE/tree/main/lrage/rerankers). This structure allows users to easily implement and integrate other retrievers or rerankers, such as those from [LlamaIndex](https://github.com/run-llama/llama_index), by simply extending the abstract classes.  

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/hoorangyee/LRAGE.git
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
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B \
    --tasks abercrombie \
    --batch_size 8 \
    --device cuda \
    --retrieve_docs \
    --top_k 3 \
    --retriever pyserini \
    --retriever_args retriever_type=bm25,bm25_index_path=YOUR_INDEX_PATH \
    --rerank \
    --reranker rerankers \
    --reranker_args reranker_type=colbert \
    ```

## Indexing

**Note**: A simplified indexing feature will be provided in a future release. If you plan to use the pyserini retriever, please refer to [Pyserini's indexing documentation](https://github.com/castorini/pyserini/blob/master/docs/usage-index.md) for guidance in the meantime.  

## Roadmap

- [ ] Implement LLM-as-a-judge functionality
- [ ] Develop a GUI Demo for easier access and visualization
- [ ] Publish and share Pile-of-law chunks
- [ ] Publish and share Pile-of-law BM25 index
- [ ] Publish and share Pile-of-law embeddings
- [ ] Update pyserini_retriever to support Pyserini prebuilt index
- [ ] Implement a simplified indexing feature
- [ ] Document more detailed usage instructions
- [ ] Publish benchmark results obtained using LRAGE

## Citation

```
@Misc{lrage,
  title =        {LARGE: Legal Retrieval Augmented Generation Evaluation Tool},
  author =       {Minhu Park and Wonseok Hwang},
  howpublished = {\url{https://github.com/hoorangyee/LRAGE}},
  year =         {2024}
}   
```
