# mmE5: Enhancing Multimodal Multilingual Embeddings with Superior Synthetic Data

This repository provides the source code, models, and datasets for our paper **mmE5: Improving Multimodal Multilingual Embeddings via High-quality Synthetic Data**. In our work, we explore innovative methods for integrating high-quality synthetic data to boost the performance and robustness of multimodal multilingual embeddings across diverse tasks.

<img width="1432" alt="mmE5 Overview" src="figures/teaser_mme5.png">

[![Paper](https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv)](https://github.com/haon-chen/mmE5)
[![Code](https://img.shields.io/badge/-Code-green?style=flat&logo=github)](https://github.com/haon-chen/mmE5)
[![Dataset](https://img.shields.io/badge/-Dataset-red?style=flat)](https://github.com/haon-chen/mmE5)
[![Models](https://img.shields.io/badge/-Models-red?style=flat)](https://github.com/haon-chen/mmE5)
---

## Latest Updates
- **2025-02:** We release the paper, code, datasets and models of mmE5.

## Model Overview
Multimodal embedding models have gained significant attention for their ability to map data from different modalities, such as text and images, into a unified representation space. However, the limited labeled multimodal data often hinders embedding performance. Recent approaches have leveraged data synthesis to address this problem, yet the quality of synthetic data remains a critical bottleneck. In this work, we identify three criteria for high-quality synthetic multimodal data. First, broad scope ensures that the generated data covers diverse tasks and modalities, making it applicable to various downstream scenarios. Second, robust cross-modal alignment makes different modalities semantically consistent. Third, high fidelity ensures that the synthetic data maintains realistic details to enhance its reliability. Guided by these principles, we synthesize datasets that: (1) cover a wide range of tasks, modality combinations, and languages, (2) are generated via a deep thinking process within a single pass of a multimodal large language model, and (3) incorporate real-world images with accurate and relevant texts, ensuring fidelity through self-evaluation and refinement. Leveraging these high-quality synthetic and labeled datasets, we train a multimodal multilingual E5 model mmE5.  Extensive experiments demonstrate that mmE5 achieves state-of-the-art performance on the MMEB Benchmark and superior multilingual performance on the XTD benchmark.

<img width="1432" alt="mmE5 Architecture Diagram" src="figures/model_architecture.jpg">

## Datasets
Our experiments leverage a comprehensive dataset that combines real-world examples with synthetic data, covering a wide range of tasks and languages. We also provide the labeled training set of MMEB benchmark that includes mined hard negatives.
- [Synthetic Dataset](https://github.com/haon-chen/mmE5)
- [MMEB with Hard Negative](https://github.com/haon-chen/mmE5)

## Experimental Results
mmE5 consistently outperforms existing methods in multimodal and multilingual tasks. Our experiments highlight significant gains in both accuracy and robustness.

<img alt="Experimental Results" src="figures/experimental_results.png">

## Quick Start
- Preparation
Download images from [Synthetic Dataset](https://github.com/haon-chen/mmE5), [MMEB with Hard Negative](https://github.com/haon-chen/mmE5), [MMEB-eval](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval), and [XTD](https://huggingface.co/datasets/Haon-Chen/XTD-10).

We have provided example scripts in the `scripts/` directory to help you get started with training and evaluation.
- Train
```
./train/train.sh
```
- Test MMEB
```
./eval/eval_full.sh
```
- Test XTD
```
./eval/eval_full_multi.sh
```

## Acknowledgement
- We have adapted code from [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec), a comprehensive implementation of transforming MLLMs to embedding models.


## Citation
```

```
