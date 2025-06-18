# Urban-CI Literature Review

This repository contains the implementation of the LLM-enabled literature review system for the paper:

**"Reimagining Urban Science: Scaling Causal Inference with Large Language Models"**  
[arXiv:2504.12345](https://arxiv.org/abs/2504.12345)

## Pipeline Components

The main pipeline (`pipeline.py`) consists of the following steps:

1. **Paper Filtering**: Identifies causal research papers from a corpus of urban science literature.
2. **Feature Extraction**: Extracts key features from each paper, including research topic, geographic focus, data modality, experiment type, and methodological approach.
3. **Feature Refinement**: Groups and categorizes similar features for clearer analysis.
4. **Sankey Diagram Data Generation**: Prepares data for visualizing the relationships between extracted features.

## Project Structure

```
.
├── pipeline.py              # Main processing pipeline
├── src/                     # Source code directory
│   ├── llm.py               # LLM interaction module
│   ├── filter.py            # Paper filtering logic
│   ├── data_processing.py   # Feature extraction and processing
│   └── sankey.py            # Sankey diagram data preparation
└── data/                    # Data directory
    ├── Cities/              # PDF papers
    └── papers/              # Metadata and processed data
```

## Getting Started

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

1. **Download data** from [Google Drive](https://drive.google.com/file/d/1062YcgCtQBFD7NrYS2BHexHBm1hpkfi8/view?usp=sharing) and place it in the `data/` directory. You can also run:

```bash
bash download_data.sh
```

3. **Add your OpenAI API key** to a `.env` file:

```bash
OPENAI_API_KEY=your_api_key
```

4. **Run the pipeline**:

```bash
python pipeline.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{xia2025reimagining,
  title     = {Reimagining Urban Science: Scaling Causal Inference with Large Language Models},
  author    = {Xia, Yutong and Qu, Ao and Zheng, Yunhan and Tang, Yihong and Zhuang, Dingyi and Liang, Yuxuan and Wang, Shenhao and Wu, Cathy and Sun, Lijun and Zimmermann, Roger and Zhao, Jinhua},
  journal   = {arXiv preprint arXiv:2504.12345},
  year      = {2025}
}
```
