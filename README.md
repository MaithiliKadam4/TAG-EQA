# TAG–EQA: Text–And–Graph for Event Question Answering via Structured Prompting Strategies

---

<div align="center">

## 🏆 **Accepted at \*SEM 2025** 🏆

[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.01391)

</div>

## 🚀 Installation

```bash
conda create -n new python=3.9
conda activate new

conda install pytorch torchvision torchaudio pytorch-cuda=11.7-c pytorch -c nvidia
conda install pandas scikit-learn

pip install pandas numpy tqdm scikit-learn matplotlib transformers accelerate torch datasets openai
pip install evaluate
```

## 📝 Flags in practice

text_or_graph accepts text or graph

prompt_type accepts zero or few or cot

If you need custom input or output paths, add those inside the script or extend argparse in the same file.

To prepare prompts run all_prompt_prep.py. The prompts will be saved in 'data/yes_no_prompt/prompt/'. 

```bash
python yes_no_t5_prompt.py --text_or_graph="text" --prompt_type="zero"
```

## 📊 Results

Results are saved in `data/full_run_1/` directory.


## 📚 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@inproceedings{kadam2025tag,
  title={TAG--EQA: Text--And--Graph for Event Question Answering via Structured Prompting Strategies},
  author={Kadam, Maithili Sanjay and Ferraro, Francis},
  booktitle={Proceedings of the 14th Joint Conference on Lexical and Computational Semantics (* SEM 2025)},
  pages={304--315},
  year={2025}
}
```

---