# From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs

> **Paper:** [arXiv:2505.16408](https://arxiv.org/abs/2505.16408)  
> **Authors:** M. Farid Adilazuarda, Chen Cecilia Liu, Iryna Gurevych, Alham Fikri Aji  
> **Code:** [github.com/faridlazuarda/from-surveys-to-narratives](https://github.com/faridlazuarda/from-surveys-to-narratives)

---

## 🧩 Overview
Large Language Models often default to **WEIRD-centric** (Western, Educated, Industrialized, Rich, Democratic) values.  
Prior work aligns models using **World Values Survey (WVS)** data — but surveys alone can **homogenize** cultures and **distort knowledge**.  

This repo reproduces the experiments from our paper and provides:
- LoRA-based adapters for cultural value alignment  
- Evaluation on cultural tasks & **MMLU**  
- The **C-Dist** metric for *cultural distinctiveness*  
- Scripts to reproduce all paper figures  

---

## ⚙️ Setup

```bash
git clone https://github.com/faridlazuarda/from-surveys-to-narratives.git
cd from-surveys-to-narratives

conda create -n valadapt python=3.10 -y
conda activate valadapt
pip install -r requirements.txt
````

(Optional) if using Hugging Face models:

```bash
huggingface-cli login
```

---

## 📂 Repo Structure

```
from-surveys-to-narratives/
├─ data/        # datasets (WVS, Wikipedia, NormAd, evaluation)
├─ script/      # bash/python scripts for training, eval, plots
├─ src/         # core modules (training, metrics, plotting)
├─ figs/        # generated figures
└─ requirements.txt
```

---

## 🧠 Data Sources

| Source        | Type                | Purpose                    |
| ------------- | ------------------- | -------------------------- |
| **WVS**       | Survey Q/A          | Cultural values (baseline) |
| **Wikipedia** | Encyclopedic text   | Knowledge context          |
| **NormAd**    | Normative scenarios | Behavioral grounding       |

Place or symlink your processed data under:

```
data/wvs/
data/wiki/
data/normad/
data/tasks/
data/mmlu/
```

---

## 🚀 Quickstart

### 1️⃣ Train a culture-specific adapter

```bash
bash script/train_wvs_normad.sh \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --culture kor \
  --train data/wvs/kor.jsonl data/normad/kor.jsonl \
  --rank 64 --lr 2e-4 --epochs 6 \
  --out outputs/llama31_it/wvs+normad/kor
```

### 2️⃣ Evaluate

```bash
# Cultural tasks
bash script/eval_cultural_tasks.sh \
  --ckpt outputs/llama31_it/wvs+normad/kor \
  --test_root data/tasks \
  --out outputs/llama31_it/wvs+normad/kor/eval_cultural.json

# MMLU (knowledge)
bash script/eval_mmlu.sh \
  --ckpt outputs/llama31_it/wvs+normad/kor \
  --mmlu_root data/mmlu \
  --out outputs/llama31_it/wvs+normad/kor/eval_mmlu.json
```

### 3️⃣ Compute Cultural Distinctiveness (**C-Dist**)

```bash
bash script/compute_cdist.sh \
  --matrices outputs/**/eval_cultural.json \
  --out figs/cdist_summary.json
```

---

## 📏 Metric: C-Dist

We measure how well each culture-specific adapter performs **best on its own culture**.

For *n* cultures, let **M ∈ ℝⁿˣⁿ** be the F1-score matrix  
(rows = adapter culture, columns = test culture):

nᵢ = Mᵢᵢ / maxⱼ Mⱼᵢ  
D = (1 / n) ∑ᵢ nᵢ

- **D ≈ 1.0 →** distinct cultural behaviors  
- **Low D →** homogenization / cultural interference

---

## 📊 Reproduce Figures

```bash
# Cross-culture heatmaps
bash script/plot_heatmaps.sh --inputs outputs/**/eval_cultural.json --out figs/heatmaps

# Summary bars (C-Dist, MMLU, invalid%)
bash script/plot_bars.sh --cdist figs/cdist_summary.json --out figs/bars
```

---

## 🧮 Models & Cultures

* **Models:** Llama-3.1-8B (base/IT), Gemma-2-9B-IT, Qwen-2.5-7B-IT
* **Cultures:** ara, ben, zho, eng, deu, ell, kor, por, spa, tur

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{adilazuarda2025from,
  title   = {From Surveys to Narratives: Rethinking Cultural Value Adaptation in LLMs},
  author  = {Adilazuarda, M. Farid and Liu, Chen Cecilia and Gurevych, Iryna and Aji, Alham Fikri},
  year    = {2025},
  journal = {arXiv preprint arXiv:2505.16408}
}
```
