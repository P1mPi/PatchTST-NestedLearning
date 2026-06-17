# Adaptive Multivariate Time Series Forecasting via Nested Learning

This repository contains the official implementation of the Bachelor's Thesis (TFG): **"Predicción adaptativa de series temporales multivariantes mediante Nested Learning"** (Universidad de Granada, 2026).

This project extends the state-of-the-art [PatchTST](https://github.com/yuqinie98/PatchTST) architecture by transforming it from a static model into an **adaptive Test-Time Adaptation (TTA) system** using the **Nested Learning** paradigm.

## 🚀 Key Contributions

Unlike traditional static forecasting models, this repository introduces a **Continuum Memory System (CMS)** that updates its weights during inference to combat *Concept Drift* and high volatility. 

Our main architectural contributions include:
- **Dual-Frequency Parametric Hierarchy:** A frozen pre-trained backbone (Slow Weights) for long-term structural patterns, coupled with a dynamic CMS module (Fast Weights) for short-term adaptation.
- **Multiple CMS Topologies:**
  - `Flatten NL`: Basic linear adaptation.
  - `CMS / CMS3`: Deep Multi-Layer Perceptrons with residual connections (`base_pred + cms_pred`).
  - `Mid-CMS`: Deep latent insertion inside the Transformer encoder layers.
- **Statistical Process Control (SPC) Trigger:** An asynchronous, intelligent trigger that only executes `loss.backward()` during inference when the error exceeds a dynamic statistical threshold ($\mu + \sigma$), preventing catastrophic forgetting caused by stochastic noise.

## 🛠️ Installation & Requirements

Clone the repository and set up the environment (we recommend using `conda`):

```bash
git clone https://github.com/P1mPi/PatchTST-NestedLearning.git
cd PatchTST-NestedLearning
conda create -n adaptive_patchtst python=3.10
conda activate adaptive_patchtst
pip install -r requirements.txt
```

## 📊 Datasets

We evaluate our model on widely used benchmarks: **ETT (ETTh1, ETTh2, ETTm1, ETTm2)**, **Weather**, and **ILI**.
You can download the datasets from the [original Autoformer repository](https://github.com/thuml/Autoformer) and place them in the `./data/` directory.

## 💻 How to Run: New Hyperparameters

We have extended the original `argparse` to support our Nested Learning framework. The new key arguments are:

- `--head_type`: Toplogy of the CMS module (`flatten`, `cms`, `cms3`).
- `--update_policy`: Trigger policy for Test-Time Adaptation (`always`, `5steps`, `spc`, `none`).
- `--cms_lr`: Learning rate exclusively assigned to the dynamic CMS optimizer (e.g., `0.0001`).
- `--use_mid_cms`: Set to `1` to inject the CMS into the latent encoder space.
- `--mid_position`: Layer index to inject the Mid-CMS (e.g., `0`, `1`, `2`).

### Example: Running Adaptive Inference with SPC Trigger

To train the model and run inference using a residual 3-layer CMS and the SPC trigger policy on the ETTh1 dataset:

```bash
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model PatchTST \
  --data ETTh1 \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 3 \
  --head_type cms3 \
  --update_policy spc \
  --cms_lr 0.0001 \
  --batch_size 128
```

## 📖 Acknowledgements

- **PatchTST:** This project is built upon the original [PatchTST implementation](https://github.com/yuqinie98/PatchTST).
- **University of Granada (UGR):** Special thanks to the HPC department (CPD Santa Lucía) and my tutors for their continuous support.
