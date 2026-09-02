# 🧠 WisdomNet — Trustable Learning Implementation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/status-learning%20project-lightgrey)

A small implementation of **WisdomNet** — a framework for *trustable learning*, where a neural network learns to recognize its own uncertain predictions and abstain (**"reject"**) rather than guess, deferring those cases to a human expert.

This is a hands-on learning exercise built on top of my [analysis report](./Analysis_Report__26K-7608_.pdf) on the WisdomNet paradigm, applied to the **Breast Cancer Wisconsin** dataset.

---

## 💡 The Idea

Instead of a standard classifier that always outputs a class, WisdomNet adds a third **"reject"** option:

| Step | What happens |
|------|---------------|
| 1️⃣ Train | Train a base model normally on the full dataset |
| 2️⃣ Detect | Find the samples it misclassifies → group them under a new `reject` label |
| 3️⃣ Extend | Attach a **conjugate neuron** to the output layer, zero-initialized so it changes nothing at first |
| 4️⃣ Fine-tune | Train *only* that neuron on the reject set, teaching the model to spot the inputs it tends to get wrong |

At inference time, the model outputs one of three classes:

> `0` → Class A &nbsp; | &nbsp; `1` → Class B &nbsp; | &nbsp; `2` → **Reject → send to human**

---

## ⚙️ What This Code Does

- 📊 Loads the **Breast Cancer Wisconsin** dataset (`sklearn.datasets.load_breast_cancer`) and standardizes features
- 🧠 Trains a base MLP (16 hidden units, ReLU → softmax) for binary classification
- 🔍 Identifies misclassified training samples and relabels them as class `2`
- 🧩 Builds `wisdom_model` by extending the output layer to 3 units, with the new neuron's weights/biases zero-initialized so it starts out equivalent to the base model
- 🎯 Fine-tunes the extended model on the reject set only
- ✅ Evaluates on the held-out test set and reports how many samples were referred to a human
- 📈 Visualizes results with:
  - A **confusion matrix** (including the reject column)
  - A **2D PCA projection** of test predictions, color-coded by class 0 / class 1 / rejected

---

## 📦 Requirements

```
numpy
tensorflow
scikit-learn
matplotlib
seaborn
```

Install everything with:

```bash
pip install numpy tensorflow scikit-learn matplotlib seaborn
```

---

## 🚀 Usage

```bash
python wisdomnet.py
```

This prints the test set size and number of rejected samples, then displays the confusion matrix and PCA plots.

---

## ⚠️ Notes / Limitations

- Binary-classification demo (2 real classes + 1 reject class) — scaling a single conjugate neuron to multi-class problems is harder and still an open area
- Performance depends heavily on the base model already being reasonably accurate — a weak base model increases the human-review burden significantly
- Built for learning/experimentation, **not** a production-ready reject-option classifier

---

## 📚 Reference

Based on concepts explored in my own analysis report:
*"Analysis Report of Evaluation of WisdomNet for Error-free ML."*
