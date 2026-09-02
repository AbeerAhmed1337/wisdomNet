WisdomNet — Trustable Learning Implementation

A small implementation of WisdomNet, a framework for trustable learning — training a neural network to recognize its own uncertain predictions and abstain ("reject") rather than guess, deferring those cases to a human expert.

This is a learning exercise built on top of my analysis report on the WisdomNet paradigm, applied to the Breast Cancer Wisconsin dataset.

Idea

Instead of a standard classifier that always outputs a class, WisdomNet adds a third "reject" option:

Train a base model normally on the full dataset.
Find its mistakes — the samples it misclassifies — and group them under a new reject label.
Attach a conjugate neuron to the output layer, initialized at zero so it doesn't change the base model's behavior yet.
Fine-tune only that neuron on the reject set, teaching the model to recognize the kinds of inputs it tends to get wrong.

At inference time, the model outputs one of three classes: 0, 1, or 2 (reject → send to human).

What this code does
Loads the Breast Cancer Wisconsin dataset (sklearn.datasets.load_breast_cancer) and standardizes features.
Trains a base MLP (16 hidden units, ReLU → softmax) for binary classification.
Identifies misclassified training samples and relabels them as class 2.
Builds wisdom_model by extending the base model's output layer to 3 units, with the new neuron's weights/biases zero-initialized so the extended model starts out equivalent to the base model.
Fine-tunes the extended model on the reject set only.
Evaluates on the held-out test set and reports how many samples were referred to a human ("rejected").
Visualizes results with:
A confusion matrix (including the reject column).
A 2D PCA projection of test predictions, color-coded by class 0 / class 1 / rejected.
Requirements
numpy
tensorflow
scikit-learn
matplotlib
seaborn

Install with:

bash
pip install numpy tensorflow scikit-learn matplotlib seaborn
Usage
bash
python wisdomnet.py

This will print the test set size and number of rejected samples, then display the confusion matrix and PCA plots.

Notes / Limitations
This is a binary-classification demo (2 real classes + 1 reject class). The original WisdomNet report notes that scaling a single conjugate neuron to multi-class problems is harder and still an open area.
Performance depends heavily on the base model already being reasonably accurate — a weak base model increases the human-review burden significantly.
This implementation is for learning/experimentation, not a production-ready reject-option classifier.
Reference

Based on concepts explored in my own analysis report: "Analysis Report of Evaluation of WisdomNet for Error-free ML."
