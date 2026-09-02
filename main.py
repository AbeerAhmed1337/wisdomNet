import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# 1. Dataset Preparation
# Loads the standard binary classification dataset directly into memory
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Pre-train the Base Network
# Uses a multilayer perceptron with 16 hidden neurons[cite: 2]
inputs = Input(shape=(X_train_scaled.shape[1],))
hidden = Dense(16, activation='relu')(inputs)
base_out = Dense(2, activation='softmax', name='base_output')(hidden)

base_model = Model(inputs, base_out)
base_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
base_model.fit(X_train_scaled, y_train, epochs=30, verbose=0)

# 3. Identify Misclassified Samples
# Isolates base network errors and groups them under a new label (2)
predictions = np.argmax(base_model.predict(X_train_scaled, verbose=0), axis=1)
misclassified_idx = np.where(predictions != y_train)[0]
X_reject = X_train_scaled[misclassified_idx]
y_reject = np.full((len(misclassified_idx),), 2) 

# 4. Construct WisdomNet with Conjugate Neuron
# Appends a zero-initialized unit to the output layer
last_hidden_layer = base_model.layers[-2].output
wisdom_out = Dense(3, activation='softmax', name='wisdom_output')(last_hidden_layer)
wisdom_model = Model(inputs=base_model.input, outputs=wisdom_out)

old_weights, old_biases = base_model.layers[-1].get_weights()
new_weights = np.zeros((old_weights.shape[0], 3))
new_biases = np.zeros((3,))

new_weights[:, :2] = old_weights
new_biases[:2] = old_biases
wisdom_model.layers[-1].set_weights([new_weights, new_biases])

# 5. Fine-train exclusively on the reject set
# Retrains the new neuron strictly on past mistakes[cite: 1, 2]
wisdom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
wisdom_model.fit(X_reject, y_reject, epochs=30, verbose=0)

# 6. Evaluate
test_preds = np.argmax(wisdom_model.predict(X_test_scaled, verbose=0), axis=1)
reject_count = np.sum(test_preds == 2)
print(f"Total Test Samples: {len(y_test)}")
print(f"Rejected Samples (Referred to Human Expert): {reject_count}")


# --- 1. Confusion Matrix Visualization ---
# The confusion matrix includes a 'Reject' column for the WisdomNet predictions
cm = confusion_matrix(y_test, test_preds, labels=[0, 1, 2])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1', 'Rejected (2)'], 
            yticklabels=['Actual 0', 'Actual 1', ''])
plt.title('WisdomNet Confusion Matrix')
plt.xlabel('WisdomNet Prediction')
plt.ylabel('True Label')


pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

plt.subplot(1, 2, 2)
colors = ['#1f77b4', '#2ca02c', '#d62728'] 
labels = ['Class 0 (Malignant)', 'Class 1 (Benign)', 'Rejected / Uncertain']

for i in range(3):
    idx = (test_preds == i)
    plt.scatter(X_test_pca[idx, 0], X_test_pca[idx, 1], 
                c=colors[i], label=labels[i], alpha=0.7, edgecolors='k')

plt.title('2D PCA Projection of WisdomNet Predictions')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.show()