## 🧠 Overview

The system:
- Preprocesses raw signature data (normalization, interpolation)
- Generates balanced positive and negative signature pairs
- Trains a Siamese LSTM network using contrastive loss
- Evaluates model performance on all person-wise combinations
- Visualizes training progress, distributions, and performance metrics


## 📈 Outputs
All results are stored under the training_outputs/ directory:

•	models/ – trained models for each test pair

•	processors/ – saved data preprocessors

•	thresholds/ – optimal cosine distance thresholds

•	test_results/ – ROC, loss, confusion matrix plots

•	all_combinations_results.pkl – summary of all training combinations


## 📉 Evaluation Metrics

•	Accuracy

•	Precision

•	Recall

•	F1-score

•	ROC AUC (via plotting)

•	Confusion matrix and distance distributions

## ⚙️ Configuration
Adjust parameters in config.py:

•	MODEL_CONFIG: model size, attention usage, etc.

•	TRAINING_CONFIG: batch size, learning rate, margin, etc.

## 🧪 Architecture Details

•	Siamese LSTM with:

o	Bidirectional LSTM layers

o	Optional attention mechanism

o	BatchNorm and dropout

o	Dense layers for embedding

•	Distance metric: Cosine distance

•	Loss: Contrastive Loss with margin


