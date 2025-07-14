## 🧠 Overview

The system:
- Preprocesses raw signature data (normalization, interpolation)
- Generates balanced positive and negative signature pairs
- Trains a Siamese LSTM network using contrastive loss
- Evaluates model performance on all person-wise combinations
- Visualizes training progress, distributions, and performance metrics

## 📊 Dataset Format

The project expects a folder structure like this:

```
Signatures/
├── person1/
│ ├── sig01.txt
│ ├── sig02.txt
│ └── ...
├── person2/
│ ├── sig01.txt
│ └── ...
└── ...
```

Each `.txt` file should contain 3 columns (X, Y, Time).  



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

## 📊 Post-Training Analysis
After all person combinations have been evaluated, the script analyze_all_combinations_results.py allows you to analyze the overall performance of the model across different test sets.
Key Features:

•	Per-Person Performance Stats:

Computes average, standard deviation, min, and max for accuracy, precision, recall, and F1-score when each person is in the test set.

•	Combination-Level Summary:

Reports best/worst performing pairs and overall statistical distributions.

•	Difficulty Estimation:

Identifies the hardest and easiest persons to recognize based on average accuracy.

•	Performance Distributions:

Categorizes accuracy values into ranges (e.g., low, medium, high) for interpretability.

•	Graphs and Visualizations:

o	Accuracy distribution

o	Threshold distribution

o	Person-wise bar chart with error bars

o	Scatter plot (threshold vs. accuracy)

All graphs are saved under the analysis_outputs/ directory as combination_analysis_plots.png.

•	Export to CSV:

o	combinations_results.csv: Performance stats per person pair

o	person_statistics.csv: Aggregate stats per person

## 📄 Publication Context
This project was developed as part of a scientific research study focused on Contactless Biometric Verification from In-Air Signatures using a Deep Siamese Network architecture.

