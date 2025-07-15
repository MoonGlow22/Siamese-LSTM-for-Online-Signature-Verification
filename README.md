# Siamese LSTM for Online Signature Verification

This project implements a signature verification system using a Siamese LSTM neural network architecture. It compares two signatures and determines whether they belong to the same person based on their dynamic properties extracted from the signature data.

## 🚀 Usage Instructions

### Step 1: Record Signatures
First, record signatures using the `record_signatures.py` file. The system records when your index and middle fingers are apart and stops recording when they are close together.

**Controls:**
* Press `z` to start recording
* Press `x` to stop and save the current signature
* Press `q` to quit the application

### Step 2: Clean Signature Data
Run `remove_lines.py` to remove `-100, -100` lines from the recorded signatures. These lines indicate when fingers were brought together to temporarily pause recording, but this feature is not used in model training.

### Step 3: Train the Model
Execute the `main.py` file in the `SiameseLSTM` folder to start model training.

### Step 4: Analyze Results (Optional)
You can analyze the training results using `results_analyzer.py` to evaluate model performance.

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

