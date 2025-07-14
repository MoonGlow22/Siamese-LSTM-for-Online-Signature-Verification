import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import os


def analyze_all_combinations_results(results_file='./training_outputs/all_combinations_results.pkl'):
    try:
        
        output_dir = "analysis_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)

        print(f"Loaded {len(all_results)} combination results.\n")

        # Group metrics by person
        person_metrics = defaultdict(lambda: defaultdict(list))

        for result in all_results:
            for person_name in result['test_person_names']:
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    person_metrics[person_name][metric].append(result.get(metric, 0))

        print("=" * 60)
        print("AVERAGE PERFORMANCE WHEN EACH PERSON IS IN TEST SET")
        print("=" * 60)

        person_stats = {}
        for person, metrics in person_metrics.items():
            stat = {}
            for metric, values in metrics.items():
                stat[metric] = {
                    'avg': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            stat['count'] = len(metrics['accuracy'])
            person_stats[person] = stat

            print(f"{person}:")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                m = stat[metric]
                print(f"  {metric.capitalize():<9}: {m['avg']:.4f} ± {m['std']:.4f}")
            print(f"  Test count: {stat['count']}\n")

        # General statistics
        all_metrics = {k: [r.get(k, 0) for r in all_results] for k in ['accuracy', 'precision', 'recall', 'f1_score', 'used_threshold']}

        print("=" * 60)
        print("OVERALL STATISTICS")
        print("=" * 60)

        for key, values in all_metrics.items():
            print(f"{key.upper()}:")
            print(f"  Mean     : {np.mean(values):.4f} ± {np.std(values):.4f}")
            print(f"  Min-Max  : {np.min(values):.4f} - {np.max(values):.4f}")
            print(f"  Median   : {np.median(values):.4f}\n")

        # Best and worst results
        best_f1_result = max(all_results, key=lambda x: x.get('f1_score', 0))
        worst_f1_result = min(all_results, key=lambda x: x.get('f1_score', 0))

        print("Best performance (F1-score):")
        print_result_summary(best_f1_result)

        print("Worst performance (F1-score):")
        print_result_summary(worst_f1_result)

        # Performance ranges
        print("=" * 60)
        print("PERFORMANCE DISTRIBUTION ANALYSIS")
        print("=" * 60)
        analyze_performance_ranges(all_metrics['accuracy'])

        # Best and worst person pairs
        print("\n" + "=" * 60)
        print("INTER-PERSON PERFORMANCE COMPARISON")
        print("=" * 60)
        sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
        print("Top combinations:")
        for i, r in enumerate(sorted_results[:5]):
            print(f"{i+1}. {' - '.join(r['test_person_names'])}: {r['accuracy']:.4f}")

        print("\nWorst combinations:")
        for i, r in enumerate(sorted_results[-5:]):
            print(f"{i+1}. {' - '.join(r['test_person_names'])}: {r['accuracy']:.4f}")

        # Difficulty analysis
        print("\n" + "=" * 60)
        print("PERSON-BASED DIFFICULTY LEVEL ANALYSIS")
        print("=" * 60)
        sorted_persons = sorted(person_stats.items(), key=lambda x: x[1]['accuracy']['avg'])

        print("Most difficult to recognize persons:")
        for i, (p, s) in enumerate(sorted_persons[:3]):
            print(f"{i+1}. {p}: {s['accuracy']['avg']:.4f} (±{s['accuracy']['std']:.4f})")

        print("\nEasiest to recognize persons:")
        for i, (p, s) in enumerate(sorted_persons[-3:]):
            print(f"{i+1}. {p}: {s['accuracy']['avg']:.4f} (±{s['accuracy']['std']:.4f})")

        # Plot
        create_analysis_plots(all_results, person_stats)

        # CSV export
        results_df = create_results_dataframe(all_results, person_stats)

        return {
            'all_results': all_results,
            'person_stats': person_stats,
            'results_df': results_df
        }

    except FileNotFoundError:
        print(f"Error: '{results_file}' file not found!")
    except Exception as e:
        print(f"Error occurred during analysis: {e}")


def print_result_summary(result):
    print(f"  Combination {result['combination_idx']}: {' - '.join(result['test_person_names'])}")
    print(f"  Accuracy : {result['accuracy']:.4f}")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall   : {result['recall']:.4f}")
    print(f"  F1-Score : {result['f1_score']:.4f}")
    print(f"  Threshold: {result['used_threshold']:.4f}\n")


def analyze_performance_ranges(accuracies):
    ranges = [
        (0.0, 0.5, "Very Low (0.0–0.5)"),
        (0.5, 0.7, "Low (0.5–0.7)"),
        (0.7, 0.8, "Medium (0.7–0.8)"),
        (0.8, 0.9, "Good (0.8–0.9)"),
        (0.9, 1.0, "Very Good (0.9–1.0)")
    ]
    for min_acc, max_acc, label in ranges:
        count = sum(min_acc <= a < max_acc for a in accuracies)
        print(f"{label:<25}: {count} combinations ({100 * count / len(accuracies):.1f}%)")


def create_analysis_plots(all_results, person_stats):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('All Combinations Analysis Results', fontsize=16)

    accuracies = [r['accuracy'] for r in all_results]
    thresholds = [r['used_threshold'] for r in all_results]

    # Histogram: Accuracy
    axes[0, 0].hist(accuracies, bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Accuracy Distribution')
    axes[0, 0].axvline(np.mean(accuracies), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # Histogram: Threshold
    axes[0, 1].hist(thresholds, bins=20, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Threshold Distribution')
    axes[0, 1].axvline(np.mean(thresholds), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # Person-based barplot
    persons = list(person_stats.keys())
    acc_means = [person_stats[p]['accuracy']['avg'] for p in persons]
    acc_stds = [person_stats[p]['accuracy']['std'] for p in persons]
    y_pos = np.arange(len(persons))
    axes[1, 0].barh(y_pos, acc_means, xerr=acc_stds, color='orange')
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(persons)
    axes[1, 0].set_title('Person-Based Performance (Accuracy)')

    # Scatter: Threshold vs Accuracy
    axes[1, 1].scatter(thresholds, accuracies, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel('Optimal Threshold')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Threshold vs Accuracy')

    plt.tight_layout()
    plt.savefig(os.path.join("analysis_outputs", "combination_analysis_plots.png"), dpi=300)
    plt.show()


def create_results_dataframe(all_results, person_stats):
    combo_df = pd.DataFrame([
        {
            'Combination_ID': r['combination_idx'],
            'Person_1': r['test_person_names'][0],
            'Person_2': r['test_person_names'][1],
            'Accuracy': r['accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1_Score': r['f1_score'],
            'Threshold': r['used_threshold']
        }
        for r in all_results
    ])
    combo_df.to_csv(os.path.join("analysis_outputs", 'combinations_results.csv'), index=False)

    person_df = pd.DataFrame([
        {
            'Person': p,
            'Accuracy_Avg': s['accuracy']['avg'],
            'Accuracy_Std': s['accuracy']['std'],
            'Min_Acc': s['accuracy']['min'],
            'Max_Acc': s['accuracy']['max'],
            'Test_Count': s['count']
        }
        for p, s in person_stats.items()
    ])
    person_df.to_csv(os.path.join("analysis_outputs", 'person_statistics.csv'), index=False)

    print("CSV files saved.")
    return {'combinations': combo_df, 'persons': person_df}


if __name__ == "__main__":
    results = analyze_all_combinations_results()
    if results:
        print("Analysis completed successfully.")
