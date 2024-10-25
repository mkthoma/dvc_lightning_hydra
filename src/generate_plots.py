import re
import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log(log_file_path):
    metrics = {}
    
    # Read the log file
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Extract training metrics
    train_metrics_match = re.search(r"Training metrics:\s*(\{.*?\})", log_content, re.DOTALL)
    if train_metrics_match:
        train_metrics_str = train_metrics_match.group(1)
        print(f"Raw train metrics string: {train_metrics_str}")  # Debug print
        
        try:
            # Remove tensor() wrappers and convert to a valid JSON format
            cleaned_metrics_str = re.sub(r'tensor\(([\d.]+)\)', r'\1', train_metrics_str)
            cleaned_metrics_str = cleaned_metrics_str.replace("'", '"')  # Replace single quotes with double quotes
            # Ensure all numbers are properly formatted
            cleaned_metrics_str = re.sub(r'(?<=\d)\.(?=\s*[,}])', '', cleaned_metrics_str)  # Remove trailing dots
            train_metrics = json.loads(cleaned_metrics_str)
            
            metrics['val_loss'] = float(train_metrics['val/loss'])
            metrics['val_acc'] = float(train_metrics['val/acc'])
            metrics['val_acc_best'] = float(train_metrics['val/acc_best'])
            metrics['train_loss'] = float(train_metrics['train/loss'])
            metrics['train_acc'] = float(train_metrics['train/acc'])
        except json.JSONDecodeError as e:
            print(f"Error parsing train metrics: {e}")
            print(f"Problematic string: {cleaned_metrics_str}")
    else:
        print("No training metrics found in the log file.")
    
    # Extract test metrics
    test_metrics_match = re.search(r"Test metrics:\s*(\[.*?\])", log_content, re.DOTALL)
    if test_metrics_match:
        test_metrics_str = test_metrics_match.group(1)
        try:
            # Convert to a valid JSON format
            cleaned_test_metrics_str = test_metrics_str.replace("'", '"')  # Replace single quotes with double quotes
            test_metrics = json.loads(cleaned_test_metrics_str)
            
            metrics['test_loss'] = test_metrics[0]['test/loss']
            metrics['test_acc'] = test_metrics[0]['test/acc']
        except json.JSONDecodeError as e:
            print(f"Error parsing test metrics: {e}")
            print(f"Problematic string: {cleaned_test_metrics_str}")
    else:
        print("No test metrics found in the log file.")
    
    return metrics

def plot_metrics(metrics, log_folder):
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(log_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare data for plotting
    categories = ['Accuracy', 'Loss']
    train_values = [metrics['train_acc'], metrics['train_loss']]
    val_values = [metrics['val_acc'], metrics['val_loss']]
    test_values = [metrics['test_acc'], metrics['test_loss']]

    # Set up the bar chart
    x = range(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar([i - width for i in x], train_values, width, label='Train', color='blue')
    ax.bar(x, val_values, width, label='Validation', color='green')
    ax.bar([i + width for i in x], test_values, width, label='Test', color='red')

    # Customize the chart
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels on the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(ax.containers[0])
    add_labels(ax.containers[1])
    add_labels(ax.containers[2])

    # Save the plot
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'metrics_comparison.png')
    plt.savefig(plot_path)
    plt.close(fig)  # Close the figure to free up memory

    print(f"Metrics plot saved to {plot_path}")

# def generate_confusion_matrix(y_true, y_pred, class_names, log_folder):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plots_dir = os.path.join(log_folder, 'plots')
#     plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
#     plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parse log file and generate plots.')
    parser.add_argument('--log_file_path', type=str, required=True,
                        help='Path to the log file to be parsed')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the provided log file path
    log_file_path = args.log_file_path
    
    # Check if the file exists
    if not os.path.exists(log_file_path):
        print(f"Error: The file {log_file_path} does not exist.")
        return
    
    # Parse the log file
    parsed_metrics = parse_log(log_file_path)

    # Print the parsed metrics
    for key, value in parsed_metrics.items():
        print(f"{key}: {value}")

    # Create plots folder under the main logs directory
    plots_folder = os.path.join('logs', 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    print(f"Plots folder: {plots_folder}")

    # Generate and save plots
    plot_metrics(parsed_metrics, plots_folder)

    # Generate and save confusion matrix


if __name__ == "__main__":
    main()
