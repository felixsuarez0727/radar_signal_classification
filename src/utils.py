import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsVisualizer:
    """
    Utility class for visualizing and saving machine learning results
    """
    def __init__(self, results_dir='results'):
        """
        Initialize ResultsVisualizer
        
        Args:
            results_dir (str): Directory to save results
        """
        self.results_dir = results_dir
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    
    def plot_training_history(self, history):
        """
        Plot and save training history (loss and accuracy)
        
        Args:
            history (dict): Training history dictionary
        """
        # Check if history is empty or None
        if not history:
            print("No training history to plot")
            return
        # Adapt for scikit-learn: if only one value per fold, plot as scatter
        plt.figure(figsize=(8, 5))
        acc = history.get('accuracy', [])
        val_acc = history.get('val_accuracy', [])
        if len(acc) <= 1 and len(val_acc) <= 1:
            plt.scatter([1], acc, label='Train Accuracy', color='blue')
            plt.scatter([1], val_acc, label='Validation Accuracy', color='orange')
            plt.title('Model Accuracy (Single Value)')
            plt.ylabel('Accuracy')
            plt.xlabel('Fold')
            plt.legend()
        else:
            plt.plot(acc, label='Train Accuracy')
            plt.plot(val_acc, label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'training_history.png')
        plt.savefig(plot_path)
        plt.close()
    
    def plot_confusion_matrix(self, confusion_matrix, class_names):
        """
        Plot and save confusion matrix
        
        Args:
            confusion_matrix (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        # Print matrix and class names for debugging
        print("Confusion matrix:")
        print(confusion_matrix)
        print("Class names:")
        print(class_names)
        if confusion_matrix is None or class_names is None or len(class_names) == 0:
            print("Cannot plot confusion matrix: Missing data")
            return
        plt.figure(figsize=(15, 10))
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
    
    def save_classification_metrics(self, metrics):
        """
        Save classification metrics to a JSON file
        
        Args:
            metrics (dict): Classification metrics
        """
        # Check if metrics is None or empty
        if not metrics:
            print("No metrics to save")
            return
            
        # Convert numpy types to native Python types
        def convert_metrics(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save metrics
        metrics_path = os.path.join(self.results_dir, 'logs', 'classification_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4, default=convert_metrics)

    def plot_signal_examples(self, X, y, class_names):
        """
        Plot and save one example signal per class.
        Args:
            X (numpy.ndarray): Signal data (samples, length, 1)
            y (numpy.ndarray): Encoded labels
            class_names (list): List of class names
        """
        import random
        plt.figure(figsize=(16, 8))
        for idx, class_name in enumerate(class_names):

            indices = np.where(y == idx)[0]
            if len(indices) == 0:
                continue

            signal_idx = random.choice(indices)
            signal = X[signal_idx].squeeze()
            plt.subplot(1, len(class_names), idx + 1)
            plt.plot(signal)
            plt.title(class_name)
            plt.xlabel('Muestras')
            plt.ylabel('Amplitud')
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'signals_examples.png')
        plt.savefig(plot_path)
        plt.close()

    def plot_class_distribution(self, y, class_names):
        """
        Plot and save the class distribution as a bar chart.
        Args:
            y (numpy.ndarray): Encoded labels
            class_names (list): List of class names
        """
        import collections
        counts = collections.Counter(y)
        plt.figure(figsize=(10, 6))
        plt.bar([class_names[i] for i in counts.keys()], counts.values(), color='skyblue')
        plt.title('Distribución de Clases')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad de señales')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, 'plots', 'class_distribution.png')
        plt.savefig(plot_path)
        plt.close()