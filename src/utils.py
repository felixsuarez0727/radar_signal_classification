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
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        
        # Save the plot
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
        
        # Save the plot
        plot_path = os.path.join(self.results_dir, 'plots', 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.close()
    
    def save_classification_metrics(self, metrics):
        """
        Save classification metrics to a JSON file
        
        Args:
            metrics (dict): Classification metrics
        """
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