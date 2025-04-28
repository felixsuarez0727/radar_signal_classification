import os
import time
import numpy as np
import logging
from tqdm import tqdm  # Librería para barras de progreso
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)

from src.model import ImprovedRadarSignalClassifier as RadarSignalClassifier

class ModelTrainer:
    def __init__(self, data_loader):
        """
        Initialize ModelTrainer
        
        Args:
            data_loader (DataLoader): Data loader object
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.data_loader = data_loader
        self.results_dir = 'results'
        
        # Create results directories
        os.makedirs(os.path.join(self.results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'logs'), exist_ok=True)
    
    def train_and_evaluate(self, epochs=50, batch_size=64, cv_splits=5):
        """
        Perform cross-validation and model training
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            cv_splits (int): Number of cross-validation splits
        
        Returns:
            dict: Cross-validation results
        """
        # Global start time
        global_start_time = time.time()
        
        # Load data
        logging.info("Preparing data for cross-validation")
        X, y = self.data_loader.load_data()
        
        # Validate data
        if X is None or y is None:
            logging.error("Failed to load data for training")
            raise ValueError("Data loading failed")
        
        logging.info(f"Total samples: {len(X)}")
        logging.info(f"Sample shape: {X.shape}")
        logging.info(f"Number of classes: {len(np.unique(y))}")
        
        # Prepare cross-validation
        skf = StratifiedKFold(
            n_splits=cv_splits, 
            shuffle=True, 
            random_state=42
        )
        
        # Results storage
        cv_results = {
            'accuracies': [],
            'losses': [],
            'detailed_reports': [],
            'histories': [],
            'confusion_matrices': []
        }
        
        # Progress bar for cross-validation
        cv_progress = tqdm(
            total=cv_splits, 
            desc="Cross-Validation Progress", 
            position=0
        )
        
        # Cross-validation loop
        for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
            logging.info(f"\nProcessing Fold {fold}")
            
            # Split data
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            logging.info(f"Fold {fold} - Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Create model
            classifier = RadarSignalClassifier(
                input_shape=X_train.shape[1:], 
                num_classes=len(np.unique(y))
            )
            
            # Train model
            logging.info(f"Training Fold {fold}")
            history = classifier.train(
                X_train, y_train, 
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = classifier.evaluate(X_val, y_val)
            logging.info(f"Fold {fold} - Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
            
            # Predictions for detailed analysis
            y_pred_proba = classifier.predict(X_val)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # Generate classification report
            report = classification_report(
                y_val, 
                y_pred, 
                target_names=self.data_loader.get_class_names(),
                output_dict=True
            )
            
            # Compute confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            
            # Store results
            cv_results['accuracies'].append(test_accuracy)
            cv_results['losses'].append(test_loss)
            cv_results['detailed_reports'].append(report)
            cv_results['histories'].append(history.history)
            cv_results['confusion_matrices'].append(cm)
            
            # Update cross-validation progress
            cv_progress.update(1)
        
        # Close progress bar
        cv_progress.close()
        
        # Global completion time
        global_end_time = time.time()
        total_time = global_end_time - global_start_time
        
        # Aggregate results
        final_results = {
            'accuracies': cv_results['accuracies'],
            'mean_accuracy': np.mean(cv_results['accuracies']),
            'std_accuracy': np.std(cv_results['accuracies']),
            'total_training_time': total_time,
            'history': cv_results['histories'][-1],  # Use last fold's history
            'metrics': cv_results['detailed_reports'][-1],
            'confusion_matrix': cv_results['confusion_matrices'][-1],
            'class_names': self.data_loader.get_class_names()
        }
        
        # Log summary
        logging.info("\n--- Training Summary ---")
        logging.info(f"Total Training Time: {total_time:.2f} seconds")
        logging.info(f"Mean Accuracy: {final_results['mean_accuracy']:.4f} ± {final_results['std_accuracy']:.4f}")
        
        return final_results