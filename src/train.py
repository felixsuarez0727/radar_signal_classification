import os
import time
import numpy as np
import logging
from tqdm import tqdm  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, 
    classification_report
)

# Try to import the TensorFlow model, but fall back to the alternative
try:
    from src.model import ImprovedRadarSignalClassifier as RadarSignalClassifier
    logging.info("Using TensorFlow-based model")
except ImportError:
    from src.model_alternative import SimpleRadarSignalClassifier as RadarSignalClassifier
    logging.info("Using scikit-learn-based model (TensorFlow not available)")

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
        
        # Use data that was already loaded by the DataLoader
        X, y = self.data_loader.X, self.data_loader.y_encoded
        
        # Validate data
        if X is None or y is None:
            logging.error("Data not available in DataLoader. Loading data...")
            X, y = self.data_loader.load_data()
        
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
            
            # Handle different history formats (TensorFlow vs. scikit-learn)
            if hasattr(history, 'history'):
                cv_results['histories'].append(history.history)
            else:
                cv_results['histories'].append(history.get('history', {}))
                
            cv_results['confusion_matrices'].append(cm)
            
            # Update progress bar
            cv_progress.update(1)
        
        # Close progress bar
        cv_progress.close()
        
        # Calculate overall metrics
        mean_accuracy = np.mean(cv_results['accuracies'])
        std_accuracy = np.std(cv_results['accuracies'])
        
        # Aggregate confusion matrices
        combined_cm = np.sum(cv_results['confusion_matrices'], axis=0)
        
        # Compute average history
        avg_history = {}
        try:
            for metric in cv_results['histories'][0].keys():
                # For each metric, compute the average across all folds
                avg_history[metric] = np.mean([h[metric] for h in cv_results['histories']], axis=0)
        except (KeyError, AttributeError, IndexError) as e:
            logging.warning(f"Could not compute average history: {e}")
            # Fallback to empty history
            avg_history = {'accuracy': [], 'val_accuracy': []}
        
        # Format results for return
        results = {
            'accuracies': cv_results['accuracies'],
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'confusion_matrix': combined_cm,
            'detailed_reports': cv_results['detailed_reports'],
            'history': avg_history,
            'class_names': self.data_loader.get_class_names(),
            'metrics': {
                'accuracy': f"{mean_accuracy:.4f} ± {std_accuracy:.4f}",
                'reports': cv_results['detailed_reports']
            }
        }
        
        # Calculate total execution time
        total_time = time.time() - global_start_time
        logging.info(f"Total CV execution time: {total_time:.2f} seconds")
        
        return results