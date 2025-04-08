import argparse
import sys
import traceback
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from src.train import ModelTrainer
from src.data_loader import DataLoader
from src.utils import ResultsVisualizer

def main():
    """
    Main entry point for radar signal classification
    """
    # Argument parsing with improved error handling
    parser = argparse.ArgumentParser(description='Radar Signal Classification')
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,  
        help='Path to HDF5 dataset'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50, 
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='Training batch size (default: 64)'
    )
    parser.add_argument(
        '--cv_splits', 
        type=int, 
        default=5, 
        help='Number of cross-validation splits (default: 5)'
    )
    
    try:
        # Parse arguments
        args = parser.parse_args()
        
        logging.info("Starting Radar Signal Classification")
        logging.info("-" * 40)
        logging.info(f"Dataset: {args.dataset}")
        logging.info(f"Epochs: {args.epochs}")
        logging.info(f"Batch Size: {args.batch_size}")
        logging.info(f"Cross-Validation Splits: {args.cv_splits}")
        logging.info("-" * 40)
        
        # Start timing
        start_time = time.time()
        
        # Initialize data loader
        logging.info("Loading dataset...")
        data_loader = DataLoader(args.dataset)
        
        # Verify data loading
        X, y = data_loader.load_data()
        logging.info(f"Dataset loaded. Total samples: {len(X)}")
        logging.info(f"Signal shape: {X.shape}")
        logging.info(f"Number of classes: {len(set(y))}")
        
        # Create trainer
        logging.info("Initializing model trainer...")
        trainer = ModelTrainer(data_loader)
        
        # Perform training and evaluation
        logging.info("Starting training process...")
        results = trainer.train_and_evaluate(
            epochs=args.epochs,
            batch_size=args.batch_size,
            cv_splits=args.cv_splits
        )
        
        # Visualize results
        logging.info("Generating visualizations...")
        visualizer = ResultsVisualizer()
        
        # Plot training history
        logging.info("Plotting training history...")
        visualizer.plot_training_history(results.get('history', {}))
        
        # Plot confusion matrix
        logging.info("Plotting confusion matrix...")
        visualizer.plot_confusion_matrix(
            results.get('confusion_matrix', None), 
            results.get('class_names', [])
        )
        
        # Save classification metrics
        logging.info("Saving classification metrics...")
        visualizer.save_classification_metrics(results.get('metrics', {}))
        
        # End timing
        end_time = time.time()
        
        logging.info("\nTraining completed successfully!")
        logging.info(f"Total Execution Time: {end_time - start_time:.2f} seconds")
        logging.info(f"Mean Accuracy: {results.get('mean_accuracy', 0):.4f} ± {results.get('std_accuracy', 0):.4f}")
    
    except Exception as e:
        logging.error("\n!! ERROR OCCURRED !!")
        logging.error(f"Error Type: {type(e).__name__}")
        logging.error(f"Error Message: {str(e)}")
        logging.error("\nDetailed Traceback:")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()