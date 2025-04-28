import argparse
import sys
import traceback
import time
import logging

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Radar Signal Classification')
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--data_percentage', type=float, default=0.2, help='Percentage of data to use (0.0 to 1.0)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--cv_splits', type=int, default=5, help='Number of cross-validation splits')
    parser.add_argument('--samples_per_class', type=int, default=25, help='Number of samples to select per class')
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    logging.info("Starting Radar Signal Classification")
    logging.info("----------------------------------------")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Cross-Validation Splits: {args.cv_splits}")
    logging.info(f"Data Percentage: {args.data_percentage*100:.1f}%")
    logging.info(f"Samples per Class: {args.samples_per_class}")
    logging.info(f"Stratified Sampling: True")
    logging.info("----------------------------------------")
    
    try:
        # Initialize data loader
        data_loader = DataLoader(
            dataset_path=args.dataset,
            data_percentage=args.data_percentage,
            stratified=True,
            samples_per_class=args.samples_per_class
        )
        
        # Load data and get train/test split
        logging.info("Loading dataset...")
        X_train, X_test, y_train, y_test = data_loader.load_data()
        
        # Initialize model trainer
        logging.info("Initializing model trainer...")
        trainer = ModelTrainer(data_loader)
        
        # Train and evaluate model
        logging.info("Starting training process...")
        results = trainer.train_and_evaluate(
            epochs=args.epochs,
            batch_size=args.batch_size,
            cv_splits=args.cv_splits
        )
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        visualizer = ResultsVisualizer()
        visualizer.plot_training_history(results['history'])
        visualizer.plot_confusion_matrix(results['confusion_matrix'], results['class_names'])
        visualizer.save_classification_metrics(results['metrics'])

        visualizer.plot_signal_examples(data_loader.X, data_loader.y_encoded, data_loader.class_names)
        visualizer.plot_class_distribution(data_loader.y_encoded, data_loader.class_names)
        
        # Print final results
        logging.info("\nTraining completed successfully!")
        logging.info(f"Total Execution Time: {time.time() - start_time:.2f} seconds")
        logging.info(f"Mean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        
    except Exception as e:
        logging.error("\n!! ERROR OCCURRED !!")
        logging.error(f"Error Type: {type(e).__name__}")
        logging.error(f"Error Message: {str(e)}")
        logging.error("\nDetailed Traceback:")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()