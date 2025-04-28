import h5py
import numpy as np
import logging
import traceback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataLoader:
    def __init__(self, file_path):
        """
        Initialize DataLoader with HDF5 file
        
        Args:
            file_path (str): Path to the HDF5 dataset
        """
        self.file_path = file_path
        self.X = None
        self.y = None
        self.label_encoder = LabelEncoder()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def load_data(self, max_samples=None):
        """
        Load data from HDF5 file
        
        Args:
            max_samples (int, optional): Limit number of samples to load
        
        Returns:
            tuple: (features, labels)
        """
        logging.info(f"Attempting to load dataset from {self.file_path}")
        
        try:
            with h5py.File(self.file_path, 'r') as hf:
                # Get all keys
                keys = list(hf.keys())
                total_signals = len(keys)
                
                # Limit samples if specified
                if max_samples is not None:
                    keys = keys[:max_samples]
                
                logging.info(f"Total signals in dataset: {total_signals}")
                logging.info(f"Loading {len(keys)} signals")
                
                # Prepare lists to store data
                signals = []
                labels = []
                
                # Process signals
                for idx, key in enumerate(keys):
                    # Parse the key
                    key_info = eval(key)
                    
                    # Create a comprehensive label combining modulation and domain
                    # Format: Modulation_Domain
                    label = f"{key_info[0]}_{key_info[1]}"
                    
                    # Get the signal
                    signal = hf[key][:]
                    
                    signals.append(signal)
                    labels.append(label)
                
                # Convert to numpy arrays
                self.X = np.array(signals)
                self.y = np.array(labels)
            
            # Reshape data for 1D CNN (samples, length, channels)
            if len(self.X.shape) == 2:
                self.X = self.X.reshape(self.X.shape[0], self.X.shape[1], 1)
            
            # Encode labels
            self.y_encoded = self.label_encoder.fit_transform(self.y)
            
            # Log dataset details
            logging.info("Dataset loaded successfully")
            logging.info(f"X shape: {self.X.shape}")
            logging.info(f"y shape: {self.y.shape}")
            logging.info(f"Unique classes: {len(np.unique(self.y_encoded))}")
            
            # Print class distribution
            unique_classes, class_counts = np.unique(self.y_encoded, return_counts=True)
            logging.info("Class Distribution:")
            for cls, count in zip(self.label_encoder.classes_, class_counts):
                logging.info(f"  - {cls}: {count} samples")
            
            return self.X, self.y_encoded
        
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess data:
        1. Normalize features
        2. Split into train/test sets
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load data if not already loaded
        if self.X is None or self.y is None:
            self.load_data()
        
        # Normalize data
        X_normalized = self._normalize_data(self.X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, 
            self.y_encoded, 
            test_size=test_size, 
            stratify=self.y_encoded, 
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def _normalize_data(self, X):
        """
        Normalize input data
        
        Args:
            X (numpy.ndarray): Input data
        
        Returns:
            numpy.ndarray: Normalized data
        """
        # Normalize each channel separately
        X_normalized = np.zeros_like(X, dtype=np.float32)
        for i in range(X.shape[1]):  # Iterate through channels
            scaler = StandardScaler()
            X_normalized[:, i, :] = scaler.fit_transform(X[:, i, :])
        
        return X_normalized
    
    def get_class_names(self):
        """
        Get decoded class names
        
        Returns:
            list: Class names
        """
        return list(self.label_encoder.classes_)