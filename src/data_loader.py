import h5py
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, dataset_path, data_percentage=1.0, stratified=True, samples_per_class=25):
        """
        Initialize DataLoader
        
        Args:
            dataset_path (str): Path to HDF5 dataset
            data_percentage (float): Percentage of data to use (0.0 to 1.0)
            stratified (bool): Whether to use stratified sampling
            samples_per_class (int): Number of samples per class to select
        """
        self.dataset_path = dataset_path
        self.data_percentage = data_percentage
        self.stratified = stratified
        self.samples_per_class = samples_per_class
        self.X = None
        self.y = None
        self.y_encoded = None
        self.class_names = None
        self.label_encoder = LabelEncoder()
        
    def _group_signals(self, signals, labels):
        """
        Group AM signals while keeping other signals separate
        
        Args:
            signals (numpy.ndarray): Signal data
            labels (numpy.ndarray): Original labels
            
        Returns:
            tuple: (signals, new_labels)
        """
        # Create a copy of labels to modify
        new_labels = labels.copy()
        
        # Group AM signals (AM-DSB, AM-SSB, ASK)
        am_indices = np.where(np.isin(labels, ['AM-DSB_AM radio', 'AM-SSB_AM radio', 'ASK_short-range']))[0]
        new_labels[am_indices] = 'AM_combined'
        
        # Keep other signals as they are
        # BPSK remains as BPSK_SATCOM
        # FMCW remains as FMCW_Radar Altimeter
        # All PULSED types remain separate
        
        return signals, new_labels
    
    def load_data(self):
        """
        Load and preprocess data from HDF5 file
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logging.info(f"Attempting to load dataset from {self.dataset_path}")
        
        try:
            with h5py.File(self.dataset_path, 'r') as f:
                # Group keys by signal type
                all_keys = list(f.keys())
                logging.info(f"Total individual keys in the HDF5 file: {len(all_keys)}")
                # Extract signal type from each key (assuming the key is a tuple or string with info)
                class_map = {}
                for k in all_keys:
                    # If the key is a tuple, use the first two elements as type and subtype
                    if isinstance(k, tuple) or (isinstance(k, str) and k.startswith("('")):
                        # Convert tuple string to actual tuple if necessary
                        if isinstance(k, str):
                            import ast
                            k_tuple = ast.literal_eval(k)
                        else:
                            k_tuple = k
                        signal_type = f"{k_tuple[0]}_{k_tuple[1]}"
                    else:
                        signal_type = k
                    if signal_type not in class_map:
                        class_map[signal_type] = []
                    class_map[signal_type].append(k)
                # Now, for each class, select samples and group AM
                X_selected = []
                y_selected = []
                for signal_type, key_list in class_map.items():
                    # Group AM
                    if any(am in signal_type for am in ['AM-DSB_AM radio', 'AM-SSB_AM radio', 'ASK_short-range']):
                        label = 'AM_combined'
                    else:
                        label = signal_type
                    # Select up to samples_per_class samples per class
                    selected_keys = np.random.choice(key_list, size=min(len(key_list), self.samples_per_class), replace=False)
                    for k in selected_keys:
                        signal = f[k][()]
                        if np.isscalar(signal):
                            signal = np.array([signal])
                        X_selected.append(signal)
                        y_selected.append(label)
                X = np.array(X_selected)
                y = np.array(y_selected)
                # Show final classes and number of samples per class
                unique, counts = np.unique(y, return_counts=True)
                logging.info(f"Final classes: {dict(zip(unique, counts))}")
                if len(X.shape) == 2:
                    X = X.reshape(X.shape[0], X.shape[1], 1)
                self.y_encoded = self.label_encoder.fit_transform(y)
                self.class_names = self.label_encoder.classes_
                X_train, X_test, y_train, y_test = train_test_split(
                    X, self.y_encoded,
                    test_size=0.2,
                    random_state=42,
                    stratify=self.y_encoded
                )
                self.X = X
                self.y = y
                logging.info("Dataset loaded successfully and grouped by class")
                logging.info(f"X shape: {X.shape}")
                logging.info(f"y shape: {y.shape}")
                logging.info(f"Unique classes: {len(np.unique(y))}")
                return X_train, X_test, y_train, y_test
                
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_class_names(self):
        """
        Get list of class names
        
        Returns:
            list: Class names
        """
        return self.class_names