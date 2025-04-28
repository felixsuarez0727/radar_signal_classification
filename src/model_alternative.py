import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class SimpleRadarSignalClassifier:
    def __init__(self, input_shape, num_classes):
        """
        Simple Radar Signal Classifier using RandomForest
        
        Args:
            input_shape (tuple): Shape of input data (not used directly)
            num_classes (int): Number of signal classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,        # Reduced to avoid overfitting
            min_samples_split=10,  # Increased for better generalization
            min_samples_leaf=5,    # Increased for better generalization
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=1):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Not used
            batch_size (int): Not used
            verbose (int): Verbosity mode
        
        Returns:
            object: Training history-like object
        """
        # Reshape data for sklearn
        X_train_reshaped = self._reshape_for_sklearn(X_train)
        X_val_reshaped = self._reshape_for_sklearn(X_val)
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        
        if verbose:
            print(f"Training model on {X_train_scaled.shape[0]} samples...")
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Calculate validation metrics
        X_val_scaled = self.scaler.transform(X_val_reshaped)
        val_pred = self.model.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_pred)
        
        if verbose:
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, val_pred))
        
        # Create history-like object
        self.history = {
            'history': {
                'accuracy': [train_acc],
                'val_accuracy': [val_acc]
            }
        }
        
        return self.history
    
    def _reshape_for_sklearn(self, X):
        """
        Reshape 3D input data to 2D for sklearn models
        
        Args:
            X (numpy.ndarray): Input data (samples, length, channels)
        
        Returns:
            numpy.ndarray: Reshaped data (samples, length*channels)
        """
        sample_count = X.shape[0]
        flattened_shape = np.prod(X.shape[1:])
        return X.reshape(sample_count, flattened_shape)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        
        Returns:
            tuple: (0 as placeholder for loss, test accuracy)
        """
        X_test_reshaped = self._reshape_for_sklearn(X_test)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Print detailed classification report
        print("\nTest Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Return tuple with 0 as placeholder for loss to match Keras interface
        return 0, accuracy
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        X_reshaped = self._reshape_for_sklearn(X)
        X_scaled = self.scaler.transform(X_reshaped)
        
        # Get probabilities instead of classes
        proba = self.model.predict_proba(X_scaled)
        return proba