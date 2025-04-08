import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

class RadarSignalClassifier:
    def __init__(self, input_shape, num_classes):
        """
        Initialize Radar Signal Classifier
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of signal classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build 1D CNN model for signal classification
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = keras.Sequential([
            # First 1D Convolutional Layer
            keras.layers.Conv1D(
                filters=64, 
                kernel_size=3, 
                activation='relu', 
                input_shape=self.input_shape
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            
            # Second 1D Convolutional Layer
            keras.layers.Conv1D(
                filters=128, 
                kernel_size=3, 
                activation='relu'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling1D(pool_size=2),
            
            # Third 1D Convolutional Layer
            keras.layers.Conv1D(
                filters=256, 
                kernel_size=3, 
                activation='relu'
            ),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling1D(),
            
            # Fully Connected Layers
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            
            # Output Layer
            keras.layers.Dense(
                self.num_classes, 
                activation='softmax'
            )
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    class TqdmCallback(keras.callbacks.Callback):
        """
        Custom callback to create a progress bar for Keras training
        """
        def __init__(self, epochs):
            super().__init__()
            self.epochs = epochs
            self.pbar = None
        
        def on_train_begin(self, logs=None):
            self.pbar = tqdm(
                total=self.epochs, 
                desc='Training Progress', 
                position=0, 
                leave=True
            )
        
        def on_epoch_end(self, epoch, logs=None):
            # Update progress bar with current epoch's metrics
            description = f"Epoch {epoch+1}/{self.epochs} "
            description += f"Loss: {logs.get('loss', 0):.4f} "
            description += f"Accuracy: {logs.get('accuracy', 0):.4f}"
            
            self.pbar.set_description(description)
            self.pbar.update(1)
        
        def on_train_end(self, logs=None):
            if self.pbar:
                self.pbar.close()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=1):
        """
        Train the model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training labels
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity mode
        
        Returns:
            keras.callbacks.History: Training history
        """
        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True
        )
        
        # Model checkpoint to save best model
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            'results/models/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
        
        # Custom TQDM progress callback
        tqdm_callback = self.TqdmCallback(epochs)
        
        # Combine callbacks
        callbacks = [
            early_stopping, 
            model_checkpoint, 
            tqdm_callback
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test labels
        
        Returns:
            tuple: (test loss, test accuracy)
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (numpy.ndarray): Input features
        
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self.model.predict(X)