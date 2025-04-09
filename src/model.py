import tensorflow as tf
import numpy as np
import keras
from keras import layers, regularizers, optimizers, callbacks
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, BatchNormalization, MaxPooling1D, Dropout, GlobalAveragePooling1D

class ImprovedRadarSignalClassifier:
    def __init__(self, input_shape, num_classes):
        """
        Improved Radar Signal Classifier with multiple optimizations
        
        Args:
            input_shape (tuple): Shape of input data
            num_classes (int): Number of signal classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_advanced_model()
    
    def _build_advanced_model(self):
        """
        Advanced 1D CNN model with multiple improvements
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = Sequential([
            # Input Layer with Regularization
            Input(shape=self.input_shape),
            
            # First Convolutional Block with Advanced Regularization
            Conv1D(
                filters=64, 
                kernel_size=3, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                padding='same'
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Second Convolutional Block
            Conv1D(
                filters=128, 
                kernel_size=3, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                padding='same'
            ),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # Third Convolutional Block
            Conv1D(
                filters=256, 
                kernel_size=3, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                padding='same'
            ),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            
            # Fully Connected Layers with Advanced Regularization
            Dense(
                256, 
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
            ),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(
                128, 
                activation='relu',
                kernel_regularizer=regularizers.l1_l2(l1=0.0001, l2=0.0001)
            ),
            BatchNormalization(),
            Dropout(0.4),
            
            # Output Layer with Softmax
            Dense(
                self.num_classes, 
                activation='softmax'
            )
        ])
        
        # Advanced Compilation with Learning Rate Scheduling
        initial_learning_rate = 0.001
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=1):
        """
        Train the model with advanced callbacks
        
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
        # Advanced Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            min_delta=0.001
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5,
            patience=5, 
            min_lr=1e-6
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'results/models/best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
        
        # Train with advanced callbacks
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=verbose,
            class_weight=self._compute_class_weights(y_train)
        )
        
        return history
    
    def _compute_class_weights(self, y_train):
        """
        Compute class weights to handle class imbalance
        
        Args:
            y_train (numpy.ndarray): Training labels
        
        Returns:
            dict: Computed class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        
        return dict(enumerate(class_weights))
    
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