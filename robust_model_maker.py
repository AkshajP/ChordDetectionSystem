import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import os
import numpy as np

inputshape = 46 #12
def parse_file(content):
    x_data=[]
    y_data=[]
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    for line in lines:
        # Remove any whitespace and split by '],['
        arrays = line.replace(' ', '').strip('[]').split('],[')
        
        if len(arrays) == 2:
            try:
                x_array = [int(x) for x in arrays[0].split(',')]
                y_array = [float(x) for x in arrays[1].split(',')]
                if len(x_array) == 24 and len(y_array) == inputshape:
                    x_data.append(x_array)
                    y_data.append(y_array)
            except (ValueError, IndexError):
                print(f"Skipping malformed line: {line}")
                continue
    return x_data, y_data

def load_data_from_folder(folder):
    all_x = []
    all_y = []
    data_files = [file for root, dirs, files in os.walk(folder) for file in files if file.endswith('.csv')]
    for data_file in data_files:
        with open(os.path.join(folder,data_file), 'r') as file:
            content = file.read()
            x,y = parse_file(content)
            all_x += x
            all_y += y
    X = np.array(all_x, dtype=np.int32)
    Y = np.array(all_y, dtype= np.float32)
    return X,Y

def create_ffnn_model():
    """
    Creates a Feed-Forward Neural Network model for predicting 
    24 float values from 45 input values
    """
    
    model = Sequential([
        InputLayer(input_shape=(46,)),
        
        # First block with residual connection
        Dense(64, activation='elu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Second block with residual connection
        Dense(96, activation='elu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third block with increased width
        Dense(128, activation='elu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Fourth block focusing on feature abstraction
        Dense(64, activation='elu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Final classification layer
        Dense(24, activation='sigmoid')
    ])
    
    # Use a more sophisticated optimizer configuration
    optimizer = Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model

def train_model(X_train, y_train, epochs=100, batch_size=32):
    """
    Train the model with the provided data
    
    Parameters:
    X_train: numpy array of shape (n_samples, 45)
    y_train: numpy array of shape (n_samples, 24)
    """
    model = create_ffnn_model()
    
    # Learning rate scheduling
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
    
    # Early stopping with longer patience
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    
    # Model checkpointing
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     'best_model_45dim.h5',
    #     monitor='val_loss',
    #     save_best_only=True
    # )
    
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, 
                   lr_scheduler, 
                   # checkpoint
                   ],
        verbose=1
    )
    
    
    model.save('robust_extraction_1000files.h5')
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    
    Parameters:
    model: Trained Keras model
    X_test: Test features
    y_test: Test labels
    """
    # Get the metric values as a dictionary
    metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    
    # Print each metric
    print("\nModel Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Get predictions for sample data
    predictions = model.predict(X_test[:3], verbose=0)
    
    print("\nSample Predictions vs Actual Values:")
    for i in range(3):
        print(f"\nSample {i+1}:")
        print("Prediction:", predictions[i].round(3))
        print("Actual:", y_test[i])


if __name__ == '__main__':
    data_folder = "extracted_robust_45_annotations"
    #data_folder = 'extracted pcp annotations (12 bin)'
    # y, X for maintaining convention from this point onwards
    y, X = load_data_from_folder(data_folder) 

    print(f"\nLoaded {len(X)} samples from CSV files")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    batch_size = min(32, int(np.sqrt(len(X))))
    print(batch_size)
    print("\nTraining model...")
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)

    # Shuffle and split
    indices = np.random.permutation(len(X))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    model, history = train_model(X_train,y_train, epochs=100, batch_size=batch_size)
    #model = load_model("pcpmodel after training on 1000 files.h5")
    print("\nEvaluating model...")
    evaluate_model(model, X_test,y_test)