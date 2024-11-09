import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam
import os
import numpy as np

inputshape = 30 # 12 for pcp, 30 for mfcc
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
    24 float values from 12 input values
    """
    
    model = Sequential([
        InputLayer(shape=(inputshape,)),

        Dense(inputshape*2, activation='relu'),
        BatchNormalization(),
        
        Dense(inputshape*4, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(inputshape*4, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(24, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, epochs=100, batch_size=32):
    """
    Train the model with the provided data
    
    Parameters:
    X_train: numpy array of shape (n_samples, 12)
    y_train: numpy array of shape (n_samples, 24)
    """
    model = create_ffnn_model()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    model.save('mfccmodel_for_1000files.h5')
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    predictions = model.predict(X_test[:3])  
    print("\nSample Predictions vs Actual Values:")
    for i in range(3):
        print(f"\nSample {i+1}:")
        print("Prediction:", predictions[i].round(3))
        print("Actual:", y_test[i])


if __name__ == '__main__':
    data_folder = "extracted_mfcc_annotations"
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
    
    print("\nEvaluating model...")
    evaluate_model(model, X_test,y_test)