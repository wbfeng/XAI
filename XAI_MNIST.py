import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shap
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 1. Download and read MNIST dataset
def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Reshape data for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to categorical one-hot encoding
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# 2. Build CNN neural network model (similar to YOLO architecture for classification)
def build_yolo_like_model(input_shape=(28, 28, 1), num_classes=10):
    """Build a CNN model similar to YOLO architecture for MNIST classification"""
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense layers (similar to YOLO's final layers)
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Alternative: Simpler model that's more suitable for MNIST
def build_simple_model(input_shape=(28, 28, 1), num_classes=10):
    """Build a simpler CNN model for MNIST classification"""
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# 3. Evaluate the model
def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model"""
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred_classes == y_true_classes)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    target_names = [str(i) for i in range(10)]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=target_names, 
                              zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    return accuracy

# 4. Use explainable tools to explain the model
def explain_model_with_shap(model, X_test_sample):
    """Explain model predictions using SHAP"""
    print("Creating SHAP explanations...")
    
    # For demonstration, use a small subset of data
    X_sample = X_test_sample[:10]  # Use only 10 samples for faster computation
    
    try:
        # Create explainer
        explainer = shap.Explainer(model, X_sample)
        
        # Calculate SHAP values
        shap_values = explainer(X_sample)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=[f'Pixel_{i}' for i in range(784)])
        plt.title('SHAP Summary Plot')
        plt.show()
        
        # Plot SHAP values for first few samples
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i in range(min(10, len(shap_values))):
            shap.summary_plot(shap_values[i], X_sample[i:i+1], 
                            feature_names=[f'Pixel_{j}' for j in range(784)], 
                            show=False, ax=axes[i])
            axes[i].set_title(f'Sample {i} Prediction')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Using simple visualization instead...")
        
        # Simple visualization of model predictions
        visualize_predictions(model, X_test_sample)

def visualize_predictions(model, X_test_sample):
    """Simple visualization of predictions"""
    print("Visualizing sample predictions...")
    
    # Get predictions for first 10 samples
    predictions = model.predict(X_test_sample[:10])
    predicted_classes = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 6))
    
    for i in range(10):
        plt.subplot(2, 5, i+1)
        # Display image
        plt.imshow(X_test_sample[i].reshape(28, 28), cmap='gray')
        
        # Show prediction and confidence
        confidence = np.max(predictions[i])
        plt.title(f'Pred: {predicted_classes[i]}\nConf: {confidence:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Predictions with Confidence', y=1.02)
    plt.show()

def analyze_model_architecture(model):
    """Analyze and display model architecture"""
    print("\nModel Architecture:")
    model.summary()
    
    # Show parameter count
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

# Main execution function
def main():
    """Main execution function"""
    print("="*60)
    print("MNIST Classification with CNN")
    print("="*60)
    
    # 1. Load MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # 2. Build model
    print("\nBuilding model...")
    model = build_simple_model()  # Using simpler model for MNIST
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Analyze architecture
    analyze_model_architecture(model)
    
    # 3. Train model (with validation split)
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    
    # 5. Explain model using SHAP
    print("\nExplaining model with SHAP...")
    explain_model_with_shap(model, X_test)
    
    # Save the model
    model.save('mnist_cnn_model.h5')
    print("\nModel saved as 'mnist_cnn_model.h5'")
    
    return model

# Additional helper function for detailed analysis
def detailed_analysis(model, X_test, y_test):
    """Perform detailed analysis of model performance"""
    print("\n" + "="*50)
    print("DETAILED MODEL ANALYSIS")
    print("="*50)
    
    # Get predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Accuracy per class
    from sklearn.metrics import accuracy_score
    
    print(f"Overall Accuracy: {accuracy_score(true_classes, predicted_classes):.4f}")
    
    # Per-class accuracy
    for i in range(10):
        class_mask = (true_classes == i)
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(true_classes[class_mask], 
                                          predicted_classes[class_mask])
            print(f"Class {i} Accuracy: {class_accuracy:.4f}")

# Run the main function
if __name__ == "__main__":
    try:
        # Run main execution
        model = main()
        
        # Perform detailed analysis
        X_train, y_train, X_test, y_test = load_mnist_data()
        detailed_analysis(model, X_test, y_test)
        
    except Exception as e:
        print(f"Error in execution: {e}")
        print("Please ensure you have installed all required packages:")
        print("pip install tensorflow shap matplotlib seaborn scikit-learn numpy")
