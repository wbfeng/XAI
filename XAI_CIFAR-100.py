import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100
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

# 1. Read CIFAR-100 dataset
def load_cifar100_data():
    """Load and preprocess CIFAR-100 dataset"""
    print("Loading CIFAR-100 dataset...")
    
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Convert labels to categorical one-hot encoding
    num_classes = 100
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# 2. Build CNN neural network model
def build_cnn_model(input_shape=(32, 32, 3), num_classes=100):
    """Build a CNN model for CIFAR-100 classification"""
    
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
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
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
    target_names = [f'Class_{i}' for i in range(100)]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=target_names, 
                              zero_division=0))
    
    # Confusion matrix (sample of 20 classes for visualization)
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot confusion matrix for first 20 classes
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm[:20, :20], annot=False, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix (First 20 Classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
    
    return y_pred_classes

# 4. Explainable AI with SHAP
def explain_model_with_shap(model, X_train_sample, X_test_sample, class_names=None):
    """Use SHAP to explain model predictions"""
    
    print("Creating SHAP explainer...")
    
    # Create a simple explainer (using background data for sampling)
    # For demonstration, we'll use a small subset of test data
    
    # Use only a few samples for SHAP explanation due to computational cost
    sample_indices = np.random.choice(len(X_test_sample), min(10, len(X_test_sample)), replace=False)
    X_explain = X_test_sample[sample_indices]
    
    # Create explainer (using TreeExplainer for neural networks)
    try:
        # For faster execution, we'll use a simpler approach with sample explanations
        print("SHAP explanation for first 5 samples...")
        
        # Get predictions for samples
        predictions = model.predict(X_explain)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Create SHAP explainer for the first sample
        explainer = shap.Explainer(model, X_train_sample[:100])  # Use subset for background
        
        # Explain first few samples
        shap_values = explainer(X_explain[:3])
        
        # Plot SHAP values
        print("Plotting SHAP explanations...")
        shap.plots.beeswarm(shap_values)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
        return shap_values
        
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Using alternative explanation method...")
        
        # Alternative: Show sample predictions with class probabilities
        for i in range(min(3, len(X_explain))):
            pred = model.predict(X_explain[i:i+1])
            predicted_class = np.argmax(pred)
            confidence = np.max(pred)
            
            print(f"Sample {i+1}: Predicted Class {predicted_class} (Confidence: {confidence:.4f})")
            
            # Show top 5 probabilities
            top_5_indices = np.argsort(pred[0])[::-1][:5]
            print("Top 5 predictions:")
            for j, idx in enumerate(top_5_indices):
                print(f"  {j+1}. Class {idx}: {pred[0][idx]:.4f}")
            print()

# Main execution
def main():
    # Step 1: Load data
    X_train, y_train, X_test, y_test = load_cifar100_data()
    
    # Step 2: Build model
    print("\nBuilding CNN model...")
    model = build_cnn_model()
    
    # Display model architecture
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Step 3: Train model (with validation split)
    print("\nTraining model...")
    
    # Use a smaller subset for quick demonstration
    X_train_subset = X_train[:10000]  # Using 10k samples for faster training
    y_train_subset = y_train[:10000]
    
    history = model.fit(
        X_train_subset, y_train_subset,
        batch_size=32,
        epochs=5,  # Reduced for demonstration
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Step 4: Evaluate model
    print("\nEvaluating model...")
    y_pred_classes = evaluate_model(model, X_test[:1000], y_test[:1000])  # Using subset for faster evaluation
    
    # Step 5: Explain model (SHAP)
    print("\nExplaining model with SHAP...")
    explain_model_with_shap(model, X_train_subset, X_test[:100])
    
    # Save the model
    model.save('cifar100_cnn_model.h5')
    print("\nModel saved as 'cifar100_cnn_model.h5'")

# Alternative explanation method for better performance
def simple_explanation(model, X_test_sample):
    """Simple explanation method for demonstration"""
    
    # Show some sample predictions with visual explanations
    plt.figure(figsize=(15, 10))
    
    # Get a few test samples
    sample_indices = np.random.choice(len(X_test_sample), 6, replace=False)
    
    for i, idx in enumerate(sample_indices):
        # Get prediction
        sample = X_test_sample[idx:idx+1]
        prediction = model.predict(sample)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Display original image
        plt.subplot(2, 3, i+1)
        plt.imshow(sample[0])
        plt.title(f'Class: {predicted_class}\nConfidence: {confidence:.3f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Sample Predictions with Confidence', y=1.02)
    plt.show()

# Run the main function
if __name__ == "__main__":
    try:
        main()
        
        # Additional simple explanation
        print("\n" + "="*50)
        print("Simple Visual Explanation")
        print("="*50)
        
        # Load data again for simple explanation
        X_train, y_train, X_test, y_test = load_cifar100_data()
        simple_explanation(build_cnn_model(), X_test)
        
    except Exception as e:
        print(f"Error in execution: {e}")
        print("Please ensure you have installed all required packages:")
        print("pip install tensorflow shap matplotlib seaborn scikit-learn")
