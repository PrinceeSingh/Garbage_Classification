import os
from src.data_loader import get_data_generators
from src.model import build_model
from keras.callbacks import ModelCheckpoint

def main():
    # Configuration
    data_dir = "data"
    img_size = (224, 224)
    batch_size = 32
    epochs = 10
    
    # Get data generators
    train_gen, val_gen = get_data_generators(data_dir, img_size, batch_size)
    num_classes = len(train_gen.class_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_gen.class_names}")
    
    # Build model
    model = build_model((*img_size, 3), num_classes)
    
    # Setup checkpoint
    checkpoint = ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint]
    )
    
    # Save final model
    model.save('models/garbage_classifier.h5')
    print("Training completed!")

if __name__ == "__main__":
    main() 