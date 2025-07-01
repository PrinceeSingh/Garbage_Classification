from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    # Define paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Data augmentation and rescaling
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    return train_generator, val_generator