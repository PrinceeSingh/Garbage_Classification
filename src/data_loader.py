from keras.utils import image_dataset_from_directory
import os

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    # Define paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Create datasets using the modern Keras API
    train_dataset = image_dataset_from_directory(
        train_dir,
        validation_split=None,
        subset=None,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_dataset = image_dataset_from_directory(
        val_dir,
        validation_split=None,
        subset=None,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    return train_dataset, val_dataset 
