import sys
import os

# other imports and training code
from src.data_loader import get_data_generators
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint 

train_dir = os.path.join('data', 'train')
test_dir = os.path.join('data', 'test')
img_size = (128, 128)
batch_size = 32

data_dir = "data"  # or the full path to your data folder if it's somewhere else
img_size = (224, 224)
batch_size = 32

train_gen, val_gen = get_data_generators(data_dir, img_size, batch_size)
train_gen, val_gen = get_data_generators(data_dir, img_size, batch_size)
num_classes = train_gen.num_classes

model = build_model((*img_size, 3), num_classes)
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint]
)

model.save('models/garbage_classifier.h5')