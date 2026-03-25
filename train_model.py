import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
BASE_DATASET_DIR = os.path.join('dataset', 'images')
CLASSES = ['caesar_salad', 'club_sandwich', 'french_fries', 'hamburger', 'pizza']
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
MODEL_SAVE_PATH = os.path.join('model', 'food_model.h5')

# Create a filtered dataset directory with only our 5 classes
FILTERED_DIR = os.path.join('dataset', 'filtered')
print("Setting up filtered dataset with 5 classes...")
for cls in CLASSES:
    src = os.path.join(BASE_DATASET_DIR, cls)
    dst = os.path.join(FILTERED_DIR, cls)
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
        # Copy first 200 images for faster training
        images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:200]
        import shutil
        for img in images:
            shutil.copy2(os.path.join(src, img), os.path.join(dst, img))
        print(f"  Copied {len(images)} images for class: {cls}")
    else:
        count = len(os.listdir(dst))
        print(f"  Already exists: {cls} ({count} images)")

print(f"\nDataset ready at: {FILTERED_DIR}")
print(f"Classes (alphabetical order): {CLASSES}\n")

# ─────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    FILTERED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = datagen.flow_from_directory(
    FILTERED_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print(f"Class indices: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}\n")

# ─────────────────────────────────────────────
# Build Model — MobileNetV2 Transfer Learning
# ─────────────────────────────────────────────
print("Loading MobileNetV2 base model (pretrained on ImageNet)...")
base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze all base layers
print(f"Base model loaded. Total layers: {len(base_model.layers)} (all frozen)\n")

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(CLASSES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model Architecture Summary:")
print(f"  Input: (224 × 224 × 3)")
print(f"  MobileNetV2 Base (frozen)")
print(f"  GlobalAveragePooling2D")
print(f"  Dense(128, relu)")
print(f"  Dense({len(CLASSES)}, softmax)  →  {CLASSES}")
print()

# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
print(f"Starting training for {EPOCHS} epochs...")
print("=" * 60)

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

# ─────────────────────────────────────────────
# Save Model
# ─────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")

final_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
print(f"\n📊 Final Results:")
print(f"   Training Accuracy  : {final_acc:.2f}%")
print(f"   Validation Accuracy: {final_val_acc:.2f}%")
print(f"\n🚀 Training complete! Run 'python app.py' to start the web app.")
