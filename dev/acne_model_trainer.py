# importing the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Assign paths from environment variables
train_dir = os.getenv("TRAIN_DIR")
test_dir = os.getenv("TEST_DIR")
model_path  = os.getenv("MODEL_PATH")
sample_image  = os.getenv("SAMPLE_IMAGE")

# defining the four severity classes
class_labels = {
    0: "Clear",
    1: "Mild",
    2: "Moderate",
    3: "Severe"
}

# defining oversample function to handle class imbalance
def oversample_to_balance(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            class_counts[class_name] = len(images)

    max_count = max(class_counts.values())

    for class_name, count in class_counts.items():
        if count < max_count:
            class_path = os.path.join(directory, class_name)
            images = os.listdir(class_path)
            needed = max_count - count
            for i in range(needed):
                src = os.path.join(class_path, random.choice(images))
                dst = os.path.join(class_path, f"dup_{i}_{os.path.basename(src)}")
                shutil.copy(src, dst)
                

# Model version 1
def train_mobilenetv2_acne_model(train_dir, test_dir, img_size=(224, 224), batch_size=32, initial_epochs=10, fine_tune_epochs=20, save_path='acne_mobilenet_model.keras'):
    
    # 1. Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.3,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
    )
    # For test data, only normalization is applied (no augmentation).
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # 2. Load Pretrained Base
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False  # Freeze initially

    # 3. Build Model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 acne classes
    ])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 4. Phase 1: Train top layers
    print("Phase 1: Training top layers")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True)
    ]

    model.fit(train_generator, validation_data=test_generator, epochs=initial_epochs)

    # 5. Phase 2: Fine-tuning entire model
    print("Phase 2: Fine-tuning MobileNetV2")
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_generator, validation_data=test_generator, epochs=fine_tune_epochs)

    print(f"✅ Model saved to: {save_path}")

    # 6. Plot Training History
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title("Loss")
    plt.legend()
    plt.show()

    return model

# Model version 2
def train_mobilenetv2_acne_model_v2(train_dir, test_dir, img_size=(160, 160), batch_size=16, initial_epochs=10, fine_tune_epochs=20, save_path="mobilenetv2_acne_model_v2.keras"):
    # 1. Data Generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.3,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30.0,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # 2. Handle Class Imbalance
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights_dict = dict(enumerate(class_weights))

    # 3. Load Pretrained MobileNetV2
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False

    # 4. Build Model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # 6. Train Top Layers
    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=initial_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # 7. Fine-Tune Top Layers of MobileNetV2
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=5e-6), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=fine_tune_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # 8. Save the model
    model.save(save_path)
    print(f"✅ Model saved to: {save_path}")

    # 9. Plot Accuracy and Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, history

# Model version 3
def train_mobilenetv2_acne_model_improved(train_dir, test_dir, img_size=(160, 160), batch_size=16, initial_epochs=20, fine_tune_epochs=20, use_class_weight=True, save_path="mobilenetv2_acne_model_improved.keras"):
    # 1. Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.3,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30.0,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

    # 2. Compute class weights (optional)
    if use_class_weight:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weights_dict = dict(enumerate(class_weights))
    else:
        class_weights_dict = None

    # 3. Load Pretrained MobileNetV2
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False  # freeze initially

    # 4. Build model with reduced regularization
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.4),
        Dense(4, activation='softmax')  # 4 severity levels
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(save_path, save_best_only=True)
    ]

    print("Phase 1: Training top layers")
    model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=initial_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    # 6. Fine-tune top 60 layers of MobileNetV2
    print("Phase 2: Fine-tuning last 60 layers of MobileNetV2")
    for layer in base_model.layers[:-60]:
        layer.trainable = False
    for layer in base_model.layers[-60:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=fine_tune_epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    model.save(save_path)
    print(f"✅ Model saved to: {save_path}")

    # 7. Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, history

# Model Predict Function
def predict_acne_severity(model_path, img_path, img_size=(160, 160)):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = round(preds[0][pred_class] * 100, 2)
    pred_label = class_labels[pred_class]
    print(f"✅ Prediction: {pred_label} ({pred_class}) with confidence: {confidence}%")
    return pred_class, pred_label, confidence

# defining main
def main():
    oversample_to_balance(train_dir)
    model, history = train_mobilenetv2_acne_model_improved(train_dir, test_dir, save_path=model_path)
    predict_acne_severity(model_path=model_path, img_path=sample_image)

if __name__ == "__main__":
    main()
