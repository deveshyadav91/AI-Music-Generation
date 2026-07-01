from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from model import build_model

# -----------------------------------
# Paths
# -----------------------------------
OUTPUT_PATH = Path("outputs")
OUTPUT_PATH.mkdir(exist_ok=True)

plt.savefig(OUTPUT_PATH / "loss.png")
plt.savefig(OUTPUT_PATH / "accuracy.png")

DATA_PATH = Path("data")
CHECKPOINT_PATH = Path("checkpoints")

CHECKPOINT_PATH.mkdir(exist_ok=True)

# -----------------------------------
# Load Data
# -----------------------------------

X = np.load(DATA_PATH / "network_input.npy")
y = np.load(DATA_PATH / "network_output.npy")

with open(DATA_PATH / "note_to_int.pkl", "rb") as f:
    note_to_int = pickle.load(f)

vocab_size = len(note_to_int)
sequence_length = X.shape[1]

print("Dataset Shape :", X.shape)

# -----------------------------------
# Train Validation Split
# -----------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)

print("Training Samples :", len(X_train))
print("Validation Samples :", len(X_val))

# -----------------------------------
# Build Model
# -----------------------------------

model = build_model(vocab_size, sequence_length)
model.summary()

# -----------------------------------
# Callbacks
# -----------------------------------

checkpoint = ModelCheckpoint(
    filepath=CHECKPOINT_PATH / "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# -----------------------------------
# Train
# -----------------------------------

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=10,

    batch_size=128,

    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ]
)

# -----------------------------------
# Save Final Model
# -----------------------------------

model.save(CHECKPOINT_PATH / "final_model.keras")

print("Training Complete.")

# -----------------------------------
# Plot Loss
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(OUTPUT_PATH / "loss.png")
plt.close()

# -----------------------------------
# Plot Accuracy
# -----------------------------------

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("outputs/accuracy.png")
plt.close()