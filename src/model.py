import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout
)


def build_model(vocab_size, sequence_length):

    model = Sequential([

        Input(shape=(sequence_length,)),

        Embedding(
            input_dim=vocab_size,
            output_dim=64
        ),

        LSTM(
            256,
            return_sequences=True
        ),

        Dropout(0.3),

        LSTM(256),

        Dropout(0.3),

        Dense(128, activation="relu"),

        Dropout(0.3),

        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model