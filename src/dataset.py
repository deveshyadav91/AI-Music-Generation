from pathlib import Path
import pickle
import numpy as np

# -----------------------
# Configuration
# -----------------------
SEQUENCE_LENGTH = 100

DATA_PATH = Path("data")

# -----------------------
# Load notes
# -----------------------
with open(DATA_PATH / "notes.pkl", "rb") as f:
    notes = pickle.load(f)

print(f"Total Notes: {len(notes)}")

# -----------------------
# Create Vocabulary
# -----------------------
pitchnames = sorted(set(notes))

note_to_int = {
    note: number
    for number, note in enumerate(pitchnames)
}

int_to_note = {
    number: note
    for number, note in enumerate(pitchnames)
}

print(f"Vocabulary Size: {len(pitchnames)}")

# -----------------------
# Create Sequences
# -----------------------
network_input = []
network_output = []

for i in range(len(notes) - SEQUENCE_LENGTH):

    sequence_in = notes[i:i + SEQUENCE_LENGTH]
    sequence_out = notes[i + SEQUENCE_LENGTH]

    network_input.append(
        [note_to_int[n] for n in sequence_in]
    )

    network_output.append(
        note_to_int[sequence_out]
    )

network_input = np.array(network_input, dtype=np.int32)
network_output = np.array(network_output, dtype=np.int32)

print("Input Shape :", network_input.shape)
print("Output Shape:", network_output.shape)

# -----------------------
# Save Everything
# -----------------------
with open(DATA_PATH / "note_to_int.pkl", "wb") as f:
    pickle.dump(note_to_int, f)

with open(DATA_PATH / "int_to_note.pkl", "wb") as f:
    pickle.dump(int_to_note, f)

np.save(DATA_PATH / "network_input.npy", network_input)
np.save(DATA_PATH / "network_output.npy", network_output)

print("\nSaved processed dataset.")