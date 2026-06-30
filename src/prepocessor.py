from pathlib import Path
from music21 import converter, instrument, note, chord
import pickle
from tqdm import tqdm

# ----------------------------
# Paths
# ----------------------------
DATASET_PATH = Path("data/midi")
OUTPUT_PATH = Path("data")

OUTPUT_PATH.mkdir(exist_ok=True)

notes = []

# ----------------------------
# Read every MIDI file
# ----------------------------
midi_files = list(DATASET_PATH.glob("*.mid"))

print(f"Found {len(midi_files)} MIDI files\n")

for file in tqdm(midi_files):

    try:
        midi = converter.parse(file)

        parts = instrument.partitionByInstrument(midi)

        if parts:
            elements = parts.parts[0].recurse()
        else:
            elements = midi.flat.notes

        for element in elements:

            # Single Note
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))

            # Chord
            elif isinstance(element, chord.Chord):
                notes.append(
                    ".".join(str(n) for n in element.normalOrder)
                )

    except Exception as e:
        print(f"Skipping {file}: {e}")

print(f"\nExtracted {len(notes)} notes/chords")

# ----------------------------
# Save notes
# ----------------------------
with open(OUTPUT_PATH / "notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("Saved notes.pkl")