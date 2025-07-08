# JSON parser
import json
import copy

def get_JSON_files(song_path):
    with open(song_path, "r") as f:
        data = json.load(f)

    functional_chords = []
    for chord in data["chords"]:
        # Convert interval_durations keys from strings to integers if present
        interval_durations = (
            {int(k): v for k, v in chord.get("interval_durations", {}).items()}
            if "interval_durations" in chord and chord["interval_durations"] else {}
        )
        
        # Handle pitch_classes - keep as original format from file
        pitch_classes = chord["pitch_classes"]
        if isinstance(pitch_classes, str):
            # If it's a string representation of a set, eval it safely
            try:
                if pitch_classes.startswith('{') and pitch_classes.endswith('}'):
                    # Convert string set representation to actual set
                    pitch_classes_str = pitch_classes.strip('{}')
                    if pitch_classes_str:
                        pitch_classes = set(map(int, pitch_classes_str.split(', ')))
                    else:
                        pitch_classes = set()
                else:
                    pitch_classes = set()
            except:
                pitch_classes = set()
        elif isinstance(pitch_classes, list):
            pitch_classes = set(pitch_classes)
        
        # Handle notes - keep as list of strings or convert properly
        notes = chord["notes"]
        if isinstance(notes, list) and len(notes) > 0:
            # Keep as list, but ensure consistent type
            if isinstance(notes[0], str):
                notes = notes  # Keep as string list
            else:
                notes = [str(note) for note in notes]
        else:
            notes = []

        # Create new chord dict preserving ALL original fields
        functional_chord = copy.deepcopy(chord)  # Start with all original data
        
        # Update only the fields that need processing
        functional_chord.update({
            "interval_durations": interval_durations,
            "pitch_classes": pitch_classes,
            "notes": notes,
            # Extract functional harmony fields for convenience
            "roman_numeral": chord.get("functional_harmony", {}).get("roman_numeral", ""),
            "functional": chord.get("functional_harmony", {}).get("functional", ""),
        })
        
        functional_chords.append(functional_chord)

    return functional_chords

def get_structure_data(song_path):
    with open(song_path, "r") as f:
        data = json.load(f)
    return {
        "sr": data.get("sr"),
        "beats": data.get("beats"),
        "bound_times": data.get("bound_times"),
        "bound_frames": data.get("bound_frames"),
        "bound_segments": data.get("bound_segments")
    }