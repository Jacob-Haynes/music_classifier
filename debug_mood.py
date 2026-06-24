"""
Quick diagnostic — run on a single track to inspect raw mood model outputs.
Usage: python debug_mood.py "/path/to/track.mp3"
"""
import sys
import os
import numpy as np
import essentia
import essentia.standard as es

essentia.log.warningActive = False
essentia.log.infoActive = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

if len(sys.argv) < 2:
    print("Usage: python debug_mood.py \"/path/to/track.mp3\"")
    sys.exit(1)

file_path = sys.argv[1]
print(f"\nAnalysing: {os.path.basename(file_path)}\n")

# Load audio
loader = es.MonoLoader(sampleRate=16000)
loader.configure(filename=file_path, sampleRate=16000)
audio = loader()

# MusicExtractor for energy
extractor = es.MusicExtractor(lowlevelStats=['mean'], rhythmStats=['mean'])
features, _ = extractor(file_path)
print(f"rhythm.beats_loudness.mean : {features['rhythm.beats_loudness.mean']:.6f}")
print(f"lowlevel.average_loudness  : {features['lowlevel.average_loudness']:.6f}")
print()

# EffNet embeddings
embeddings = es.TensorflowPredictEffnetDiscogs(
    graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"),
    output="PartitionedCall:1"
)
activations = embeddings(audio)
print(f"Activations shape: {activations.shape}")
print()

# Mood models — try model/Softmax first, then PartitionedCall:0 as fallback
for mood in ['mood_party', 'mood_aggressive', 'mood_relaxed']:
    pb = os.path.join(MODELS_DIR, f"{mood}-discogs-effnet-1.pb")
    for output_name in ["model/Softmax", "PartitionedCall:0", "Softmax:0"]:
        try:
            model = es.TensorflowPredict2D(graphFilename=pb, output=output_name)
            preds = model(activations)
            mean = np.mean(preds, axis=0)
            print(f"{mood} [{output_name}]")
            print(f"  raw mean output : {mean}")
            print(f"  index[0]        : {mean[0]:.4f}")
            print(f"  index[1]        : {mean[1]:.4f}")
            print()
            break
        except Exception as e:
            print(f"{mood} [{output_name}] FAILED: {e}")
