import os
import urllib.request
import json
import shutil
import numpy as np
import essentia
import essentia.standard as es
from mutagen.id3 import ID3, TCON, TXXX, COMM, TIT3, ID3NoHeaderError
from mutagen import File
from multiprocessing import Pool, cpu_count

# Silence Essentia
essentia.log.warningActive = False
essentia.log.infoActive = False

# --- CONFIGURATION ---
LIBRARY_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music to sort'
DESTINATION_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music'
MODELS_DIR = './models'
# Use about half your available cores to stay safe/fast
NUM_WORKERS = 10

MODEL_URLS = {
    "genre_pb": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_json": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "embeddings_pb": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
}

# Global dictionary to hold models for each worker process
worker_models = {}


def init_worker():
    """Initializes the AI models once for each worker process."""
    with open(os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.json"), 'r') as f:
        metadata = json.load(f)

    worker_models['labels'] = metadata['classes']
    worker_models['loader'] = es.MonoLoader(sampleRate=16000)
    worker_models['extractor'] = es.MusicExtractor(lowlevelStats=['mean'], rhythmStats=['mean'], tonalStats=['mean'])
    worker_models['embeddings'] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
    worker_models['genre'] = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0"
    )


def analyze_track(file_path):
    """The function each worker runs for a single track."""
    filename = os.path.basename(file_path)
    try:
        # Load audio (Resampled to 16kHz)
        worker_models['loader'].configure(filename=file_path, sampleRate=16000)
        audio = worker_models['loader']()

        # Extract features
        features, _ = worker_models['extractor'](file_path)
        danceability = features['rhythm.danceability']
        energy = features['lowlevel.average_loudness']

        # AI Genre Inference
        activations = worker_models['embeddings'](audio)
        predictions = worker_models['genre'](activations)
        mean_predictions = np.mean(predictions, axis=0)

        top_index = np.argmax(mean_predictions)
        top_genre = worker_models['labels'][top_index]
        # Strip the "MainGenre---" prefix and just keep the specific sub-genre
        if "---" in top_genre:
            top_genre = top_genre.split("---")[-1]
        confidence = mean_predictions[top_index] * 100

        current_file_path = file_path
        # If it's a WAV, convert to AIFF for better Rekordbox compatibility
        if file_path.lower().endswith('.wav'):
            aiff_path = file_path.rsplit('.', 1)[0] + '.aif'
            # afconvert is native to macOS. Use BEI24 for AIFF data format.
            try:
                os.system(f'afconvert -f AIFF -d BEI24 "{file_path}" "{aiff_path}"')
                if os.path.exists(aiff_path):
                    os.remove(file_path)
                    current_file_path = aiff_path
                    filename = os.path.basename(aiff_path)
            except Exception:
                pass # Fallback to original file if conversion fails

        # Tag the file using mutagen.File for multi-format support (MP3, WAV, AIFF)
        audio_file = File(current_file_path)
        if audio_file is None:
            return {"original_path": current_file_path, "filename": filename, "status": "error", "error": "Not a supported audio file"}

        if audio_file.tags is None:
            audio_file.add_tags()

        # Clean up old/conflicting tags for Rekordbox
        # We delete TIT1/TIT3 and any COMM frames to ensure a clean overwrite
        for key in list(audio_file.tags.keys()):
            if key.startswith("TIT1") or key.startswith("TIT3") or key.startswith("COMM"):
                del audio_file.tags[key]

        # Update tags using ID3 frames (Mutagen maps these correctly for AIFF/WAV)
        audio_file.tags["TCON"] = TCON(encoding=3, text=[top_genre])
        audio_file.tags["TIT3"] = TIT3(encoding=3, text=[f"E: {energy:.1f}"])
        # desc='' is essential for Rekordbox to show the comment
        comment_text = f"D: {danceability:.2f} | Conf: {confidence:.1f}%"
        audio_file.tags["COMM::eng"] = COMM(encoding=3, lang='eng', desc='', text=[comment_text])
        
        # Save with ID3v2.3 for maximum Rekordbox compatibility
        audio_file.save(v2_version=3)

        return {"original_path": current_file_path, "filename": filename, "genre": top_genre, "status": "success"}
    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}


def ensure_models_exist():
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    for url in MODEL_URLS.values():
        filepath = os.path.join(MODELS_DIR, url.split('/')[-1])
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)


def sanitize_folder_name(name):
    clean = name.split('---')[-1]
    return "".join([c for c in clean if c.isalnum() or c in (' ', '-', '_')]).strip()


if __name__ == "__main__":
    ensure_models_exist()

    print("📋 Scanning library...")
    all_tracks = []
    for root, _, files in os.walk(LIBRARY_PATH):
        for f in files:
            if f.lower().endswith(('.mp3', '.wav', '.aiff', '.aif')):
                all_tracks.append(os.path.join(root, f))

    total = len(all_tracks)
    print(f"🚀 Found {total} tracks. Parallelizing across {NUM_WORKERS} cores...")

    # Start the worker pool
    move_queue = []
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        # map_async allows us to track progress
        results = []
        for i, res in enumerate(pool.imap_unordered(analyze_track, all_tracks), 1):
            results.append(res)
            if res['status'] == "success":
                print(f"[{i / total * 100:.1f}%] ({i}/{total}) ✅ {res['genre']} | {res['filename']}")
                move_queue.append(res)
            else:
                print(f"[{i / total * 100:.1f}%] ({i}/{total}) ❌ Failed: {res['filename']} - {res['error']}")

    if move_queue:
        choice = input(f"\nAnalysis complete. Move {len(move_queue)} files? (y/n): ")
        if choice.lower() == 'y':
            for item in move_queue:
                target_dir = os.path.join(DESTINATION_PATH, sanitize_folder_name(item['genre']))
                if not os.path.exists(target_dir): os.makedirs(target_dir)
                shutil.move(item['original_path'], os.path.join(target_dir, item['filename']))
            print("\n✅ Library organized.")