import os
import json
import shutil
import subprocess
import numpy as np
import essentia
import essentia.standard as es
from mutagen.id3 import TCON, COMM, TIT3
from mutagen import File
from multiprocessing import Pool
from utils import get_vibe, convert_wav_to_aiff, download_with_progress

essentia.log.warningActive = False
essentia.log.infoActive = False

# --- CONFIGURATION ---
LIBRARY_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music to sort'
DESTINATION_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
NUM_WORKERS = 10

MODEL_URLS = {
    "genre_pb": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_json": "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "embeddings_pb": "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
}

worker_models = {}


def init_worker():
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
    filename = os.path.basename(file_path)
    wav_to_delete = None
    try:
        worker_models['loader'].configure(filename=file_path, sampleRate=16000)
        audio = worker_models['loader']()

        features, _ = worker_models['extractor'](file_path)
        danceability = features['rhythm.danceability']
        energy = features['lowlevel.average_loudness']
        vibe = get_vibe(energy, danceability)

        activations = worker_models['embeddings'](audio)
        predictions = worker_models['genre'](activations)
        mean_predictions = np.mean(predictions, axis=0)

        top_index = np.argmax(mean_predictions)
        top_genre = worker_models['labels'][top_index]
        if "---" in top_genre:
            top_genre = top_genre.split("---")[-1]

        current_file_path = file_path
        if file_path.lower().endswith('.wav'):
            converted = convert_wav_to_aiff(file_path)
            if converted != file_path:
                current_file_path = converted
                filename = os.path.basename(converted)
                wav_to_delete = file_path  # defer deletion until after save

        audio_file = File(current_file_path)
        if audio_file is None:
            return {"original_path": current_file_path, "filename": filename, "status": "error", "error": "Not a supported audio file"}

        if audio_file.tags is None:
            audio_file.add_tags()

        for key in list(audio_file.tags.keys()):
            if key.startswith("TIT1") or key.startswith("TIT3") or key.startswith("COMM"):
                del audio_file.tags[key]

        audio_file.tags["TCON"] = TCON(encoding=3, text=[top_genre])
        audio_file.tags["TIT3"] = TIT3(encoding=3, text=[f"E: {energy:.1f}"])
        audio_file.tags["COMM::eng"] = COMM(encoding=3, lang='eng', desc='', text=[f"{vibe} | D: {danceability:.2f}"])
        audio_file.save(v2_version=3)

        # Safe to remove the source WAV now that the AIFF is fully tagged
        if wav_to_delete:
            os.remove(wav_to_delete)

        return {"original_path": current_file_path, "filename": filename, "genre": top_genre, "status": "success"}
    except Exception as e:
        return {"filename": filename, "status": "error", "error": str(e)}


def ensure_models_exist():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for url in MODEL_URLS.values():
        filepath = os.path.join(MODELS_DIR, url.split('/')[-1])
        if not os.path.exists(filepath):
            print(f"Downloading {os.path.basename(filepath)}...")
            download_with_progress(url, filepath)


def sanitize_folder_name(name):
    clean = name.split('---')[-1]
    return "".join([c for c in clean if c.isalnum() or c in (' ', '-', '_')]).strip()


if __name__ == "__main__":
    ensure_models_exist()

    print("📋 Scanning library...")
    all_tracks = [
        os.path.join(root, f)
        for root, _, files in os.walk(LIBRARY_PATH)
        for f in files
        if not f.startswith('.') and f.lower().endswith(('.mp3', '.wav', '.aiff', '.aif'))
    ]

    total = len(all_tracks)
    if total == 0:
        print("No tracks found. Check your LIBRARY_PATH.")
        exit(0)

    print(f"🚀 Found {total} tracks. Parallelizing across {NUM_WORKERS} cores...")

    move_queue = []
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(analyze_track, all_tracks), 1):
            if res['status'] == "success":
                print(f"[{i / total * 100:.1f}%] ({i}/{total}) ✅ {res['genre']} | {res['filename']}")
                move_queue.append(res)
            else:
                print(f"[{i / total * 100:.1f}%] ({i}/{total}) ❌ Failed: {res['filename']} - {res['error']}")

    if move_queue:
        choice = input(f"\nAnalysis complete. Move {len(move_queue)} files? (y/n): ")
        if choice.lower() == 'y':
            moved = failed = 0
            for item in move_queue:
                try:
                    target_dir = os.path.join(DESTINATION_PATH, sanitize_folder_name(item['genre']))
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(item['original_path'], os.path.join(target_dir, item['filename']))
                    moved += 1
                except Exception as e:
                    failed += 1
                    print(f"❌ Move failed: {item['filename']} — {e}")
            print(f"\n✅ Moved: {moved} | Failed: {failed}")