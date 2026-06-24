import os
import re
import sys
import numpy as np
import essentia
import essentia.standard as es
from mutagen import File
from mutagen.id3 import TCON, TIT3, COMM
from multiprocessing import Pool
from utils import get_vibe

essentia.log.warningActive = False
essentia.log.infoActive = False

LIBRARY_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
NUM_WORKERS = 10
MUSIC_EXTENSIONS = ('.mp3', '.aiff', '.aif', '.wav')

REQUIRED_MODELS = [
    "discogs-effnet-bs64-1.pb",
    "mood_party-discogs-effnet-1.pb",
    "mood_aggressive-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb",
]

VALID_COMM_PATTERN = re.compile(r'^(PEAK|HARD|FLOOR|HYPNOTIC|LATE|DRIVE) \| P: \d+\.\d{2}$')
VALID_TIT3_PATTERN = re.compile(r'^E: \d+\.\d{2}$')

worker_models = {}


def init_worker():
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    worker_models['loader'] = es.MonoLoader(sampleRate=16000)
    worker_models['extractor'] = es.MusicExtractor(lowlevelStats=['mean'], rhythmStats=['mean'])
    worker_models['embeddings'] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
    worker_models['mood_party'] = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, "mood_party-discogs-effnet-1.pb"),
        output="model/Softmax")
    worker_models['mood_aggressive'] = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, "mood_aggressive-discogs-effnet-1.pb"),
        output="model/Softmax")
    worker_models['mood_relaxed'] = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, "mood_relaxed-discogs-effnet-1.pb"),
        output="model/Softmax")


def _tags_are_valid(audio):
    if audio is None or audio.tags is None:
        return False
    comm = audio.tags.get("COMM::eng")
    tit3 = audio.tags.get("TIT3")
    if comm is None or tit3 is None:
        return False
    return (
        VALID_COMM_PATTERN.match(str(comm.text[0]).strip()) and
        VALID_TIT3_PATTERN.match(str(tit3.text[0]).strip())
    )


def repair_track(file_path):
    try:
        audio = File(file_path)
        if _tags_are_valid(audio):
            return {"filename": os.path.basename(file_path), "status": "skipped"}

        worker_models['loader'].configure(filename=file_path, sampleRate=16000)
        audio_data = worker_models['loader']()

        features, _ = worker_models['extractor'](file_path)
        energy = features['rhythm.beats_loudness.mean']
        current_genre = os.path.basename(os.path.dirname(file_path))

        activations = worker_models['embeddings'](audio_data)
        party = float(np.mean(worker_models['mood_party'](activations), axis=0)[1])
        aggressive = float(np.mean(worker_models['mood_aggressive'](activations), axis=0)[0])
        relaxed = float(np.mean(worker_models['mood_relaxed'](activations), axis=0)[1])
        vibe = get_vibe(party, aggressive, relaxed)

        if audio is None:
            return {"filename": os.path.basename(file_path), "status": "error", "error": "Not a supported audio file"}
        if audio.tags is None:
            audio.add_tags()

        for key in list(audio.tags.keys()):
            if key.startswith("TIT1") or key.startswith("TIT3") or key.startswith("COMM"):
                del audio.tags[key]

        audio.tags["TCON"] = TCON(encoding=3, text=[current_genre])
        audio.tags["TIT3"] = TIT3(encoding=3, text=[f"E: {energy:.2f}"])
        audio.tags["COMM::eng"] = COMM(encoding=3, lang='eng', desc='', text=[f"{vibe} | P: {party:.2f}"])
        audio.save(v2_version=3)

        return {"filename": os.path.basename(file_path), "status": "repaired", "vibe": vibe, "energy": energy}
    except Exception as e:
        return {"filename": os.path.basename(file_path), "status": "error", "error": str(e)}


if __name__ == "__main__":
    missing = [m for m in REQUIRED_MODELS if not os.path.exists(os.path.join(MODELS_DIR, m))]
    if missing:
        print(f"❌ Missing models: {', '.join(missing)}")
        print("Run dj_organiser.py once to download all required models.")
        sys.exit(1)

    print("🔍 Scanning library for tracks with malformed tags...")
    files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(LIBRARY_PATH)
        for f in fs
        if not f.startswith('.') and f.lower().endswith(MUSIC_EXTENSIONS)
    ]

    total = len(files)
    if total == 0:
        print("No tracks found. Check your LIBRARY_PATH.")
        exit(0)

    print(f"📋 Found {total} tracks. Checking tags...")

    repaired = skipped = errors = 0
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(repair_track, files), 1):
            percent = (i / total) * 100
            if res['status'] == "repaired":
                repaired += 1
                print(f"[{percent:.1f}%] ({i}/{total}) 🔧 {res['vibe']} | E: {res['energy']:.2f} | {res['filename']}")
            elif res['status'] == "skipped":
                skipped += 1
            else:
                errors += 1
                print(f"[{percent:.1f}%] ({i}/{total}) ❌ {res['filename']} — {res['error']}")

    print(f"\n✅ Done. Repaired: {repaired} | Already correct: {skipped} | Errors: {errors}")
    if repaired:
        print("Run 'Reload Tags' in Rekordbox to pick up the changes.")