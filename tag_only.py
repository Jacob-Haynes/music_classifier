import os
import json
import numpy as np
import essentia
import essentia.standard as es
from mutagen.id3 import ID3, TCON, TIT3, COMM, ID3NoHeaderError
from mutagen import File
from multiprocessing import Pool

essentia.log.warningActive = False
essentia.log.infoActive = False

LIBRARY_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music'
MODELS_DIR = './models'
NUM_WORKERS = 10
MUSIC_EXTENSIONS = ('.mp3', '.aiff', '.aif', '.wav')

worker_models = {}


def init_worker():
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    with open(os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.json"), 'r') as f:
        metadata = json.load(f)
    worker_models['labels'] = metadata['classes']
    worker_models['loader'] = es.MonoLoader(sampleRate=16000)
    worker_models['extractor'] = es.MusicExtractor(lowlevelStats=['mean'])
    worker_models['embeddings'] = es.TensorflowPredictEffnetDiscogs(
        graphFilename=os.path.join(MODELS_DIR, "discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
    worker_models['genre'] = es.TensorflowPredict2D(
        graphFilename=os.path.join(MODELS_DIR, "genre_discogs400-discogs-effnet-1.pb"),
        input="serving_default_model_Placeholder", output="PartitionedCall:0")


def get_vibe(energy, dance):
    """Translates the numbers into a human-readable vibe word."""
    if energy > 0.7:
        return "PEAK" if dance > 1.2 else "INTENSE"
    elif energy > 0.4:
        return "GROOVE" if dance > 1.2 else "MODERN"
    else:
        return "HYPNOTIC" if dance > 1.2 else "DEEP"


def patch_tags(file_path):
    try:
        worker_models['loader'].configure(filename=file_path, sampleRate=16000)
        audio_data = worker_models['loader']()
        features, _ = worker_models['extractor'](file_path)

        danceability = features['rhythm.danceability']
        energy = features['lowlevel.average_loudness']
        current_genre = os.path.basename(os.path.dirname(file_path))

        # Determine the vibe word
        vibe = get_vibe(energy, danceability)

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
            except Exception:
                pass

        audio = File(current_file_path)
        if audio is None:
            return {"filename": os.path.basename(current_file_path), "status": "error", "error": "Not a supported audio file"}

        if audio.tags is None:
            audio.add_tags()

        # Clean up old/conflicting tags for Rekordbox
        # We delete TIT1/TIT3 and any COMM frames to ensure a clean overwrite
        for key in list(audio.tags.keys()):
            if key.startswith("TIT1") or key.startswith("TIT3") or key.startswith("COMM"):
                del audio.tags[key]

        # Update tags using ID3 frames
        audio.tags["TCON"] = TCON(encoding=3, text=[current_genre])
        audio.tags["TIT3"] = TIT3(encoding=3, text=[f"E: {energy:.1f}"])
        # desc='' is essential for Rekordbox
        comment_text = f"{vibe} | D: {danceability:.2f}"
        audio.tags["COMM::eng"] = COMM(encoding=3, lang='eng', desc='', text=[comment_text])

        # Save with ID3v2.3 for maximum Rekordbox compatibility
        audio.save(v2_version=3)

        return {
            "filename": os.path.basename(file_path),
            "status": "success",
            "genre": current_genre,
            "energy": energy,
            "dance": danceability,
            "vibe": vibe
        }
    except Exception as e:
        return {"filename": os.path.basename(file_path), "status": "error", "error": str(e)}


if __name__ == "__main__":
    print("🔍 Scanning library...")
    files_to_patch = [os.path.join(r, f) for r, _, fs in os.walk(LIBRARY_PATH) for f in fs if
                      not f.startswith('.') and f.lower().endswith(MUSIC_EXTENSIONS)]

    total = len(files_to_patch)
    print(f"🚀 Found {total} tracks. Patching with Vibe labels...")

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(patch_tags, files_to_patch), 1):
            percent = (i / total) * 100
            if res['status'] == "success":
                # Clean Terminal Output
                print(
                    f"[{percent:.1f}%] ({i}/{total}) ✅ {res['genre']} | {res['vibe']} | E: {res['energy']:.1f} | {res['filename']}")
            else:
                print(f"[{percent:.1f}%] ({i}/{total}) ❌ Failed: {res['filename']}")

    print("\n🎉 Patching complete. Run 'Reload Tags' in Rekordbox!")