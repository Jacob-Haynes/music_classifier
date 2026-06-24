import os
import essentia
import essentia.standard as es
from mutagen.id3 import TCON, TIT3, COMM
from mutagen import File
from multiprocessing import Pool
from utils import get_vibe, convert_wav_to_aiff

essentia.log.warningActive = False
essentia.log.infoActive = False

LIBRARY_PATH = '/Users/jacobhaynes/Library/CloudStorage/GoogleDrive-jacobhaynes49@gmail.com/My Drive/Music'
NUM_WORKERS = 10
MUSIC_EXTENSIONS = ('.mp3', '.aiff', '.aif', '.wav')

worker_models = {}


def init_worker():
    essentia.log.warningActive = False
    essentia.log.infoActive = False
    worker_models['extractor'] = es.MusicExtractor(lowlevelStats=['mean'])


def patch_tags(file_path):
    wav_to_delete = None
    try:
        features, _ = worker_models['extractor'](file_path)
        danceability = features['rhythm.danceability']
        energy = features['lowlevel.average_loudness']
        current_genre = os.path.basename(os.path.dirname(file_path))
        vibe = get_vibe(energy, danceability)

        current_file_path = file_path
        if file_path.lower().endswith('.wav'):
            converted = convert_wav_to_aiff(file_path)
            if converted != file_path:
                current_file_path = converted
                wav_to_delete = file_path  # defer deletion until after save

        audio = File(current_file_path)
        if audio is None:
            return {"filename": os.path.basename(current_file_path), "status": "error", "error": "Not a supported audio file"}

        if audio.tags is None:
            audio.add_tags()

        for key in list(audio.tags.keys()):
            if key.startswith("TIT1") or key.startswith("TIT3") or key.startswith("COMM"):
                del audio.tags[key]

        audio.tags["TCON"] = TCON(encoding=3, text=[current_genre])
        audio.tags["TIT3"] = TIT3(encoding=3, text=[f"E: {energy:.1f}"])
        audio.tags["COMM::eng"] = COMM(encoding=3, lang='eng', desc='', text=[f"{vibe} | D: {danceability:.2f}"])
        audio.save(v2_version=3)

        if wav_to_delete:
            os.remove(wav_to_delete)

        return {
            "filename": os.path.basename(current_file_path),
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
    files_to_patch = [
        os.path.join(r, f)
        for r, _, fs in os.walk(LIBRARY_PATH)
        for f in fs
        if not f.startswith('.') and f.lower().endswith(MUSIC_EXTENSIONS)
    ]

    total = len(files_to_patch)
    if total == 0:
        print("No tracks found. Check your LIBRARY_PATH.")
        exit(0)

    print(f"🚀 Found {total} tracks. Patching with Vibe labels...")

    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(patch_tags, files_to_patch), 1):
            percent = (i / total) * 100
            if res['status'] == "success":
                print(f"[{percent:.1f}%] ({i}/{total}) ✅ {res['genre']} | {res['vibe']} | E: {res['energy']:.1f} | {res['filename']}")
            else:
                print(f"[{percent:.1f}%] ({i}/{total}) ❌ Failed: {res['filename']}")

    print("\n🎉 Patching complete. Run 'Reload Tags' in Rekordbox!")