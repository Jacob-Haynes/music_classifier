import os
import subprocess
import urllib.request


def get_vibe(energy, dance):
    if energy > 0.7:
        return "PEAK" if dance > 1.2 else "INTENSE"
    elif energy > 0.4:
        return "GROOVE" if dance > 1.2 else "MODERN"
    else:
        return "HYPNOTIC" if dance > 1.2 else "DEEP"


def convert_wav_to_aiff(file_path):
    """
    Converts a WAV to AIFF using afconvert (macOS built-in).
    Returns the new file path on success, or the original path if conversion fails.
    Does NOT delete the source — caller is responsible for cleanup after a successful save.
    """
    aiff_path = file_path.rsplit('.', 1)[0] + '.aif'
    result = subprocess.run(
        ['afconvert', '-f', 'AIFF', '-d', 'BEI24', file_path, aiff_path],
        capture_output=True
    )
    if result.returncode == 0 and os.path.exists(aiff_path) and os.path.getsize(aiff_path) > 0:
        return aiff_path
    # Clean up a partial/zero-byte output if conversion failed
    if os.path.exists(aiff_path) and os.path.getsize(aiff_path) == 0:
        os.remove(aiff_path)
    return file_path


def download_with_progress(url, filepath):
    """Downloads a file to a temp path, then atomically renames it on success."""
    filename = os.path.basename(filepath)
    tmp_path = filepath + '.tmp'

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            mb_done = min(block_num * block_size, total_size) / 1_000_000
            mb_total = total_size / 1_000_000
            print(f"\r  {filename}: {mb_done:.1f} / {mb_total:.1f} MB", end='', flush=True)

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook)
        print()
        os.rename(tmp_path, filepath)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Failed to download {filename}: {e}")