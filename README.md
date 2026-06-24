# Music Classifier & DJ Organizer

An AI-powered music organization toolset designed for DJs. This project uses machine learning models (Essentia & TensorFlow) to automatically analyze, tag, and organize your music library based on genre, energy, and danceability.

## Features

- **Automated Genre Classification:** Uses the `discogs-effnet` model to predict genres from the Discogs-400 dataset.
- **Vibe Analysis:** Calculates energy and danceability scores to categorize tracks into vibes (PEAK, INTENSE, GROOVE, MODERN, HYPNOTIC, DEEP).
- **Auto-Tagging:** Writes analysis results directly to file metadata (ID3 tags) including Genre, Subtitle, and Comments.
- **Library Organization:** Automatically sorts files into genre-based folders.
- **Parallel Processing:** Utilizes multiple CPU cores for fast analysis of large music libraries.

## Scripts

### 1. `dj_organiser.py`
The primary tool for scanning an unsorted library, predicting genres, tagging files, and optionally moving them into organized folders.
- **Workflow:** Scan -> Analyze -> Tag -> Confirm Move.
- **Tags Updated:**
  - `TCON` (Genre): Predicted sub-genre.
  - `TIT3` (Subtitle): Energy score (`E: x.x`).
  - `COMM` (Comment): Vibe label and danceability (`PEAK | D: x.xx`).

### 2. `tag_only.py`
For updating tags on an already organized library without moving files. Uses the existing parent folder name as the genre rather than running AI inference.
- **Workflow:** Scan -> Analyze -> Tag.
- **Tags Updated:**
  - `TCON` (Genre): Parent folder name.
  - `TIT3` (Subtitle): Energy score (`E: x.x`).
  - `COMM` (Comment): Vibe label and danceability (`PEAK | D: x.xx`).

### 4. `utils.py`
Shared helpers used by all three scripts: `get_vibe()`, `convert_wav_to_aiff()`, and `download_with_progress()`. Not run directly.

### 3. `tag_repair.py`
Scans an already organized library and re-tags only tracks whose comment tag doesn't match the expected `VIBE | D: x.xx` format. Useful for fixing tracks tagged by an older version of the scripts without re-processing everything.
- **Workflow:** Scan -> Check tag format -> Re-tag if malformed.
- **Tags Updated:** Same as `tag_only.py`, only on tracks that need it.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd music-classifier
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   *Note: Essentia with TensorFlow support is required.*
   ```bash
   pip install numpy mutagen essentia-tensorflow
   ```

4. **Models:**
   `dj_organiser.py` automatically downloads the required models (~300 MB) to a `models/` directory alongside the script on first run, with progress output. `tag_only.py`, `tag_repair.py`, and `utils.py` use only Essentia's built-in feature extractor and require no model downloads.

## Configuration

Each script has a `LIBRARY_PATH` variable at the top — update it to point to your music library before running:

```python
LIBRARY_PATH = '/path/to/your/music'
```

`dj_organiser.py` also has a `DESTINATION_PATH` for where organized files should be moved. You can also adjust `NUM_WORKERS` to match your CPU core count.

## Usage

### Organize a new batch of tracks (analyze, tag, and move)
```bash
python dj_organiser.py
```

### Re-tag an already organized library (no moving)
```bash
python tag_only.py
```

### Fix tracks with malformed tags only
```bash
python tag_repair.py
```

## How It Works

1. **Audio Loading:** Tracks are resampled to 16kHz (mono) as required by the Effnet models.
2. **Feature Extraction:** Essentia's `MusicExtractor` calculates low-level and rhythmic features.
3. **Inference (`dj_organiser.py` only):**
   - Audio is passed through the `discogs-effnet` feature extractor to get embeddings.
   - Embeddings are fed into a classification head to predict genres.
4. **Vibe Logic:** Applied by all three scripts using energy and danceability thresholds:

| Vibe | Energy | Danceability |
|---|---|---|
| PEAK | High (> 0.7) | High (> 1.2) |
| INTENSE | High (> 0.7) | Low (≤ 1.2) |
| GROOVE | Moderate (0.4–0.7) | High (> 1.2) |
| MODERN | Moderate (0.4–0.7) | Low (≤ 1.2) |
| HYPNOTIC | Low (≤ 0.4) | High (> 1.2) |
| DEEP | Low (≤ 0.4) | Low (≤ 1.2) |

## Rekordbox Integration

After running any script, select the updated tracks in Rekordbox, right-click, and choose **"Reload Tags"** to import the AI-generated Genre, Energy, and Vibe comments into your collection.