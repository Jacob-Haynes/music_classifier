# Music Classifier & DJ Organizer

An AI-powered music organization toolset designed for DJs. This project uses machine learning models (Essentia & TensorFlow) to automatically analyze, tag, and organize your music library based on genre, energy, and danceability.

## Features

- **Automated Genre Classification:** Uses the `discogs-effnet` model to predict genres from the Discogs-400 dataset.
- **Vibe Analysis:** Calculates energy and danceability scores to categorize tracks into "vibes" (e.g., PEAK, GROOVE, HYPNOTIC, DEEP).
- **Auto-Tagging:** Writes analysis results directly to file metadata (ID3 tags) including Genre, Subtitle, and Comments.
- **Library Organization:** Automatically sorts files into genre-based folders.
- **Parallel Processing:** Utilizes multiple CPU cores for fast analysis of large music libraries.

## Scripts

### 1. `dj_organiser.py`
The primary tool for scanning a library, predicting genres, tagging files, and optionally moving them into organized folders.
- **Workflow:** Scan -> Analyze -> Tag -> Confirm Move.
- **Tags Updated:** 
  - `TCON` (Genre): Predicted sub-genre.
  - `TIT1` (Content group): Energy score.
  - `COMM` (Comment): Danceability and confidence score.

### 2. `tag_only.py`
A specialized script for updating tags on an already organized library without moving files. It introduces a "Vibe" label based on energy/danceability thresholds.
- **Workflow:** Scan -> Analyze -> Tag.
- **Tags Updated:**
  - `TCON` (Genre): Uses existing parent folder name as genre.
  - `TIT3` (Subtitle): Energy score.
  - `COMM` (Comment): Vibe label (e.g., "PEAK") and danceability.

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
   The scripts will automatically download the necessary models to the `./models` directory on the first run.

## Configuration

Before running the scripts, open them and update the `LIBRARY_PATH` variable to point to your music library:

```python
LIBRARY_PATH = '/path/to/your/music'
```

You can also adjust `NUM_WORKERS` to match your CPU core count for optimal performance.

## Usage

### Run the Organizer
```bash
python dj_organiser.py
```

### Run the Tag Patcher (Vibe Labels)
```bash
python tag_only.py
```

## How It Works

1. **Audio Loading:** Tracks are resampled to 16kHz (mono) as required by the Effnet models.
2. **Feature Extraction:** Essentia's `MusicExtractor` calculates low-level and rhythmic features.
3. **Inference:** 
   - Audio is passed through the `discogs-effnet` feature extractor to get embeddings.
   - Embeddings are fed into a classification head to predict genres.
4. **Vibe Logic:** `tag_only.py` applies a custom heuristic:
   - **PEAK:** High Energy, High Danceability.
   - **GROOVE:** Moderate Energy, High Danceability.
   - **HYPNOTIC:** Low Energy, High Danceability.
   - **INTENSE:** High Energy, Low Danceability.
   - **DEEP:** Low Energy, Low Danceability.

## Rekordbox Integration

After running these scripts, you can see the updated tags in Rekordbox by selecting your tracks, right-clicking, and choosing **"Reload Tags"**. This will import the AI-generated Genre, Energy, and Vibe comments into your collection.
