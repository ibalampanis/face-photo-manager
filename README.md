# 📸 Face Photo Manager

A Streamlit application for managing large photo collections using SOTA face recognition.
Automatically detects faces, clusters them by identity, allows tagging, and enables
photo selection/export by person.

## Technology Stack

| Component         | Technology                                    |
| ----------------- | --------------------------------------------- |
| Face Detection    | RetinaFace (via InsightFace)                  |
| Face Recognition  | ArcFace embeddings (`buffalo_l` model)        |
| Similarity Search | FAISS (Facebook AI Similarity Search)         |
| Clustering        | DBSCAN with cosine distance                   |
| Database          | SQLite                                        |
| UI                | Streamlit                                     |
| Inference Backend | ONNX Runtime ≥ 1.18 (CoreML on Apple Silicon) |

## Installation

```bash
# Clone the repo
git clone git@github.com:ibalampanis/face-photo-manager.git
cd face-photo-manager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
source venv/bin/activate
streamlit run app.py
```

> **Apple Silicon (M1/M2/M3):** GPU acceleration via CoreML is enabled automatically.
> Make sure `onnxruntime >= 1.18` is installed — do **not** install `onnxruntime-silicon`
> (deprecated), as it conflicts with NumPy 2.

## Notes

- Face models (`buffalo_l`) are downloaded automatically on first run to `~/.insightface/models/`
- Photo data and the SQLite database are stored locally and excluded from version control
