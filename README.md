# 📸 Face Photo Manager

A Streamlit application for managing large photo collections using SOTA face recognition.
Automatically detects faces, clusters them by identity, allows tagging, and enables
photo selection/export by person.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Face Detection | RetinaFace (via InsightFace) |
| Face Recognition | ArcFace embeddings (buffalo_l model) |
| Similarity Search | FAISS (Facebook AI Similarity Search) |
| Clustering | DBSCAN with cosine distance |
| Database | SQLite |
| UI | Streamlit |

## Installation

```bash
# Clone and enter the project
cd face_photo_manager

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional):
pip install onnxruntime-gpu