import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import insightface
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import faiss
import logging

import onnxruntime

logger = logging.getLogger(__name__)


def _best_providers() -> list:
    """Return ONNX execution providers in priority order: CUDA > CoreML (MPS) > CPU."""
    available = onnxruntime.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",      # NVIDIA GPU
        "CoreMLExecutionProvider",    # Apple Silicon (MPS / Neural Engine)
        "CPUExecutionProvider",       # fallback
    ]
    providers = [p for p in preferred if p in available]
    logger.info("ONNX providers selected: %s", providers)
    return providers


class FaceEngine:
    """SOTA face detection and recognition using InsightFace (ArcFace/RetinaFace)."""

    def __init__(self, det_size: Tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5):
        self.det_size = det_size
        self.det_thresh = det_thresh
        self._app = None

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self._app = FaceAnalysis(
                name="buffalo_l",
                providers=_best_providers()
            )
            self._app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
        return self._app

    def detect_faces(self, image_path: str) -> list:
        """Detect faces and extract embeddings from an image."""
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return []

        faces = self.app.get(img)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            embedding = face.embedding
            confidence = float(face.det_score)
            results.append({
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "embedding": embedding,
                "confidence": confidence,
            })
        return results

    def extract_face_thumbnail(self, image_path: str, bbox: tuple,
                                padding: float = 0.3, size: int = 150) -> Optional[np.ndarray]:
        """Extract a face thumbnail with padding."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        fw, fh = x2 - x1, y2 - y1
        pad_w, pad_h = int(fw * padding), int(fh * padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_crop = cv2.resize(face_crop, (size, size))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        return face_crop

    @staticmethod
    def cluster_faces(embeddings: np.ndarray, eps: float = 0.65,
                      min_samples: int = 2) -> np.ndarray:
        """Cluster face embeddings using DBSCAN with cosine distance via FAISS."""
        if len(embeddings) == 0:
            return np.array([])

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = (embeddings / norms).astype(np.float32)

        # Use DBSCAN with precomputed cosine distance
        # cosine_distance = 1 - cosine_similarity
        similarity_matrix = np.dot(normalized, normalized.T)
        distance_matrix = 1.0 - similarity_matrix
        distance_matrix = np.clip(distance_matrix, 0, 2)

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = clustering.fit_predict(distance_matrix)

        # Assign unclustered faces (-1) to nearest cluster using FAISS
        clustered_mask = labels >= 0
        if clustered_mask.any() and (~clustered_mask).any():
            clustered_embs = normalized[clustered_mask]
            clustered_labels = labels[clustered_mask]

            dim = clustered_embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(clustered_embs)

            unclustered_embs = normalized[~clustered_mask]
            distances, indices = index.search(unclustered_embs, 1)

            unclustered_indices = np.where(~clustered_mask)[0]
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                # Only assign if similarity is high enough
                if dist[0] > (1.0 - eps * 1.2):
                    labels[unclustered_indices[i]] = clustered_labels[idx[0]]

        # Re-assign remaining -1 to individual clusters
        max_label = labels.max() if len(labels) > 0 else -1
        for i in range(len(labels)):
            if labels[i] == -1:
                max_label += 1
                labels[i] = max_label

        return labels

    @staticmethod
    def find_representative_faces(embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Find the most representative face (closest to centroid) per cluster."""
        representatives = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            cluster_embs = embeddings[mask]
            centroid = cluster_embs.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            normalized = cluster_embs / (np.linalg.norm(cluster_embs, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(normalized, centroid)
            best_idx = np.argmax(similarities)

            original_indices = np.where(mask)[0]
            representatives[int(label)] = int(original_indices[best_idx])

        return representatives