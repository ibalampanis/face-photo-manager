import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Optional
import threading


class FaceDatabase:
    """Thread-safe SQLite database for face data persistence."""

    def __init__(self, db_path: str = "data/faces.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                source_folder TEXT NOT NULL,
                work_folder TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                processed INTEGER DEFAULT 0,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                UNIQUE(project_id, filename)
            );

            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER NOT NULL,
                project_id INTEGER NOT NULL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                embedding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT -1,
                tag TEXT DEFAULT '',
                confidence REAL DEFAULT 0.0,
                FOREIGN KEY (image_id) REFERENCES images(id),
                FOREIGN KEY (project_id) REFERENCES projects(id)
            );

            CREATE TABLE IF NOT EXISTS person_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                tag TEXT NOT NULL DEFAULT 'Unknown',
                representative_face_id INTEGER,
                FOREIGN KEY (project_id) REFERENCES projects(id),
                UNIQUE(project_id, cluster_id)
            );

            CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(project_id, cluster_id);
            CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);
            CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id);
        """)
        conn.commit()

    def create_project(self, name: str, source_folder: str, work_folder: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT OR REPLACE INTO projects (name, source_folder, work_folder) VALUES (?, ?, ?)",
            (name, source_folder, work_folder)
        )
        conn.commit()
        return cursor.lastrowid

    def get_project(self, name: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM projects WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_all_projects(self) -> list:
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def add_image(self, project_id: int, filename: str, filepath: str,
                  width: int = 0, height: int = 0) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT OR IGNORE INTO images (project_id, filename, filepath, width, height) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, filename, filepath, width, height)
        )
        conn.commit()
        if cursor.lastrowid == 0:
            row = conn.execute(
                "SELECT id FROM images WHERE project_id = ? AND filename = ?",
                (project_id, filename)
            ).fetchone()
            return row["id"]
        return cursor.lastrowid

    def mark_image_processed(self, image_id: int):
        conn = self._get_conn()
        conn.execute("UPDATE images SET processed = 1 WHERE id = ?", (image_id,))
        conn.commit()

    def get_unprocessed_images(self, project_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM images WHERE project_id = ? AND processed = 0",
            (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_images(self, project_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM images WHERE project_id = ?", (project_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def add_face(self, image_id: int, project_id: int, bbox: tuple,
                 embedding: np.ndarray, confidence: float = 0.0) -> int:
        conn = self._get_conn()
        emb_blob = embedding.astype(np.float32).tobytes()
        cursor = conn.execute(
            "INSERT INTO faces (image_id, project_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, "
            "embedding, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, project_id, *bbox, emb_blob, confidence)
        )
        conn.commit()
        return cursor.lastrowid

    def get_faces_for_project(self, project_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM faces WHERE project_id = ?", (project_id,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            results.append(d)
        return results

    def get_faces_for_image(self, image_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM faces WHERE image_id = ?", (image_id,)
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            results.append(d)
        return results

    def update_face_clusters(self, face_ids: list, cluster_ids: list):
        conn = self._get_conn()
        for fid, cid in zip(face_ids, cluster_ids):
            conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (int(cid), fid))
        conn.commit()

    def set_person_tag(self, project_id: int, cluster_id: int, tag: str,
                       representative_face_id: int = None):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO person_tags (project_id, cluster_id, tag, representative_face_id) "
            "VALUES (?, ?, ?, ?)",
            (project_id, cluster_id, tag, representative_face_id)
        )
        conn.execute(
            "UPDATE faces SET tag = ? WHERE project_id = ? AND cluster_id = ?",
            (tag, project_id, cluster_id)
        )
        conn.commit()

    def get_person_tags(self, project_id: int) -> dict:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM person_tags WHERE project_id = ?", (project_id,)
        ).fetchall()
        return {r["cluster_id"]: dict(r) for r in rows}

    def get_images_by_cluster(self, project_id: int, cluster_id: int) -> list:
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT DISTINCT i.* FROM images i
               JOIN faces f ON f.image_id = i.id
               WHERE f.project_id = ? AND f.cluster_id = ?""",
            (project_id, cluster_id)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_images_by_clusters(self, project_id: int, cluster_ids: list) -> list:
        if not cluster_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(cluster_ids))
        rows = conn.execute(
            f"""SELECT DISTINCT i.* FROM images i
                JOIN faces f ON f.image_id = i.id
                WHERE f.project_id = ? AND f.cluster_id IN ({placeholders})""",
            [project_id] + [int(c) for c in cluster_ids]
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_face_count(self, project_id: int) -> dict:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT cluster_id, COUNT(*) as cnt FROM faces WHERE project_id = ? GROUP BY cluster_id",
            (project_id,)
        ).fetchall()
        return {r["cluster_id"]: r["cnt"] for r in rows}

    def delete_project_faces(self, project_id: int):
        conn = self._get_conn()
        conn.execute("DELETE FROM faces WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM person_tags WHERE project_id = ?", (project_id,))
        conn.execute("UPDATE images SET processed = 0 WHERE project_id = ?", (project_id,))
        conn.commit()

    def merge_clusters(self, project_id: int, source_cluster: int, target_cluster: int):
        conn = self._get_conn()
        target_tag_row = conn.execute(
            "SELECT tag FROM person_tags WHERE project_id = ? AND cluster_id = ?",
            (project_id, target_cluster)
        ).fetchone()
        target_tag = target_tag_row["tag"] if target_tag_row else "Unknown"

        conn.execute(
            "UPDATE faces SET cluster_id = ?, tag = ? WHERE project_id = ? AND cluster_id = ?",
            (target_cluster, target_tag, project_id, source_cluster)
        )
        conn.execute(
            "DELETE FROM person_tags WHERE project_id = ? AND cluster_id = ?",
            (project_id, source_cluster)
        )
        conn.commit()