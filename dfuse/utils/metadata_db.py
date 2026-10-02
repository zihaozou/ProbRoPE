"""
SQLite-based metadata database for video datasets.

This module provides a thread-safe, multi-process-safe metadata storage using SQLite
with WAL (Write-Ahead Logging) mode. It supports monocular, stereo, and multiview
dataset types with normalized schema for queryable camera parameters and prompts.

Usage:
    from dfuse.utils.metadata_db import MetadataDB

    # Context manager (recommended)
    with MetadataDB("metadata.db") as db:
        db.upsert_sequence("seq_001", "stereo", 100, 1080, 1920, {"images": "images.h5"})
        db.upsert_camera_params_batch("seq_001", "left", intrinsics_list, extrinsics_list)
        db.upsert_prompt("seq_001", "left", "A video of a car driving...")

    # Manual management
    db = MetadataDB("metadata.db")
    try:
        seq = db.get_sequence("seq_001")
    finally:
        db.close()

Concurrency Notes:
    - WAL mode allows concurrent readers with one writer
    - Multiple processes can safely write (SQLite serializes via locking)
    - busy_timeout (default 30s) handles lock contention automatically
    - For bulk inserts, use batch methods with transactions
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


_SCHEMA_SQL = """
-- Sequence metadata table
CREATE TABLE IF NOT EXISTS sequences (
    seq_id TEXT PRIMARY KEY,
    dataset_type TEXT NOT NULL CHECK (dataset_type IN ('monocular', 'stereo', 'multiview', 'static')),
    num_frames INTEGER NOT NULL,
    height INTEGER NOT NULL,
    width INTEGER NOT NULL,
    paths_json TEXT NOT NULL,
    baseline REAL,           -- stereo only
    camera_ids_json TEXT     -- multiview only, e.g., '["cam_0","cam_1"]'
);

-- Per-frame camera parameters (normalized)
CREATE TABLE IF NOT EXISTS camera_params (
    seq_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,     -- 'data', 'left'/'right', 'cam_0'/...
    frame_idx INTEGER NOT NULL,
    intrinsics BLOB NOT NULL,    -- numpy 3x3 float32
    extrinsics BLOB NOT NULL,    -- numpy 4x4 float32
    PRIMARY KEY (seq_id, camera_id, frame_idx)
);

-- Index for fast sequence loading
CREATE INDEX IF NOT EXISTS idx_camera_params_seq ON camera_params(seq_id);

-- Per-camera prompts
CREATE TABLE IF NOT EXISTS camera_prompts (
    seq_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    prompt TEXT NOT NULL,
    PRIMARY KEY (seq_id, camera_id)
);

-- Per-category processing state (CO3D pipeline resume support)
CREATE TABLE IF NOT EXISTS categories (
    category TEXT PRIMARY KEY,
    done_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def _serialize_matrix(matrix: np.ndarray) -> bytes:
    """Serialize numpy matrix to bytes using numpy's native format."""
    buf = io.BytesIO()
    np.save(buf, matrix.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


def _deserialize_matrix(data: bytes) -> np.ndarray:
    """Deserialize bytes to numpy matrix."""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)


class MetadataDB:
    """
    SQLite-based metadata database for video datasets.

    Supports monocular, stereo, and multiview dataset types with:
    - WAL mode for concurrent read/write access
    - Normalized schema for queryable camera parameters
    - Per-camera prompts storage
    - Batch operations for efficient bulk inserts

    Args:
        db_path: Path to SQLite database file (will be created if not exists)
        busy_timeout: Timeout in milliseconds for acquiring locks (default: 30000ms)

    Attributes:
        db_path: Path to the database file
        conn: SQLite connection object
        readonly: Whether database is opened in read-only mode
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        busy_timeout: int = 30000,
        readonly: bool = False,
    ):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            busy_timeout: Timeout in milliseconds for acquiring locks (default: 30000ms)
            readonly: If True, open in read-only mode (safe for NFS multi-process reading)
        """
        self.db_path = Path(db_path)
        self.readonly = readonly

        if readonly:


            if not self.db_path.exists():
                raise FileNotFoundError(f"Database file not found: {self.db_path}")

            uri = f"file:{self.db_path.absolute()}?mode=ro&immutable=1"
            self.conn = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
            )
        else:

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            self.conn = sqlite3.connect(
                str(self.db_path),
                timeout=busy_timeout / 1000.0,
                check_same_thread=False,
            )


            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute(f"PRAGMA busy_timeout={busy_timeout}")


            self.conn.executescript(_SCHEMA_SQL)
            self.conn.commit()


            self._migrate_schema()

        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> "MetadataDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close connection."""
        self.close()

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def _check_readonly(self) -> None:
        """Raise error if attempting write operation in readonly mode."""
        if self.readonly:
            raise RuntimeError(
                "Cannot perform write operation: database opened in read-only mode. "
                "Create a new MetadataDB instance with readonly=False to write."
            )

    def _migrate_schema(self) -> None:
        """Migrate schema."""

        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='sequences'"
        ).fetchone()
        if row is None:
            return

        create_sql = row[0] if isinstance(row, (tuple, list)) else row["sql"]
        if "'static'" in create_sql:
            return


        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sequences_new (
                seq_id TEXT PRIMARY KEY,
                dataset_type TEXT NOT NULL CHECK (dataset_type IN ('monocular', 'stereo', 'multiview', 'static')),
                num_frames INTEGER NOT NULL,
                height INTEGER NOT NULL,
                width INTEGER NOT NULL,
                paths_json TEXT NOT NULL,
                baseline REAL,
                camera_ids_json TEXT
            );
            INSERT OR IGNORE INTO sequences_new SELECT * FROM sequences;
            DROP TABLE sequences;
            ALTER TABLE sequences_new RENAME TO sequences;
        """)
        self.conn.commit()


    def upsert_sequence(
        self,
        seq_id: str,
        dataset_type: str,
        num_frames: int,
        height: int,
        width: int,
        paths: Dict[str, str],
        baseline: Optional[float] = None,
        camera_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Insert or update a sequence record.

        Args:
            seq_id: Unique sequence identifier
            dataset_type: One of 'monocular', 'stereo', 'multiview'
            num_frames: Number of frames in sequence
            height: Frame height in pixels
            width: Frame width in pixels
            paths: Dictionary of data file paths (e.g., {"images": "images.h5"})
            baseline: Stereo baseline distance (stereo only)
            camera_ids: List of camera IDs (multiview only, e.g., ["cam_0", "cam_1"])
        """
        self._check_readonly()
        paths_json = json.dumps(paths)
        camera_ids_json = json.dumps(camera_ids) if camera_ids else None

        self.conn.execute(
            """
            INSERT INTO sequences (seq_id, dataset_type, num_frames, height, width,
                                   paths_json, baseline, camera_ids_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(seq_id) DO UPDATE SET
                dataset_type = excluded.dataset_type,
                num_frames = excluded.num_frames,
                height = excluded.height,
                width = excluded.width,
                paths_json = excluded.paths_json,
                baseline = excluded.baseline,
                camera_ids_json = excluded.camera_ids_json
            """,
            (
                seq_id,
                dataset_type,
                num_frames,
                height,
                width,
                paths_json,
                baseline,
                camera_ids_json,
            ),
        )
        self.conn.commit()

    def get_sequence(self, seq_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full sequence metadata including camera params and prompts.

        Args:
            seq_id: Sequence identifier

        Returns:
            Dictionary with sequence metadata, camera_params dict, and prompts dict.
            Returns None if sequence not found.
        """
        row = self.conn.execute(
            "SELECT * FROM sequences WHERE seq_id = ?", (seq_id,)
        ).fetchone()

        if row is None:
            return None

        result = {
            "seq_id": row["seq_id"],
            "dataset_type": row["dataset_type"],
            "num_frames": row["num_frames"],
            "height": row["height"],
            "width": row["width"],
            "resolution": [row["height"], row["width"]],
            "paths": json.loads(row["paths_json"]),
            "baseline": row["baseline"],
            "camera_ids": json.loads(row["camera_ids_json"])
            if row["camera_ids_json"]
            else None,
        }


        result["camera_params"] = self._load_camera_params_for_sequence(seq_id)


        result["prompts"] = self._load_prompts_for_sequence(seq_id)

        return result

    def get_sequence_info(self, seq_id: str) -> Optional[Dict[str, Any]]:
        """
        Get sequence metadata WITHOUT camera params (faster for listing).

        Args:
            seq_id: Sequence identifier

        Returns:
            Dictionary with basic sequence metadata (no camera_params).
            Returns None if sequence not found.
        """
        row = self.conn.execute(
            "SELECT * FROM sequences WHERE seq_id = ?", (seq_id,)
        ).fetchone()

        if row is None:
            return None

        return {
            "seq_id": row["seq_id"],
            "dataset_type": row["dataset_type"],
            "num_frames": row["num_frames"],
            "height": row["height"],
            "width": row["width"],
            "resolution": [row["height"], row["width"]],
            "paths": json.loads(row["paths_json"]),
            "baseline": row["baseline"],
            "camera_ids": json.loads(row["camera_ids_json"])
            if row["camera_ids_json"]
            else None,
        }

    def get_all_seq_ids(self) -> List[str]:
        """Get list of all sequence IDs in database."""
        rows = self.conn.execute("SELECT seq_id FROM sequences").fetchall()
        return [row["seq_id"] for row in rows]

    def get_all_sequences(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all sequences with full metadata.

        Returns:
            Dictionary mapping seq_id to sequence metadata (including camera_params).
        """
        seq_ids = self.get_all_seq_ids()
        return {seq_id: self.get_sequence(seq_id) for seq_id in seq_ids}

    def get_all_sequence_infos(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all sequences with basic metadata (no camera_params, faster).

        Returns:
            Dictionary mapping seq_id to basic sequence metadata.
        """
        rows = self.conn.execute("SELECT * FROM sequences").fetchall()
        result = {}
        for row in rows:
            result[row["seq_id"]] = {
                "seq_id": row["seq_id"],
                "dataset_type": row["dataset_type"],
                "num_frames": row["num_frames"],
                "height": row["height"],
                "width": row["width"],
                "resolution": [row["height"], row["width"]],
                "paths": json.loads(row["paths_json"]),
                "baseline": row["baseline"],
                "camera_ids": json.loads(row["camera_ids_json"])
                if row["camera_ids_json"]
                else None,
            }
        return result

    def delete_sequence(self, seq_id: str) -> None:
        """
        Delete a sequence and all associated camera params and prompts.

        Args:
            seq_id: Sequence identifier to delete
        """
        self._check_readonly()
        self.conn.execute("DELETE FROM camera_params WHERE seq_id = ?", (seq_id,))
        self.conn.execute("DELETE FROM camera_prompts WHERE seq_id = ?", (seq_id,))
        self.conn.execute("DELETE FROM sequences WHERE seq_id = ?", (seq_id,))
        self.conn.commit()

    def sequence_exists(self, seq_id: str) -> bool:
        """Check if a sequence exists in the database."""
        row = self.conn.execute(
            "SELECT 1 FROM sequences WHERE seq_id = ?", (seq_id,)
        ).fetchone()
        return row is not None


    def upsert_camera_params(
        self,
        seq_id: str,
        camera_id: str,
        frame_idx: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
    ) -> None:
        """
        Insert or update camera parameters for a single frame.

        Args:
            seq_id: Sequence identifier
            camera_id: Camera identifier ('data', 'left', 'right', 'cam_0', etc.)
            frame_idx: Frame index (0-based)
            intrinsics: 3x3 intrinsic matrix
            extrinsics: 4x4 extrinsic matrix (world-to-camera)
        """
        self._check_readonly()
        intrinsics_blob = _serialize_matrix(intrinsics)
        extrinsics_blob = _serialize_matrix(extrinsics)

        self.conn.execute(
            """
            INSERT INTO camera_params (seq_id, camera_id, frame_idx, intrinsics, extrinsics)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(seq_id, camera_id, frame_idx) DO UPDATE SET
                intrinsics = excluded.intrinsics,
                extrinsics = excluded.extrinsics
            """,
            (seq_id, camera_id, frame_idx, intrinsics_blob, extrinsics_blob),
        )
        self.conn.commit()

    def upsert_camera_params_batch(
        self,
        seq_id: str,
        camera_id: str,
        intrinsics_list: Union[np.ndarray, List[np.ndarray]],
        extrinsics_list: Union[np.ndarray, List[np.ndarray]],
    ) -> None:
        """
        Bulk insert camera parameters for all frames of a camera.

        Args:
            seq_id: Sequence identifier
            camera_id: Camera identifier
            intrinsics_list: Array of shape (T, 3, 3) or list of T 3x3 matrices
            extrinsics_list: Array of shape (T, 4, 4) or list of T 4x4 matrices
        """
        self._check_readonly()
        intrinsics_arr = np.asarray(intrinsics_list)
        extrinsics_arr = np.asarray(extrinsics_list)

        if intrinsics_arr.ndim == 2:

            intrinsics_arr = np.broadcast_to(
                intrinsics_arr, (len(extrinsics_arr), 3, 3)
            ).copy()

        num_frames = len(extrinsics_arr)


        batch_data = []
        for frame_idx in range(num_frames):
            intrinsics_blob = _serialize_matrix(intrinsics_arr[frame_idx])
            extrinsics_blob = _serialize_matrix(extrinsics_arr[frame_idx])
            batch_data.append(
                (seq_id, camera_id, frame_idx, intrinsics_blob, extrinsics_blob)
            )


        self.conn.execute(
            "DELETE FROM camera_params WHERE seq_id = ? AND camera_id = ?",
            (seq_id, camera_id),
        )
        self.conn.executemany(
            """
            INSERT INTO camera_params (seq_id, camera_id, frame_idx, intrinsics, extrinsics)
            VALUES (?, ?, ?, ?, ?)
            """,
            batch_data,
        )
        self.conn.commit()

    def get_camera_params(
        self,
        seq_id: str,
        camera_id: Optional[str] = None,
        frame_idx: Optional[int] = None,
        frame_idx_start: Optional[int] = None,
        frame_idx_end: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query camera parameters with optional filters.

        Args:
            seq_id: Sequence identifier
            camera_id: Filter by camera ID (optional)
            frame_idx: Filter by exact frame index (optional)
            frame_idx_start: Filter by frame index >= start (inclusive, optional)
            frame_idx_end: Filter by frame index < end (exclusive, optional)

        Returns:
            List of dicts with camera_id, frame_idx, intrinsics (3x3), extrinsics (4x4)
        """
        query = "SELECT * FROM camera_params WHERE seq_id = ?"
        params: List[Any] = [seq_id]

        if camera_id is not None:
            query += " AND camera_id = ?"
            params.append(camera_id)

        if frame_idx is not None:
            query += " AND frame_idx = ?"
            params.append(frame_idx)

        if frame_idx_start is not None:
            query += " AND frame_idx >= ?"
            params.append(frame_idx_start)

        if frame_idx_end is not None:
            query += " AND frame_idx < ?"
            params.append(frame_idx_end)

        query += " ORDER BY camera_id, frame_idx"

        rows = self.conn.execute(query, params).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "camera_id": row["camera_id"],
                    "frame_idx": row["frame_idx"],
                    "intrinsics": _deserialize_matrix(row["intrinsics"]),
                    "extrinsics": _deserialize_matrix(row["extrinsics"]),
                }
            )

        return result

    def _load_camera_params_for_sequence(
        self, seq_id: str
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all camera parameters for a sequence, grouped by camera_id.

        Returns:
            Dict[camera_id, {"intrinsics": (T,3,3), "extrinsics": (T,4,4)}]
        """
        rows = self.conn.execute(
            """
            SELECT camera_id, frame_idx, intrinsics, extrinsics
            FROM camera_params
            WHERE seq_id = ?
            ORDER BY camera_id, frame_idx
            """,
            (seq_id,),
        ).fetchall()


        camera_data: Dict[str, Dict[str, List]] = {}
        for row in rows:
            cam_id = row["camera_id"]
            if cam_id not in camera_data:
                camera_data[cam_id] = {"intrinsics": [], "extrinsics": []}
            camera_data[cam_id]["intrinsics"].append(
                _deserialize_matrix(row["intrinsics"])
            )
            camera_data[cam_id]["extrinsics"].append(
                _deserialize_matrix(row["extrinsics"])
            )


        result = {}
        for cam_id, data in camera_data.items():
            result[cam_id] = {
                "intrinsics": np.stack(data["intrinsics"]),
                "extrinsics": np.stack(data["extrinsics"]),
            }

        return result


    def upsert_prompt(self, seq_id: str, camera_id: str, prompt: str) -> None:
        """
        Insert or update prompt for a camera.

        Args:
            seq_id: Sequence identifier
            camera_id: Camera identifier
            prompt: Text prompt/caption for this camera view
        """
        self._check_readonly()
        self.conn.execute(
            """
            INSERT INTO camera_prompts (seq_id, camera_id, prompt)
            VALUES (?, ?, ?)
            ON CONFLICT(seq_id, camera_id) DO UPDATE SET
                prompt = excluded.prompt
            """,
            (seq_id, camera_id, prompt),
        )
        self.conn.commit()

    def get_prompt(self, seq_id: str, camera_id: str) -> Optional[str]:
        """
        Get prompt for a specific camera.

        Args:
            seq_id: Sequence identifier
            camera_id: Camera identifier

        Returns:
            Prompt string or None if not found
        """
        row = self.conn.execute(
            "SELECT prompt FROM camera_prompts WHERE seq_id = ? AND camera_id = ?",
            (seq_id, camera_id),
        ).fetchone()
        return row["prompt"] if row else None

    def get_prompts(self, seq_id: str) -> Dict[str, str]:
        """
        Get all prompts for a sequence.

        Args:
            seq_id: Sequence identifier

        Returns:
            Dict mapping camera_id to prompt
        """
        rows = self.conn.execute(
            "SELECT camera_id, prompt FROM camera_prompts WHERE seq_id = ?",
            (seq_id,),
        ).fetchall()
        return {row["camera_id"]: row["prompt"] for row in rows}

    def _load_prompts_for_sequence(self, seq_id: str) -> Dict[str, str]:
        """Load all prompts for a sequence."""
        return self.get_prompts(seq_id)


    def begin_transaction(self) -> None:
        """Begin an explicit transaction for batch operations."""
        self._check_readonly()
        self.conn.execute("BEGIN IMMEDIATE")

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        self.conn.commit()

    def rollback_transaction(self) -> None:
        """Rollback the current transaction."""
        self.conn.rollback()

    def upsert_full_sequence(
        self,
        seq_id: str,
        dataset_type: str,
        num_frames: int,
        height: int,
        width: int,
        paths: Dict[str, str],
        camera_params: Dict[str, Dict[str, np.ndarray]],
        prompts: Optional[Dict[str, str]] = None,
        baseline: Optional[float] = None,
        camera_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Insert a complete sequence with all camera params and prompts in one transaction.

        Args:
            seq_id: Unique sequence identifier
            dataset_type: One of 'monocular', 'stereo', 'multiview'
            num_frames: Number of frames
            height: Frame height
            width: Frame width
            paths: Dictionary of data file paths
            camera_params: Dict[camera_id, {"intrinsics": (T,3,3), "extrinsics": (T,4,4)}]
            prompts: Optional dict mapping camera_id to prompt string
            baseline: Stereo baseline (stereo only)
            camera_ids: List of camera IDs (multiview only)
        """
        self._check_readonly()
        self.begin_transaction()
        try:

            paths_json = json.dumps(paths)
            camera_ids_json = json.dumps(camera_ids) if camera_ids else None

            self.conn.execute(
                """
                INSERT INTO sequences (seq_id, dataset_type, num_frames, height, width,
                                       paths_json, baseline, camera_ids_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(seq_id) DO UPDATE SET
                    dataset_type = excluded.dataset_type,
                    num_frames = excluded.num_frames,
                    height = excluded.height,
                    width = excluded.width,
                    paths_json = excluded.paths_json,
                    baseline = excluded.baseline,
                    camera_ids_json = excluded.camera_ids_json
                """,
                (
                    seq_id,
                    dataset_type,
                    num_frames,
                    height,
                    width,
                    paths_json,
                    baseline,
                    camera_ids_json,
                ),
            )


            self.conn.execute("DELETE FROM camera_params WHERE seq_id = ?", (seq_id,))


            for cam_id, params in camera_params.items():
                intrinsics_arr = np.asarray(params["intrinsics"])
                extrinsics_arr = np.asarray(params["extrinsics"])

                if intrinsics_arr.ndim == 2:
                    intrinsics_arr = np.broadcast_to(
                        intrinsics_arr, (len(extrinsics_arr), 3, 3)
                    ).copy()

                batch_data = []
                for frame_idx in range(len(extrinsics_arr)):
                    batch_data.append(
                        (
                            seq_id,
                            cam_id,
                            frame_idx,
                            _serialize_matrix(intrinsics_arr[frame_idx]),
                            _serialize_matrix(extrinsics_arr[frame_idx]),
                        )
                    )

                self.conn.executemany(
                    """
                    INSERT INTO camera_params (seq_id, camera_id, frame_idx, intrinsics, extrinsics)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    batch_data,
                )


            if prompts:
                self.conn.execute(
                    "DELETE FROM camera_prompts WHERE seq_id = ?", (seq_id,)
                )
                for cam_id, prompt in prompts.items():
                    self.conn.execute(
                        """
                        INSERT INTO camera_prompts (seq_id, camera_id, prompt)
                        VALUES (?, ?, ?)
                        """,
                        (seq_id, cam_id, prompt),
                    )

            self.commit_transaction()
        except Exception:
            self.rollback_transaction()
            raise


    def mark_category_done(self, category: str) -> None:
        """Mark a category as fully processed.

        Used by the CO3D processing pipeline to record that every sequence
        for the given category has been successfully upserted. Idempotent
        via INSERT OR REPLACE so re-runs of already-done categories are
        safe no-ops.
        """
        self._check_readonly()
        self.conn.execute(
            "INSERT OR REPLACE INTO categories (category, done_at) "
            "VALUES (?, CURRENT_TIMESTAMP)",
            (category,),
        )
        self.conn.commit()

    def get_done_categories(self) -> set:
        """Get done categories."""
        try:
            rows = self.conn.execute("SELECT category FROM categories").fetchall()
            return {row["category"] for row in rows}
        except sqlite3.OperationalError:
            return set()

    def count_sequences(self) -> int:
        """Get total number of sequences in database."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM sequences").fetchone()
        return row["cnt"]

    def count_sequences_by_type(self) -> Dict[str, int]:
        """Get count of sequences grouped by dataset type."""
        rows = self.conn.execute(
            "SELECT dataset_type, COUNT(*) as cnt FROM sequences GROUP BY dataset_type"
        ).fetchall()
        return {row["dataset_type"]: row["cnt"] for row in rows}

    def vacuum(self) -> None:
        """Optimize database by reclaiming unused space."""
        self.conn.execute("VACUUM")

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to merge WAL into main database."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
