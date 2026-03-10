import shutil
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic"}


def get_image_files(folder: str) -> List[Path]:
    """Get all image files from a folder recursively."""
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(folder_path.rglob(f"*{ext}"))
        files.extend(folder_path.rglob(f"*{ext.upper()}"))

    return sorted(set(files))


def copy_folder(source: str, destination: str, overwrite: bool = False) -> str:
    """Copy source folder to destination, preserving structure."""
    src = Path(source)
    dst = Path(destination)

    if not src.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    if dst.exists() and not overwrite:
        logger.info(f"Destination already exists: {destination}")
        return str(dst)

    if dst.exists() and overwrite:
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    logger.info(f"Copied {source} -> {destination}")
    return str(dst)


def export_selection(image_paths: List[str], output_folder: str) -> int:
    """Export selected images to an output folder."""
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in image_paths:
        src = Path(img_path)
        if src.exists():
            dst = out / src.name
            # Handle name collisions
            if dst.exists():
                stem = src.stem
                suffix = src.suffix
                i = 1
                while dst.exists():
                    dst = out / f"{stem}_{i}{suffix}"
                    i += 1
            shutil.copy2(src, dst)
            count += 1

    return count