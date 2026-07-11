"""
Filesystem scanner that discovers image files from a dataset directory.

Supports two common dataset layouts:

    Flat layout (no categories):
        dataset/
        ├── img001.jpg
        ├── img002.png
        └── img003.webp

    Categorized layout (subfolder = category):
        dataset/
        ├── tops/
        │   ├── img001.jpg
        │   └── img002.jpg
        ├── bottoms/
        │   └── img003.png
        └── dresses/
            └── img004.webp

This module has no Django dependencies and can be imported standalone.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Same set used by the rest of ModaMind (test_pipeline.py, upload validators).
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class ImageRecord:
    """
    A single discovered image and its metadata.

    Attributes:
        absolute_path: Full filesystem path to the image file.
        relative_path: Path relative to the dataset root (for portability).
        category:      Inferred from the parent subfolder name, or
                       'uncategorized' for images sitting directly in the
                       dataset root.
        filename:      Just the filename component (e.g. 'img001.jpg').
    """

    absolute_path: str
    relative_path: str
    category: str
    filename: str


@dataclass
class ScanResult:
    """
    Complete output of a dataset scan.

    Attributes:
        records:         All valid ImageRecord objects discovered.
        categories_found: Sorted list of unique category names.
        skipped_files:    Files that were skipped (unsupported extension, etc.).
        root_dir:         The dataset root that was scanned.
    """

    records: List[ImageRecord] = field(default_factory=list)
    categories_found: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    root_dir: str = ""


class DatasetScanner:
    """
    Walks a dataset directory and produces a list of ImageRecord objects.

    The scanner infers categories from immediate subdirectory names. Files
    in the root of the dataset directory are assigned the category
    'uncategorized'. Nested subdirectories deeper than one level are
    scanned recursively, but the category is always taken from the
    first-level subfolder.

    Usage:
        scanner = DatasetScanner("/path/to/dataset")
        result = scanner.scan()
        for record in result.records:
            print(record.relative_path, record.category)
    """

    UNCATEGORIZED = "uncategorized"

    def __init__(self, dataset_dir: str) -> None:
        """
        Args:
            dataset_dir: Absolute or relative path to the dataset root.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path exists but is not a directory.
        """
        self.dataset_dir = os.path.abspath(dataset_dir)

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: '{self.dataset_dir}'"
            )
        if not os.path.isdir(self.dataset_dir):
            raise NotADirectoryError(
                f"Path is not a directory: '{self.dataset_dir}'"
            )

    def scan(self) -> ScanResult:
        """
        Walk the dataset directory and collect all supported image files.

        Returns:
            A ScanResult containing all discovered ImageRecord objects,
            the unique categories found, and any skipped files.
        """
        result = ScanResult(root_dir=self.dataset_dir)
        categories_set: set[str] = set()

        for dirpath, _dirnames, filenames in os.walk(self.dataset_dir):
            for filename in sorted(filenames):
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, self.dataset_dir)

                if not self._is_supported_image(filename):
                    result.skipped_files.append(rel_path)
                    continue

                category = self._infer_category(dirpath)
                categories_set.add(category)

                result.records.append(
                    ImageRecord(
                        absolute_path=abs_path,
                        relative_path=rel_path,
                        category=category,
                        filename=filename,
                    )
                )

        result.categories_found = sorted(categories_set)

        logger.info(
            "DatasetScanner: found %d image(s) across %d category/ies in '%s'. "
            "Skipped %d unsupported file(s).",
            len(result.records),
            len(result.categories_found),
            self.dataset_dir,
            len(result.skipped_files),
        )

        return result

    def _is_supported_image(self, filename: str) -> bool:
        """Check if a filename has a supported image extension."""
        return filename.lower().endswith(SUPPORTED_EXTENSIONS)

    def _infer_category(self, dirpath: str) -> str:
        """
        Derive the category name from the directory structure.

        If the image is directly inside the dataset root, returns
        'uncategorized'. Otherwise, returns the name of the first-level
        subfolder (the immediate child of dataset_dir that contains
        or is an ancestor of dirpath).
        """
        if os.path.normpath(dirpath) == os.path.normpath(self.dataset_dir):
            return self.UNCATEGORIZED

        # Walk up from dirpath to find the first-level subfolder.
        # Example: dataset_dir = /data, dirpath = /data/tops/summer
        # -> first-level subfolder = "tops"
        rel = os.path.relpath(dirpath, self.dataset_dir)
        first_level = rel.split(os.sep)[0]
        return first_level
