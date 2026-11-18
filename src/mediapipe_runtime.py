"""Work around MediaPipe 0.10 Windows resource loading on Unicode paths.

Only model/graph resources are cached, never user images or training data.
The original package and models in the project remain unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import threading

_lock = threading.Lock()


def prepare_mediapipe():
    import mediapipe as mp
    from mediapipe.python import solution_base
    if os.name != "nt" or str(solution_base.__file__).isascii():
        return mp
    with _lock:
        if str(solution_base.__file__).isascii():
            return mp
        original = Path(solution_base.__file__).resolve()
        source_package = original.parents[1]
        cache = Path(os.environ.get("VSL_MEDIAPIPE_CACHE", str(Path(tempfile.gettempdir()) / "vsl-mediapipe"))) / mp.__version__
        if not str(cache).isascii():
            raise RuntimeError("MediaPipe cần đường dẫn không dấu. Đặt VSL_MEDIAPIPE_CACHE tới một thư mục không dấu có quyền ghi.")
        package = cache / "mediapipe"
        marker = cache / ".ready"
        if not marker.exists():
            shutil.copytree(source_package / "modules", package / "modules", dirs_exist_ok=True)
            (package / "python").mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, package / "python" / "solution_base.py")
            marker.write_text(mp.__version__, encoding="ascii")
        # SolutionBase derives the native resource root from this module path.
        # Keep Python imports in place; redirect only its graph/model lookup root.
        solution_base.__file__ = str(package / "python" / "solution_base.py")
    return mp
