"""
Frame-Accurate Checkpoint Management and Lossless Video Segment Concatenation.
Ensures pause/resume continuity and batch boundary safety with zero duplicate or lost frames.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import FFMPEG_BIN
from .preprocessor import get_media_info

class CheckpointManager:
    """
    Tracks conversion progress at the exact frame level and orchestrates
    multi-segment encoding for seamless pause and resume.
    """
    def __init__(self, output_path: Path, input_path: Path):
        self.output_path = Path(output_path).resolve()
        self.input_path = Path(input_path).resolve()
        self.checkpoint_file = self.output_path.with_name(f"{self.output_path.name}.checkpoint.json")

    def exists(self) -> bool:
        return self.checkpoint_file.exists()

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Verify input path matches
            if data.get("input_path") == str(self.input_path):
                return data
        except Exception as e:
            print(f"[Checkpoint] Warning: Could not read checkpoint: {e}")
        return None

    def save(
        self,
        processed_frames: int,
        total_frames: int,
        fps_rational: str,
        segments: List[str]
    ):
        data = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "fps_rational": fps_rational,
            "segments": segments
        }
        # Atomic write
        tmp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(self.checkpoint_file)

    def remove(self):
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
            except Exception:
                pass

    def get_segment_path(self, segment_index: int) -> Path:
        stem = self.output_path.stem
        ext = self.output_path.suffix
        return self.output_path.with_name(f"{stem}_part{segment_index:04d}{ext}")

def concatenate_segments(
    segment_paths: List[Path],
    final_output_path: Path,
    cleanup_segments: bool = True
) -> bool:
    """
    Losslessly joins encoded video segments using FFmpeg concat demuxer.
    Preserves exact continuous DTS/PTS without re-encoding or inserting freeze frames.
    """
    if not segment_paths:
        return False
    if len(segment_paths) == 1:
        # Only one segment, simple rename/replace
        if segment_paths[0] != final_output_path:
            segment_paths[0].replace(final_output_path)
        return True

    # Validate all segments exist
    for p in segment_paths:
        if not p.exists() or p.stat().st_size == 0:
            print(f"[Concat Error] Segment missing or empty: {p}")
            return False

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write concat list
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        list_file = Path(tf.name)
        for seg in segment_paths:
            escaped_path = str(seg.resolve()).replace("'", "'\\''")
            tf.write(f"file '{escaped_path}'\n")

    try:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(final_output_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[Concat Error] FFmpeg concat failed: {res.stderr[-400:]}")
            return False

        if cleanup_segments:
            for p in segment_paths:
                try:
                    p.unlink()
                except Exception:
                    pass

        return True
    finally:
        if list_file.exists():
            list_file.unlink()
