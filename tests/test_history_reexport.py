"""
Unit tests for Conversion History and Fast Re-export hub.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from gui import server

class TestHistoryReexport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.history_file = self.tmp_path / "history.json"
        self._orig_history_file = server.HISTORY_FILE
        server.HISTORY_FILE = self.history_file

    def tearDown(self):
        server.HISTORY_FILE = self._orig_history_file
        self.tmp.cleanup()

    def test_load_empty_history(self):
        self.assertEqual(server.load_history(), [])

    def test_add_and_load_history(self):
        # Create dummy output and depth files
        out_file = self.tmp_path / "test_3d_hsbs.mp4"
        out_file.write_bytes(b"dummy video content")
        depth_file = self.tmp_path / "test_depth.mp4"
        depth_file.write_bytes(b"dummy depth content")

        entry = {
            "id": "conv_12345_test",
            "timestamp": 1234567890,
            "date_str": "2026-09-15 10:00",
            "input_path": "/path/to/source.mp4",
            "output_path": str(out_file),
            "depth_path": str(depth_file),
            "has_depth": True,
            "format": "hsbs",
            "strength_3d": 5.0,
            "style_3d": "natural",
            "depth_profile": "balanced",
            "depth_stride": 1,
            "total_frames": 100,
            "resolution": "1920x1080",
            "duration": 4.5
        }

        server.add_history_entry(entry)
        loaded = server.load_history()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], "conv_12345_test")
        self.assertTrue(loaded[0]["output_exists"])
        self.assertTrue(loaded[0]["depth_exists"])

    def test_delete_history_entry(self):
        entry1 = {"id": "item_1", "output_path": "/tmp/out1.mp4"}
        entry2 = {"id": "item_2", "output_path": "/tmp/out2.mp4"}
        server.add_history_entry(entry1)
        server.add_history_entry(entry2)

        self.assertEqual(len(server.load_history()), 2)
        deleted = server.delete_history_entry("item_1")
        self.assertTrue(deleted)
        self.assertEqual(len(server.load_history()), 1)
        self.assertEqual(server.load_history()[0]["id"], "item_2")

    def test_clear_history(self):
        entry = {"id": "item_1", "output_path": "/tmp/out1.mp4"}
        server.add_history_entry(entry)
        self.assertTrue(server.HISTORY_FILE.exists())
        server.clear_history()
        self.assertFalse(server.HISTORY_FILE.exists())
        self.assertEqual(server.load_history(), [])

    def test_api_history_endpoints(self):
        # GET /api/history
        res = asyncio.run(server.get_conversion_history())
        self.assertIn("history", res)
        self.assertEqual(len(res["history"]), 0)

        # Add entry
        server.add_history_entry({"id": "item_abc", "output_path": "/tmp/abc.mp4"})
        res = asyncio.run(server.get_conversion_history())
        self.assertEqual(len(res["history"]), 1)

        # DELETE /api/history/item_abc
        del_res = asyncio.run(server.remove_history_entry("item_abc"))
        self.assertEqual(del_res, {"status": "deleted"})

        # DELETE non-existent
        del_res_404 = asyncio.run(server.remove_history_entry("item_abc"))
        self.assertEqual(del_res_404.status_code, 404)

        # POST /api/history/clear
        clear_res = asyncio.run(server.clear_all_history())
        self.assertEqual(clear_res, {"status": "cleared"})

    def test_api_open_path(self):
        test_file = self.tmp_path / "dummy.txt"
        test_file.write_text("hello")

        # Mock request with JSON
        class FakeRequest:
            def __init__(self, data):
                self._data = data
            async def json(self):
                return self._data

        req = FakeRequest({"path": str(test_file)})
        with patch("gui.server.open_in_system") as mock_open:
            res = asyncio.run(server.open_specific_path(req))
            self.assertEqual(res, {"status": "ok"})
            mock_open.assert_called_once()

if __name__ == "__main__":
    unittest.main()
