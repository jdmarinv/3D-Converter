"""
Unit and integration tests for process pause, resume, and termination controls.
"""
import asyncio
import os
import signal
import subprocess
import sys
import time
import unittest
from gui import server

class TestPauseResume(unittest.TestCase):
    def setUp(self):
        server.conversion_state.update(
            status="idle",
            process=None,
            cancel_requested=False,
            log=""
        )

    def tearDown(self):
        proc = server.conversion_state.get("process")
        if proc and proc.poll() is None:
            server.terminate_conversion(proc)
        server.conversion_state.update(status="idle", process=None)

    def test_pause_and_resume_subprocess(self):
        # Spawn a sleeping loop python subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; [time.sleep(0.1) for _ in range(100)]"],
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        try:
            self.assertIsNone(proc.poll())
            # Pause
            paused = server.pause_conversion(proc)
            self.assertTrue(paused)
            time.sleep(0.1)
            self.assertIsNone(proc.poll())

            # Resume
            resumed = server.resume_conversion(proc)
            self.assertTrue(resumed)
            time.sleep(0.1)
            self.assertIsNone(proc.poll())
        finally:
            server.terminate_conversion(proc)
            proc.wait()
            self.assertIsNotNone(proc.poll())

    def test_terminate_paused_process(self):
        # Verify that terminate_conversion properly resumes and kills a paused process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; [time.sleep(0.1) for _ in range(100)]"],
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        try:
            server.pause_conversion(proc)
            server.terminate_conversion(proc)
            proc.wait(timeout=2.0)
            self.assertIsNotNone(proc.poll())
        except subprocess.TimeoutExpired:
            proc.kill()
            self.fail("Paused process did not terminate promptly.")

    def test_pause_resume_api_endpoints(self):
        # When idle: pausing should return 400
        res = asyncio.run(server.pause_active_conversion())
        self.assertEqual(res.status_code, 400)

        # When idle: resuming should return 400
        res = asyncio.run(server.resume_active_conversion())
        self.assertEqual(res.status_code, 400)

        # Mock active process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; [time.sleep(0.1) for _ in range(100)]"],
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        server.conversion_state.update(status="running", process=proc)
        try:
            res_pause = asyncio.run(server.pause_active_conversion())
            self.assertEqual(res_pause, {"status": "paused"})
            self.assertEqual(server.conversion_state["status"], "paused")

            res_resume = asyncio.run(server.resume_active_conversion())
            self.assertEqual(res_resume, {"status": "running"})
            self.assertEqual(server.conversion_state["status"], "running")
        finally:
            server.terminate_conversion(proc)
            proc.wait()

if __name__ == "__main__":
    unittest.main()
