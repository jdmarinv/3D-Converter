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

    @unittest.skipIf(os.name == "nt", "POSIX process-group behavior")
    def test_terminate_kills_child_after_parent_exits_on_sigterm(self):
        script = (
            "import signal,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']); "
            "print(child.pid,flush=True); time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE,
            text=True, start_new_session=True
        )
        child_pid = int(proc.stdout.readline().strip())
        proc.stdout.close()
        server.terminate_conversion(proc)
        proc.wait(timeout=2.0)
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            os.kill(child_pid, signal.SIGKILL)
            self.fail("Child encoder survived conversion cancellation")

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
