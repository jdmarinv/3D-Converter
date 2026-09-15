"""Exercise the real portal worker, encoder and repair; no AI weights required."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from gui import server
from src.encoder_3d import VideoStreamWriter
from src.preprocessor import get_media_info, read_video_frames
from src.cadence_analyzer import analyze_visual_cadence

class PortalIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.mp4'
        writer = VideoStreamWriter(self.source, 160, 96, 24, codec='libx264')
        for i in range(48):
            frame=np.zeros((96,160,3),dtype=np.uint8)
            frame[:, (i*3)%130:(i*3)%130+25]=255
            writer.write_frame(frame)
        writer.close()
        server.conversion_state.update(status='idle',process=None,cadence=None,cancel_requested=False)
    def tearDown(self): self.tmp.cleanup()
    def run_worker(self, **kwargs):
        req=server.ConversionRequest(input_path=str(self.source),output_path=str(self.root/'out.mp4'),**kwargs)
        server.conversion_state.update(output_file=req.output_path,status='running')
        server.run_conversion_worker(req)
        self.assertEqual(server.conversion_state['status'],'completed',server.conversion_state['log'])
        return Path(req.output_path)
    def test_portal_defaults_to_symmetric_stereo(self):
        req=server.ConversionRequest(input_path=str(self.source))
        self.assertEqual(req.render_mode,'both')
    def test_repair_real_worker(self):
        out=self.run_worker(operation='repair',repair_layout='2d',duration=1)
        self.assertEqual(get_media_info(out)['nb_frames'],24)
    def test_convert_trim_and_diagnostic_real_worker(self):
        out=self.run_worker(custom_depth=str(self.source),auto_crop=False,save_depth=False,
                            start_time=0.5,duration=1,profile=None)
        self.assertEqual(get_media_info(out)['nb_frames'],24)
        self.assertEqual(server.conversion_state['cadence']['verdict'],'PASS')
    def test_preserve_existing_output(self):
        req=server.ConversionRequest(input_path=str(self.source),output_path=str(self.source))
        response=asyncio.run(server.start_conversion(req))
        self.assertEqual(response.status_code,400)
    def test_allow_conversion_when_output_file_exists(self):
        existing_out=self.root/'existing_out.mp4'
        existing_out.write_bytes(b'dummy')
        req=server.ConversionRequest(input_path=str(self.source),output_path=str(existing_out))
        with patch.object(server.threading,"Thread") as mock_thread:
            mock_thread.return_value.start=lambda:None
            res=asyncio.run(server.start_conversion(req))
            self.assertEqual(res.get("status"),"started")
            server.conversion_state['status']='idle'
    def test_reject_second_job(self):
        server.conversion_state['status']='running'
        response=asyncio.run(server.start_conversion(server.ConversionRequest(input_path=str(self.source))))
        self.assertEqual(response.status_code,409)
    def test_frame_preview_matches_requested_frame_and_status_is_json_safe(self):
        server.conversion_state.update(
            status='running', input_file=str(self.source), source_fps=24,
            start_time=0.0, current_frame=20, total_frames=48,
            preview_frame=-1, preview_jpeg=None)
        response=asyncio.run(server.frame_preview(frame=12))
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.media_type,'image/jpeg')
        self.assertTrue(response.body.startswith(b'\xff\xd8'))
        self.assertEqual(server.conversion_state['preview_frame'],12)
        status=asyncio.run(server.get_status())
        self.assertNotIn('preview_jpeg',status)
        json.dumps(status)
    def test_selected_video_preview_replaces_previous_completed_media(self):
        server.selected_media.update(path=str(self.source),fps=24)
        server.conversion_state.update(
            status='completed',input_file=str(self.root/'old.jpg'),source_fps=0,
            total_frames=0,preview_frame=-1,preview_jpeg=None)
        response=asyncio.run(server.frame_preview(frame=0))
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.body.startswith(b'\xff\xd8'))
    def test_ffmpeg_error_raises(self):
        writer=VideoStreamWriter(self.root/'bad.mp4',160,96,24,codec='not_an_encoder')
        with self.assertRaises(RuntimeError): writer.close()
    def test_cadence_detects_periodic_duplicates(self):
        out=self.root/'bad_cadence.mp4'; writer=VideoStreamWriter(out,160,96,24,codec='libx264')
        prev=None
        for i,f in read_video_frames(self.source):
            emit=prev if i and i%3==0 else f
            writer.write_frame(emit);prev=emit
        writer.close()
        report=analyze_visual_cadence(self.source,out,fmt='2d')
        self.assertTrue(report['stutter_detected'])
        self.assertEqual(report['detected_period'],3)

if __name__=='__main__': unittest.main()
