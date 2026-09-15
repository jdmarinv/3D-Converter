"""Optional motion-compensated repair of existing 2D/SBS files; never part of synthesis."""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from fractions import Fraction
from .config import FFMPEG_BIN
from .preprocessor import get_media_info


def repair_defective_cadence(input_path, output_path, mode="interpolate",
                             diff_threshold=1.5, target_fps_rational=None,
                             layout="2d", start_time=0.0, duration=0.0):
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output:
        raise ValueError("Choose a new output file; original file is preserved.")
    if mode != "interpolate":
        raise ValueError("Decimation without motion reconstruction changes timing and is unsupported.")
    if layout not in ("2d", "sbs", "hsbs"):
        raise ValueError("Select 2D, Full SBS or Half SBS input.")
    info = get_media_info(source)
    fps = target_fps_rational or info['fps_rational']
    if Fraction(fps) != Fraction(info['fps_rational']):
        raise ValueError("Repair preserves the source frame rate.")
    seconds = min(duration or info['duration'], info['duration'] - start_time)
    if seconds <= 0 or start_time < 0:
        raise ValueError("Selected time range is outside the video.")
    width, height = info['width'], info['height']
    if width % 2 or height % 2 or (layout != '2d' and width % 4):
        raise ValueError("SBS requires even eye dimensions for H.264.")
    # Shared decimation BEFORE splitting prevents different frame selections in each eye.
    # Preserve PTS: resetting to N/fps after decimation would shorten the video.
    base = f'[0:v]fps=fps={fps}:round=near,setpts=PTS-STARTPTS,mpdecimate=max=1' if info['is_vfr'] else '[0:v]setpts=PTS-STARTPTS,mpdecimate=max=1'
    interp = f'minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bilat:me=epzs:vsbmc=1:scd_threshold=10'
    if layout == '2d':
        filters = f'{base},{interp}[motion]'
    else:
        half = width // 2
        filters = (f'{base},split=2[l][r];[l]crop={half}:{height}:0:0,{interp}[li];'
                   f'[r]crop={half}:{height}:{half}:0,{interp}[ri];[li][ri]hstack[motion]')
    filters += f';[motion]tpad=stop_mode=clone:stop_duration=0.2,trim=duration={seconds},setsar=1[v]'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cadence-', dir=output.parent) as tmp:
        temp = Path(tmp) / 'repaired.mp4'
        cmd = [FFMPEG_BIN, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
               '-ss', str(start_time), '-t', str(seconds + 1), '-i', str(source),
               '-filter_complex', filters, '-map', '[v]', '-map', '0:a?',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-ac', '2', '-b:a', '192k', '-t', str(seconds),
               '-movflags', '+faststart', '-progress', 'pipe:1', str(temp)]
        total = round(seconds * float(Fraction(fps)))
        started = time.monotonic()
        with tempfile.TemporaryFile(mode='w+') as errors:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors, text=True)
            try:
                for line in proc.stdout:
                    if line.startswith('frame='):
                        current = min(int(line.split('=')[1]), total)
                        rate = current / max(time.monotonic() - started, 0.01)
                        remaining = int((total-current) / max(rate, 0.01))
                        print('PORTAL_PROGRESS ' + json.dumps({'current_frame': current,
                            'total_frames': total, 'fps': round(rate, 2), 'eta': f'{remaining//60:02d}:{remaining%60:02d}', 'percent': 100 * current / max(total, 1),
                            'log': f'Interpolating motion: {current}/{total} frames'}), flush=True)
                if proc.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read()[-2000:] or 'FFmpeg repair failed')
            finally:
                if proc.stdout: proc.stdout.close()
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            result = get_media_info(temp)
            if abs(result['duration'] - seconds) > max(0.1, 2 / info['fps']):
                raise RuntimeError('Repair duration does not match selected input range')
            subprocess.run([FFMPEG_BIN, '-v', 'error', '-xerror', '-i', str(temp),
                            '-map', '0', '-c', 'copy', '-f', 'null', '-'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            temp.replace(output)
    print(f'✓ Motion repair saved: {output}', flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--layout', choices=['2d', 'sbs', 'hsbs'], default='2d')
    p.add_argument('--start-time', type=float, default=0)
    p.add_argument('--duration', type=float, default=0)
    a = p.parse_args()
    repair_defective_cadence(a.input, a.output, layout=a.layout,
                            start_time=a.start_time, duration=a.duration)

if __name__ == '__main__':
    main()
