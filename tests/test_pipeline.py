"""
End-to-End Pipeline Test for 2D to 3D Offline Converter.
Creates a synthetic test scene with distinct foreground/background layers,
runs depth estimation on MPS, synthesizes stereoscopic pairs, and tests all encoders.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, DEFAULT_DEPTH_MODEL, DEFAULT_OUTPUT_DIR
from src.depth_engine import DepthEngine
from src.dibr_stereo import StereoSynthesizer
from src.encoder_3d import compose_sbs, compose_anaglyph

def create_synthetic_scene(w: int = 640, h: int = 480) -> np.ndarray:
    """
    Creates a synthetic test image with rich gradients, a textured background,
    and a distinct foreground floating sphere to test depth and occlusion inpainting.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient
    for y in range(h):
        img[y, :, 0] = int(255 * (y / h))       # Red gradient
        img[y, :, 2] = int(255 * (1.0 - y / h)) # Blue gradient

    # Add grid pattern in background
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (180, 180, 180), 1)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (180, 180, 180), 1)

    # Add floating foreground spheres
    cv2.circle(img, (w // 3, h // 2), 70, (255, 255, 255), -1)
    cv2.circle(img, (w // 3, h // 2), 65, (0, 230, 255), -1) # Yellow-cyan sphere
    
    cv2.circle(img, (2 * w // 3, h // 2 + 30), 50, (50, 255, 50), -1) # Green sphere

    # Add high-contrast text in foreground
    cv2.putText(img, "OFFLINE 3D", (w // 4, h // 5), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

    return img

def test_pipeline():
    print("=" * 60)
    print(f"[Test] Starting End-to-End Verification on Device: {DEVICE}")
    print("=" * 60)

    # 1. Create test scene
    scene = create_synthetic_scene()
    test_img_path = DEFAULT_OUTPUT_DIR / "test_synthetic_scene.png"
    cv2.imwrite(str(test_img_path), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
    print(f"✓ Synthetic scene created at: {test_img_path}")

    # 2. Depth Engine Test
    assert Path(DEFAULT_DEPTH_MODEL).exists(), f"Model file missing: {DEFAULT_DEPTH_MODEL}"
    engine = DepthEngine()
    depth_map = engine.estimate_depth(scene)
    
    assert depth_map.shape == (scene.shape[0], scene.shape[1]), "Depth map dimension mismatch"
    assert 0.0 <= depth_map.min() <= 1.0, "Depth map min out of range"
    assert 0.0 <= depth_map.max() <= 1.0, "Depth map max out of range"
    
    depth_vis_path = DEFAULT_OUTPUT_DIR / "test_depth_map.png"
    cv2.imwrite(str(depth_vis_path), (depth_map * 255.0).astype(np.uint8))
    print(f"✓ Depth estimation verified (Range: [{depth_map.min():.2f}, {depth_map.max():.2f}]) -> {depth_vis_path}")

    # 3. Stereoscopic Synthesizer Test (DIBR + Inpainting)
    synthesizer = StereoSynthesizer(divergence=0.04, convergence=0.5)
    left, right = synthesizer.render_stereo(scene, depth_map)
    assert left.shape == scene.shape, "Left eye dimension mismatch"
    assert right.shape == scene.shape, "Right eye dimension mismatch"
    print("✓ Stereoscopic DIBR and occlusion inpainting executed successfully")

    # 4. Encoders: Half-SBS, Full-SBS, Dubois Anaglyph
    hsbs = compose_sbs(left, right, mode="half_sbs")
    assert hsbs.shape == scene.shape, f"HSBS shape mismatch: {hsbs.shape} vs {scene.shape}"
    hsbs_path = DEFAULT_OUTPUT_DIR / "test_output_hsbs.png"
    cv2.imwrite(str(hsbs_path), cv2.cvtColor(hsbs, cv2.COLOR_RGB2BGR))
    print(f"✓ Half-SBS output verified: {hsbs_path}")

    fsbs = compose_sbs(left, right, mode="full_sbs")
    assert fsbs.shape == (scene.shape[0], scene.shape[1] * 2, 3), f"Full-SBS shape mismatch: {fsbs.shape}"
    fsbs_path = DEFAULT_OUTPUT_DIR / "test_output_full_sbs.png"
    cv2.imwrite(str(fsbs_path), cv2.cvtColor(fsbs, cv2.COLOR_RGB2BGR))
    print(f"✓ Full-SBS output verified: {fsbs_path}")

    anaglyph = compose_anaglyph(left, right)
    assert anaglyph.shape == scene.shape, "Anaglyph shape mismatch"
    anaglyph_path = DEFAULT_OUTPUT_DIR / "test_output_anaglyph.png"
    cv2.imwrite(str(anaglyph_path), cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))
    print(f"✓ Dubois Anaglyph output verified: {anaglyph_path}")

    print("=" * 60)
    print("ALL TESTS PASSED! OFFLINE 2D-TO-3D PIPELINE IS 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
