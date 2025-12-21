"""Optional MediaPipe preprocessing and face restore."""

from __future__ import annotations

import sys
from pathlib import Path

from fitme.config import MODELS_DIR, OUTPUT_GEMINI


def preprocess_images(
    person_path: str,
    clothing_path: str,
) -> tuple[str, str, str, str | None]:
    """Face/pose analysis for richer prompts. Returns paths and extra prompt text."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            PoseLandmark,
            PoseLandmarker,
            PoseLandmarkerOptions,
        )
    except ImportError:
        print("Error: mediapipe is required for --preprocess")
        print("Install with: pip install fitme[preprocess]")
        sys.exit(1)

    import cv2

    img = cv2.imread(person_path)
    if img is None:
        print(f"[preprocess] Warning: Could not read {person_path}")
        return person_path, clothing_path, "", None
    h, w = img.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
    )

    face_info = ""
    face_crop_path: str | None = None

    face_model = MODELS_DIR / "face_landmarker.task"
    if face_model.exists():
        print("[preprocess] Running FaceLandmarker...")
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(face_model)),
            num_faces=1,
        )
        with FaceLandmarker.create_from_options(face_options) as landmarker:
            result = landmarker.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            face_x1, face_x2 = min(xs), max(xs)
            face_y1, face_y2 = min(ys), max(ys)
            face_w = face_x2 - face_x1
            face_h = face_y2 - face_y1
            face_cx = (face_x1 + face_x2) / 2
            face_cy = (face_y1 + face_y2) / 2
            face_info = (
                f"Face region (normalized 0-1): "
                f"center=({face_cx:.3f},{face_cy:.3f}), "
                f"bbox=({face_x1:.3f},{face_y1:.3f})"
                f"-({face_x2:.3f},{face_y2:.3f}), "
                f"size={face_w:.3f}x{face_h:.3f}"
            )
            margin = 0.4
            cx1 = max(0, int((face_x1 - face_w * margin) * w))
            cx2 = min(w, int((face_x2 + face_w * margin) * w))
            cy1 = max(0, int((face_y1 - face_h * margin) * h))
            cy2 = min(h, int((face_y2 + face_h * margin) * h))
            face_crop = img[cy1:cy2, cx1:cx2]
            OUTPUT_GEMINI.mkdir(parents=True, exist_ok=True)
            face_crop_path = str(
                OUTPUT_GEMINI / f"_face_crop_{Path(person_path).stem}.png",
            )
            cv2.imwrite(face_crop_path, face_crop)
            print(f"[preprocess] Face crop saved: {face_crop_path}")
        else:
            print("[preprocess] Warning: No face detected")
    else:
        print(f"[preprocess] Warning: {face_model} not found, skipping face")

    pose_info = ""
    pose_model = MODELS_DIR / "pose_landmarker_heavy.task"
    if pose_model.exists():
        print("[preprocess] Running PoseLandmarker...")
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            num_poses=1,
        )
        with PoseLandmarker.create_from_options(pose_options) as landmarker:
            result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            pl = PoseLandmark
            l_sh, r_sh = lm[pl.LEFT_SHOULDER], lm[pl.RIGHT_SHOULDER]
            l_hp, r_hp = lm[pl.LEFT_HIP], lm[pl.RIGHT_HIP]
            shoulder_w = abs(l_sh.x - r_sh.x)
            hip_w = abs(l_hp.x - r_hp.x)
            torso_h = abs((l_sh.y + r_sh.y) / 2 - (l_hp.y + r_hp.y) / 2)
            nose = lm[pl.NOSE]
            pose_info = (
                f"Pose landmarks (normalized 0-1): nose=({nose.x:.3f},{nose.y:.3f}), "
                f"shoulder_width={shoulder_w:.3f}, hip_width={hip_w:.3f}, "
                f"torso_height={torso_h:.3f}"
            )
        else:
            print("[preprocess] Warning: No pose detected")
    else:
        print(f"[preprocess] Warning: {pose_model} not found, skipping pose")

    extra_prompt = ""
    if face_info or pose_info:
        extra_prompt = (
            "\n\nPREPROCESSING DATA (MediaPipe analysis of target person):\n"
        )
        if face_info:
            extra_prompt += f"{face_info}\n"
        if pose_info:
            extra_prompt += f"{pose_info}\n"
        if face_crop_path:
            extra_prompt += (
                "\nImage 3 is a close-up crop of the target person's face. "
                "Reproduce this face EXACTLY.\n"
            )
        extra_prompt += "Use the pose landmarks to maintain the exact body position.\n"

    return person_path, clothing_path, extra_prompt, face_crop_path


def restore_face(original_path: str, generated_path: str) -> str | None:
    """Blend the original face onto the generated try-on image."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
    except ImportError:
        print("[postprocess] mediapipe not available, skipping face restore")
        return None

    import cv2
    import numpy as np

    face_model = MODELS_DIR / "face_landmarker.task"
    if not face_model.exists():
        print("[postprocess] face_landmarker.task not found, skipping")
        return None

    orig_img = cv2.imread(original_path)
    gen_img = cv2.imread(generated_path)
    if orig_img is None or gen_img is None:
        return None

    orig_h, orig_w = orig_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]
    if (gen_w, gen_h) != (orig_w, orig_h):
        gen_img = cv2.resize(
            gen_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4,
        )

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model)),
        num_faces=1,
    )

    def detect_face_bbox(img):
        ih, iw = img.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        )
        with FaceLandmarker.create_from_options(face_options) as landmarker:
            result = landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        fw, fh = max(xs) - min(xs), max(ys) - min(ys)
        mx, my = fw * 0.15, fh * 0.15
        return (
            max(0, int((min(xs) - mx) * iw)),
            max(0, int((min(ys) - my) * ih)),
            min(iw, int((max(xs) + mx) * iw)),
            min(ih, int((max(ys) + my) * ih)),
        )

    orig_bbox = detect_face_bbox(orig_img)
    gen_bbox = detect_face_bbox(gen_img)
    if orig_bbox is None or gen_bbox is None:
        return None

    ox1, oy1, ox2, oy2 = orig_bbox
    gx1, gy1, gx2, gy2 = gen_bbox
    orig_face = orig_img[oy1:oy2, ox1:ox2]
    gen_face_w, gen_face_h = gx2 - gx1, gy2 - gy1
    orig_face_resized = cv2.resize(
        orig_face, (gen_face_w, gen_face_h), interpolation=cv2.INTER_LANCZOS4,
    )

    gen_face_crop = gen_img[gy1:gy2, gx1:gx2]
    orig_lab = cv2.cvtColor(orig_face_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(gen_face_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    for ch in range(3):
        o_mean, o_std = orig_lab[:, :, ch].mean(), orig_lab[:, :, ch].std()
        g_mean, g_std = gen_lab[:, :, ch].mean(), gen_lab[:, :, ch].std()
        if o_std > 0:
            orig_lab[:, :, ch] = (
                (orig_lab[:, :, ch] - o_mean) * (g_std / o_std) + g_mean
            )
    orig_face_corrected = cv2.cvtColor(
        np.clip(orig_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR,
    )

    mask = np.zeros((gen_face_h, gen_face_w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (gen_face_w // 2, gen_face_h // 2),
        (int(gen_face_w * 0.45), int(gen_face_h * 0.45)),
        0,
        0,
        360,
        1.0,
        -1,
    )
    feather = max(max(gen_face_w, gen_face_h) // 3, 3)
    feather += 1 if feather % 2 == 0 else 0
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    mask_3ch = np.stack([mask] * 3, axis=-1)

    result = gen_img.copy()
    gen_region = result[gy1:gy2, gx1:gx2].astype(np.float32)
    blended = orig_face_corrected.astype(np.float32) * mask_3ch + gen_region * (
        1.0 - mask_3ch
    )
    result[gy1:gy2, gx1:gx2] = blended.astype(np.uint8)

    gen_path = Path(generated_path)
    restored_path = gen_path.parent / f"{gen_path.stem}_face_restored{gen_path.suffix}"
    cv2.imwrite(str(restored_path), result)
    print(f"[postprocess] Face restored: {restored_path}")
    return str(restored_path)
