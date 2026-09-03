#!/usr/bin/env python3
"""
Face-protected virtual webcam.

Tracks a face across frames, reduces fine facial detail while preserving broad
appearance and expression, and publishes the result as a virtual camera.

Examples:
  python face_protected_webcam_tracked.py --auto-camera --preview
  python face_protected_webcam_tracked.py --camera-name "Logitech" --preview
  python face_protected_webcam_tracked.py --face-detail 40 --auto-camera --preview
"""

import argparse
import math
import os
import sys
import time
import random
import secrets

import cv2
import numpy as np
import pyvirtualcam

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None


class FaceProtector:
    """
    Persistent single-face protection.

    Uses Haar face detection to acquire/reacquire the face, then pyramidal
    Lucas-Kanade optical flow plus an affine transform to follow it between
    detector hits. This avoids the one-frame-on / one-frame-off behavior of
    detector-only protection.
    """

    def __init__(
        self,
        detail=32,
        padding=0.06,
        redetect_every=5,
        hold_frames=60,
        randomize=True,
        seed=None,
    ):
        self.base_detail = max(8, int(detail))
        self.padding = max(0.0, float(padding))
        self.redetect_every = max(1, int(redetect_every))
        self.hold_frames = max(1, int(hold_frames))

        # Choose one stable protection profile per launch. Keeping it fixed
        # within a session avoids distracting frame-to-frame flicker while
        # preventing every session from using the exact same transformation.
        if seed is None:
            seed = secrets.randbits(64)
        self.seed = int(seed)
        rng = random.Random(self.seed)

        if randomize:
            self.detail = max(16, self.base_detail + rng.randint(-6, 6))
            self.quant_levels = rng.randint(18, 30)
            self.quant_offset = (
                rng.randint(0, 7),
                rng.randint(0, 7),
                rng.randint(0, 7),
            )
            self.hue_shift = rng.randint(-5, 5)
            self.saturation_scale = rng.uniform(0.90, 1.10)
            self.value_scale = rng.uniform(0.96, 1.04)
            self.channel_gains = (
                rng.uniform(0.97, 1.03),
                rng.uniform(0.97, 1.03),
                rng.uniform(0.97, 1.03),
            )
            self.pixel_aspect = rng.uniform(0.94, 1.06)
            self.grid_phase = (rng.randint(0, 4), rng.randint(0, 4))
            self.feather_divisor = rng.randint(20, 30)
        else:
            self.detail = self.base_detail
            self.quant_levels = 24
            self.quant_offset = (0, 0, 0)
            self.hue_shift = 0
            self.saturation_scale = 1.0
            self.value_scale = 1.0
            self.channel_gains = (1.0, 1.0, 1.0)
            self.pixel_aspect = 1.0
            self.grid_phase = (0, 0)
            self.feather_divisor = 24

        frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self.frontal = cv2.CascadeClassifier(frontal_path)
        self.profile = cv2.CascadeClassifier(profile_path)

        if self.frontal.empty():
            raise RuntimeError(f"Could not load face detector: {frontal_path}")

        self.box = None
        self.prev_box = None
        self.velocity = np.zeros(4, dtype=np.float32)
        self.points = None
        self.prev_gray = None
        self.frame_no = 0
        self.lost_frames = 0
        self.ever_detected = False

    def profile_summary(self):
        return (
            f"seed={self.seed} detail={self.detail} "
            f"quant_levels={self.quant_levels} hue={self.hue_shift:+d} "
            f"sat={self.saturation_scale:.3f} value={self.value_scale:.3f} "
            f"grid_phase={self.grid_phase}"
        )

    @staticmethod
    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0

    @staticmethod
    def _clamp_box(box, width, height):
        x, y, w, h = [int(round(v)) for v in box]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        return (x, y, w, h)

    def _expanded(self, box, width, height, extra=0.0):
        x, y, w, h = box
        p = self.padding + max(0.0, extra)
        px = int(w * p)
        py = int(h * p)
        return self._clamp_box(
            (x - px, y - py, w + 2 * px, h + 2 * py),
            width,
            height,
        )

    def _detect(self, gray):
        """
        Detect at multiple image scales.

        Downscaled passes help reacquire very large close-up faces that can be
        harder for the Haar cascade to detect at full resolution.
        """
        h, w = gray.shape[:2]
        candidates = []

        for scale in (1.0, 0.75, 0.5):
            if scale == 1.0:
                work = gray
            else:
                work = cv2.resize(
                    gray,
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            eq = cv2.equalizeHist(work)

            min_side = max(36, int(64 * scale))

            frontal = self.frontal.detectMultiScale(
                eq,
                scaleFactor=1.07,
                minNeighbors=4,
                minSize=(min_side, min_side),
            )

            for x, y, fw, fh in frontal:
                inv = 1.0 / scale
                candidates.append(
                    (
                        int(x * inv),
                        int(y * inv),
                        int(fw * inv),
                        int(fh * inv),
                    )
                )

            if not self.profile.empty():
                prof = self.profile.detectMultiScale(
                    eq,
                    scaleFactor=1.07,
                    minNeighbors=4,
                    minSize=(min_side, min_side),
                )
                for x, y, fw, fh in prof:
                    inv = 1.0 / scale
                    candidates.append(
                        (
                            int(x * inv),
                            int(y * inv),
                            int(fw * inv),
                            int(fh * inv),
                        )
                    )

                flipped = cv2.flip(eq, 1)
                prof_flip = self.profile.detectMultiScale(
                    flipped,
                    scaleFactor=1.07,
                    minNeighbors=4,
                    minSize=(min_side, min_side),
                )
                for x, y, fw, fh in prof_flip:
                    inv = 1.0 / scale
                    fx = work.shape[1] - int(x) - int(fw)
                    candidates.append(
                        (
                            int(fx * inv),
                            int(y * inv),
                            int(fw * inv),
                            int(fh * inv),
                        )
                    )

        if not candidates:
            return None

        # Remove obviously invalid boxes.
        valid = []
        for b in candidates:
            x, y, bw, bh = self._clamp_box(b, w, h)
            if bw >= 50 and bh >= 50:
                valid.append((x, y, bw, bh))

        if not valid:
            return None

        if self.box is not None:
            scored = []
            cx0 = self.box[0] + self.box[2] / 2
            cy0 = self.box[1] + self.box[3] / 2
            diag = max(1.0, (w * w + h * h) ** 0.5)

            for b in valid:
                cx = b[0] + b[2] / 2
                cy = b[1] + b[3] / 2
                dist = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5 / diag

                # Prefer overlap, nearby centers, and plausible scale changes.
                old_area = max(1, self.box[2] * self.box[3])
                new_area = max(1, b[2] * b[3])
                scale_ratio = new_area / old_area
                scale_penalty = abs(math.log(max(scale_ratio, 1e-6)))

                score = (
                    3.0 * self._iou(self.box, b)
                    - 0.8 * dist
                    - 0.2 * scale_penalty
                    + 0.25 * new_area / (w * h)
                )
                scored.append((score, b))

            return max(scored, key=lambda item: item[0])[1]

        return max(valid, key=lambda b: b[2] * b[3])


    def _seed_points(self, gray, box):
        x, y, w, h = box

        # Track features mostly inside the central face area.
        inset_x = int(w * 0.08)
        inset_y = int(h * 0.08)
        x1 = max(0, x + inset_x)
        y1 = max(0, y + inset_y)
        x2 = min(gray.shape[1], x + w - inset_x)
        y2 = min(gray.shape[0], y + h - inset_y)

        mask = np.zeros_like(gray)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=7,
        )
        return pts

    def _track(self, gray):
        if self.prev_gray is None or self.points is None or len(self.points) < 6:
            return False

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.points,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if new_pts is None or status is None:
            return False

        good_old = self.points[status.reshape(-1) == 1].reshape(-1, 2)
        good_new = new_pts[status.reshape(-1) == 1].reshape(-1, 2)

        if len(good_old) < 6:
            return False

        matrix, inliers = cv2.estimateAffinePartial2D(
            good_old,
            good_new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if matrix is None:
            return False

        x, y, w, h = self.box
        corners = np.float32(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        ).reshape(-1, 1, 2)

        moved = cv2.transform(corners, matrix).reshape(-1, 2)
        x1, y1 = moved.min(axis=0)
        x2, y2 = moved.max(axis=0)

        tracked = self._clamp_box(
            (x1, y1, x2 - x1, y2 - y1),
            gray.shape[1],
            gray.shape[0],
        )

        # Smooth small tracker noise without creating noticeable lag.
        old = np.array(self.box, dtype=np.float32)
        new = np.array(tracked, dtype=np.float32)
        smoothed = 0.18 * old + 0.82 * new

        self.prev_box = tuple(self.box)
        self.box = self._clamp_box(
            smoothed,
            gray.shape[1],
            gray.shape[0],
        )

        delta = np.array(self.box, dtype=np.float32) - old
        self.velocity = 0.55 * self.velocity + 0.45 * delta

        self.points = good_new.reshape(-1, 1, 2)
        return True

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        tracked_ok = False
        if self.box is not None:
            tracked_ok = self._track(gray)

        # Re-detect often, and every frame whenever tracking is uncertain.
        should_detect = (
            self.box is None
            or not tracked_ok
            or self.frame_no % self.redetect_every == 0
            or self.lost_frames > 0
        )

        detection = self._detect(gray) if should_detect else None

        if detection is not None:
            old_box = np.array(self.box, dtype=np.float32) if self.box is not None else None

            if self.box is None or not tracked_ok:
                self.prev_box = self.box
                self.box = detection
            else:
                old = np.array(self.box, dtype=np.float32)
                det = np.array(detection, dtype=np.float32)

                # Follow large detector scale changes more aggressively. This is
                # important when the user rapidly moves toward the camera.
                old_area = max(1.0, old[2] * old[3])
                det_area = max(1.0, det[2] * det[3])
                ratio = det_area / old_area
                det_weight = 0.75 if ratio > 1.35 or ratio < 0.74 else 0.50

                merged = (1.0 - det_weight) * old + det_weight * det
                self.prev_box = tuple(self.box)
                self.box = self._clamp_box(merged, w, h)

            if old_box is not None:
                delta = np.array(self.box, dtype=np.float32) - old_box
                self.velocity = 0.45 * self.velocity + 0.55 * delta
            else:
                self.velocity[:] = 0

            self.points = self._seed_points(gray, self.box)
            self.lost_frames = 0
            self.ever_detected = True

        elif tracked_ok:
            self.lost_frames = 0
            self.ever_detected = True

        elif self.box is not None:
            # Fail closed. Never immediately expose a clean face just because
            # the detector/tracker missed during fast motion.
            self.lost_frames += 1

            b = np.array(self.box, dtype=np.float32)

            # Extrapolate the last observed translation and scale.
            predicted = b + self.velocity

            # Aggressively enlarge the protected region while lost. Rapid
            # forward motion usually means the face is getting larger.
            grow = min(1.75, 1.08 ** self.lost_frames)
            cx = predicted[0] + predicted[2] / 2.0
            cy = predicted[1] + predicted[3] / 2.0
            nw = predicted[2] * grow
            nh = predicted[3] * grow

            predicted = np.array(
                [cx - nw / 2.0, cy - nh / 2.0, nw, nh],
                dtype=np.float32,
            )
            self.box = self._clamp_box(predicted, w, h)

            # Keep trying to track inside the expanded region.
            self.points = self._seed_points(gray, self.box)

            # Decay motion prediction gradually rather than snapping to zero.
            self.velocity *= 0.82

            # Privacy-oriented fallback. Once a face has been seen, do not
            # fully clear protection after a short timeout. After prolonged
            # loss, expand to most of the frame instead.
            if self.lost_frames > self.hold_frames:
                margin_x = int(w * 0.08)
                margin_y = int(h * 0.05)
                self.box = (
                    margin_x,
                    margin_y,
                    max(1, w - 2 * margin_x),
                    max(1, h - 2 * margin_y),
                )
                self.points = self._seed_points(gray, self.box)

        self.prev_gray = gray
        self.frame_no += 1

        if self.box is None:
            return None

        # Add extra margin during uncertainty.
        extra = min(0.32, self.lost_frames * 0.018)
        return self._expanded(self.box, w, h, extra=extra)


    def protect(self, frame):
        box = self.update(frame)
        if box is None:
            return frame

        x, y, w, h = box
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return frame

        # Preserve head shape, pose, expression, and broad identity cues while
        # removing much of the fine facial texture. The exact transform is
        # randomized once at launch and remains stable for the whole session.
        phase_x, phase_y = self.grid_phase

        # A tiny fixed crop phase changes where the pixel grid lands without
        # causing frame-to-frame shimmer.
        x1 = min(max(0, phase_x), max(0, w - 2))
        y1 = min(max(0, phase_y), max(0, h - 2))
        work = roi[y1:, x1:]
        if work.size == 0:
            work = roi

        target_w = min(self.detail, max(8, work.shape[1]))
        target_h = max(
            8,
            int(round(
                target_w
                * work.shape[0]
                / max(work.shape[1], 1)
                * self.pixel_aspect
            )),
        )

        tiny = cv2.resize(
            work,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

        # Mild fixed-per-session color remapping.
        hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + self.hue_shift, 180.0)
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1] * self.saturation_scale, 0, 255
        )
        hsv[:, :, 2] = np.clip(
            hsv[:, :, 2] * self.value_scale, 0, 255
        )
        tiny = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Slight fixed channel gains vary the color transform further.
        gains = np.array(self.channel_gains, dtype=np.float32).reshape(1, 1, 3)
        tiny = np.clip(
            tiny.astype(np.float32) * gains, 0, 255
        ).astype(np.uint8)

        # Session-specific quantization levels and bin offsets.
        step = max(1, 256 // self.quant_levels)
        arr = tiny.astype(np.int16)
        for c in range(3):
            offset = self.quant_offset[c] % step
            arr[:, :, c] = (
                ((arr[:, :, c] + offset) // step) * step
                + step // 2
                - offset
            )
        tiny = np.clip(arr, 0, 255).astype(np.uint8)

        protected_work = cv2.resize(
            tiny,
            (work.shape[1], work.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        protected = roi.copy()
        protected[y1:, x1:] = protected_work

        # Feather only the outer edge to avoid a harsh rectangular cutout.
        feather = max(3, min(w, h) // self.feather_divisor)
        mask = np.zeros((h, w), dtype=np.uint8)
        if w > 2 * feather and h > 2 * feather:
            cv2.rectangle(
                mask,
                (feather, feather),
                (w - feather - 1, h - feather - 1),
                255,
                -1,
            )
        else:
            mask[:] = 255
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]

        out = frame.copy()
        original = out[y:y + h, x:x + w].astype(np.float32)
        mixed = original * (1.0 - alpha) + protected.astype(np.float32) * alpha
        out[y:y + h, x:x + w] = np.clip(mixed, 0, 255).astype(np.uint8)

        return out


def make_test_pattern(width, height, frame_no):
    """Animated test image; useful for testing the virtual-camera path alone."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    bars = [
        (255, 255, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
        (255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 0, 0),
    ]
    bar_w = max(1, width // len(bars))
    for i, color in enumerate(bars):
        x1 = i * bar_w
        x2 = width if i == len(bars) - 1 else (i + 1) * bar_w
        frame[:, x1:x2] = color

    # Moving marker proves the stream is live rather than a frozen frame.
    x = int((frame_no * 9) % max(1, width))
    cv2.circle(frame, (x, height // 2), 24, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.putText(frame, "VIRTUAL CAMERA TEST", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, "VIRTUAL CAMERA TEST", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame



def list_named_cameras():
    """Enumerate Windows cameras by real device name/backend."""
    if enumerate_cameras is None:
        print("Camera-name enumeration requires: python -m pip install cv2-enumerate-cameras")
        return []

    cameras = []
    seen = set()
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        try:
            items = enumerate_cameras(backend)
        except Exception as exc:
            print(f"Enumeration failed for backend {backend}: {exc}")
            continue
        for info in items:
            key = (info.index, info.backend, info.name)
            if key in seen:
                continue
            seen.add(key)
            cameras.append(info)

    if not cameras:
        print("No cameras were enumerated through MSMF or DirectShow.")
        return []

    print("Cameras Windows/OpenCV can enumerate:")
    for n, info in enumerate(cameras):
        try:
            backend_name = cv2.videoio_registry.getBackendName(info.backend)
        except Exception:
            backend_name = str(info.backend)
        print(f"  [{n}] {info.name!r}  index={info.index}  backend={backend_name}")
    return cameras


def open_named_capture(name_query, width, height, fps):
    """Open an enumerated camera whose name contains name_query."""
    if enumerate_cameras is None:
        return None

    q = name_query.casefold()
    matches = []
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        try:
            for info in enumerate_cameras(backend):
                if q in info.name.casefold():
                    matches.append(info)
        except Exception:
            pass

    for info in matches:
        try:
            backend_name = cv2.videoio_registry.getBackendName(info.backend)
        except Exception:
            backend_name = str(info.backend)
        print(f"Trying {info.name!r} at index {info.index} with {backend_name}...")
        cap = cv2.VideoCapture(info.index, info.backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            print(f"Opened {info.name!r} using {backend_name}.")
            return cap
        cap.release()

    return None


def open_first_physical_capture(width, height, fps):
    """Auto-select the first enumerated non-virtual camera."""
    if enumerate_cameras is None:
        return None

    virtual_words = ("virtual", "obs", "manycam", "snap camera")
    for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
        try:
            items = enumerate_cameras(backend)
        except Exception:
            continue
        for info in items:
            lname = info.name.casefold()
            if any(word in lname for word in virtual_words):
                continue
            try:
                backend_name = cv2.videoio_registry.getBackendName(info.backend)
            except Exception:
                backend_name = str(info.backend)
            print(f"Trying enumerated physical camera {info.name!r} "
                  f"(index={info.index}, backend={backend_name})...")
            cap = cv2.VideoCapture(info.index, info.backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"Opened {info.name!r}.")
                return cap, info.name
            cap.release()
    return None

def _capture_backends(requested="auto"):
    """Return capture backends to try, in order."""
    requested = requested.lower()
    if requested == "msmf":
        return [("MSMF", cv2.CAP_MSMF)]
    if requested == "dshow":
        return [("DirectShow", cv2.CAP_DSHOW)]
    if requested == "any":
        return [("automatic", cv2.CAP_ANY)]

    if os.name == "nt":
        # MSMF works better than DirectShow on some current Windows/OpenCV setups.
        return [
            ("MSMF", cv2.CAP_MSMF),
            ("DirectShow", cv2.CAP_DSHOW),
            ("automatic", cv2.CAP_ANY),
        ]
    return [("automatic", cv2.CAP_ANY)]


def open_capture(index, width, height, fps, backend="auto", quiet=False):
    """Open a physical camera, trying sensible Windows backends automatically."""
    for backend_name, backend_id in _capture_backends(backend):
        if not quiet:
            print(f"Trying physical camera {index} with {backend_name}...")
        cap = cv2.VideoCapture(index, backend_id)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        # isOpened() alone can be misleading; require one real frame.
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            if not quiet:
                try:
                    actual = cap.getBackendName()
                except Exception:
                    actual = backend_name
                print(f"Opened physical camera {index} using {actual}.")
            return cap

        cap.release()

    return None


def probe_cameras(max_index, width, height, fps, backend="auto"):
    """Print camera indices that can actually return a frame."""
    print(f"Probing camera indices 0..{max_index}...")
    found = []
    for index in range(max_index + 1):
        cap = open_capture(index, width, height, fps, backend=backend, quiet=True)
        if cap is not None:
            try:
                actual = cap.getBackendName()
            except Exception:
                actual = "unknown backend"
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  camera {index}: OK ({actual}, {w}x{h})")
            found.append(index)
            cap.release()
        else:
            print(f"  camera {index}: unavailable")
    if not found:
        print("No readable cameras found.")
    else:
        print("Readable camera indices:", ", ".join(map(str, found)))
    return found


def parse_args():
    p = argparse.ArgumentParser(description="Face-protected virtual webcam")
    p.add_argument("--camera", type=int, default=0,
                   help="physical webcam index (default: 0)")
    p.add_argument("--camera-name",
                   help="open a physical camera by a substring of its Windows device name")
    p.add_argument("--list-cameras", action="store_true",
                   help="list Windows camera names/indices/backends and exit")
    p.add_argument("--auto-camera", action="store_true",
                   help="auto-select the first enumerated non-virtual camera")
    p.add_argument("--backend", choices=("auto", "msmf", "dshow", "any"),
                   default="auto",
                   help="camera input backend")
    p.add_argument("--probe-cameras", nargs="?", const=5, type=int, metavar="MAX_INDEX",
                   help="test camera indices 0..MAX_INDEX, then exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)

    p.add_argument("--face-detail", type=int, default=32,
                   help="pixel width retained in protected face; lower protects more, default: 32")
    p.add_argument("--face-padding", type=float, default=0.06,
                   help="extra area around detected face; default: 0.06")
    p.add_argument("--redetect-every", type=int, default=2,
                   help="re-anchor tracker with face detection every N frames; default: 2")
    p.add_argument("--face-hold-frames", type=int, default=60,
                   help="keep protecting while trying to reacquire a lost face; default: 60")

    p.add_argument("--seed", type=int,
                   help="reproduce a specific randomized protection profile")
    p.add_argument("--no-randomize", action="store_true",
                   help="disable per-launch protection-profile randomization")

    p.add_argument("--preview", action="store_true",
                   help="show the processed frames locally too")
    p.add_argument("--test-pattern", action="store_true",
                   help="send an animated test pattern instead of opening the webcam")
    return p.parse_args()

def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("width, height, and fps must be positive")

    if args.list_cameras:
        list_named_cameras()
        return 0

    if args.probe_cameras is not None:
        if args.probe_cameras < 0:
            raise SystemExit("--probe-cameras MAX_INDEX must be >= 0")
        probe_cameras(args.probe_cameras, args.width, args.height, args.fps, args.backend)
        return 0

    face_protector = FaceProtector(
        detail=args.face_detail,
        padding=args.face_padding,
        redetect_every=args.redetect_every,
        hold_frames=args.face_hold_frames,
        randomize=not args.no_randomize,
        seed=args.seed,
    )

    cap = None
    input_description = None
    if not args.test_pattern:
        if args.camera_name:
            cap = open_named_capture(args.camera_name, args.width, args.height, args.fps)
            input_description = f"physical webcam matching {args.camera_name!r}"
        elif args.auto_camera:
            result = open_first_physical_capture(args.width, args.height, args.fps)
            if result is not None:
                cap, selected_name = result
                input_description = f"physical webcam {selected_name!r}"
        else:
            cap = open_capture(args.camera, args.width, args.height, args.fps, args.backend)
            input_description = f"physical webcam index {args.camera}"

        if cap is None:
            raise SystemExit(
                "Could not open a physical webcam.\n"
                "Recommended Windows path:\n"
                "  python -m pip install cv2-enumerate-cameras\n"
                "  python main.py --list-cameras\n"
                "  python main.py --camera-name \"PART OF CAMERA NAME\" --preview\n"
                "or try: python main.py --auto-camera --preview\n"
                "Also verify the camera works in the Windows Camera app and that desktop camera access is enabled.\n"
                "Use --test-pattern --preview to test only the virtual-camera output."
            )

    try:
        with pyvirtualcam.Camera(
            width=args.width,
            height=args.height,
            fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera: {cam.device}")
            print(f"Protection profile: {face_protector.profile_summary()}")
            if args.test_pattern:
                print("Input: animated test pattern")
            else:
                print(f"Input: {input_description}")
            print("Select the virtual camera named above in your video app.")
            print("Press Ctrl+C to stop", end="")
            if args.preview:
                print(", or press Q/Esc in the preview window.")
            else:
                print(".")

            frame_no = 0
            while True:
                if args.test_pattern:
                    frame = make_test_pattern(args.width, args.height, frame_no)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        print("\nWebcam read failed.", file=sys.stderr)
                        break
                    if frame.shape[1] != args.width or frame.shape[0] != args.height:
                        frame = cv2.resize(frame, (args.width, args.height),
                                           interpolation=cv2.INTER_AREA)

                frame = face_protector.protect(frame)
                cam.send(frame)

                if args.preview:
                    cv2.imshow("Face-protected webcam preview", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                frame_no += 1
                cam.sleep_until_next_frame()

    except RuntimeError as exc:
        print("\nCould not open a virtual camera.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "Install/initialize a supported virtual-camera driver first; "
            "see the setup notes below.",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
