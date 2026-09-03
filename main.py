#!/usr/bin/env python3
"""
Watermarked virtual webcam.

Reads a physical webcam, stamps repeated diagonal text across every frame,
and publishes the result as a virtual camera using pyvirtualcam.

Examples:
  python watermarked_webcam.py --preview
  python watermarked_webcam.py --camera 1 --preview
  python watermarked_webcam.py --test-pattern --preview
  python watermarked_webcam.py --text "NOT FOR MACHINE LEARNING RE-USE"
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import pyvirtualcam

try:
    from cv2_enumerate_cameras import enumerate_cameras
except ImportError:
    enumerate_cameras = None


def build_watermark(width, height, text, angle=28.0, opacity=0.82,
                    font_scale=1.0):
    """
    Pre-render 8 parallel diagonal watermark lines on an oversized canvas.

    The extra border lets apply_watermark() move a crop along the line direction
    without clipping the pre-rendered text. Text is repeated at an exact period,
    so when the animation phase wraps it continues smoothly.
    """
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.8, min(width, height) / 900.0) * font_scale
    thickness = 3
    outline = 8

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    gap = max(70, int(tw * 0.22))
    x_step = tw + gap

    # Enough padding for one complete animation period in any direction.
    pad = x_step + max(width, height) // 3
    big_w = width + 2 * pad
    big_h = height + 2 * pad

    canvas = np.zeros((big_h, big_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((big_h, big_w), dtype=np.uint8)

    a = math.radians(angle)

    # Exactly eight parallel source rows across the span that can intersect
    # the visible output frame after rotation.
    normal_span = abs(math.sin(a)) * width + abs(math.cos(a)) * height
    center_y = big_h / 2.0
    row_step = normal_span / 8.0
    ys = [
        center_y - normal_span / 2.0 + row_step * (i + 0.5)
        for i in range(8)
    ]

    # Repeat individual phrases far beyond the visible frame so phrases
    # continuously enter and leave at the frame edges.
    first_x = -x_step
    last_x = big_w + x_step

    for y in ys:
        x = first_x
        while x <= last_x:
            pos = (int(x), int(y))
            cv2.putText(
                canvas, text, pos,
                font, scale, (0, 0, 0), outline, cv2.LINE_AA
            )
            cv2.putText(
                canvas, text, pos,
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            cv2.putText(
                canvas_mask, text, pos,
                font, scale, 255, outline, cv2.LINE_AA
            )
            x += x_step

    center = (big_w // 2, big_h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        canvas, M, (big_w, big_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rotated_mask = cv2.warpAffine(
        canvas_mask, M, (big_w, big_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = (rotated_mask.astype(np.float32) / 255.0)
    alpha *= max(0.0, min(1.0, opacity))
    alpha = alpha[:, :, None]

    # In OpenCV image coordinates, this is the source-crop direction along
    # the rotated text line. Moving the crop this way makes the visible words
    # move in the opposite direction: top-right -> bottom-left.
    crop_vx = math.cos(a)
    crop_vy = -math.sin(a)

    return rotated, alpha, pad, float(x_step), crop_vx, crop_vy


def _blend_layer(frame, layer, alpha):
    out = (
        frame.astype(np.float32) * (1.0 - alpha)
        + layer.astype(np.float32) * alpha
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_watermark(frame, layer, alpha, pad, period, crop_vx, crop_vy,
                    frame_no=0, fps=30.0, speed=70.0, static=False):
    """
    Move the words at constant speed along their diagonal,
    from top-right toward bottom-left.
    """
    h, w = frame.shape[:2]

    if static:
        phase = 0.0
    else:
        # Linear motion only. No sine/cosine/circular jitter.
        phase = ((frame_no / max(fps, 0.001)) * speed) % period

    # Shift the source crop up-right. The visible watermark therefore moves
    # down-left. The phase wraps exactly at one phrase-repeat period, making
    # the loop seamless.
    sx = int(round(pad + crop_vx * phase))
    sy = int(round(pad + crop_vy * phase))

    sx = max(0, min(sx, layer.shape[1] - w))
    sy = max(0, min(sy, layer.shape[0] - h))

    wm = layer[sy:sy + h, sx:sx + w]
    a = alpha[sy:sy + h, sx:sx + w]

    return _blend_layer(frame, wm, a)

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
    p = argparse.ArgumentParser(description="Watermarked virtual webcam")
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
                   help="camera input backend; on Windows auto tries MSMF, DirectShow, then default")
    p.add_argument("--probe-cameras", nargs="?", const=5, type=int, metavar="MAX_INDEX",
                   help="test camera indices 0..MAX_INDEX (default MAX_INDEX: 5), then exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--text", default="NOT FOR MACHINE LEARNING RE-USE")
    p.add_argument("--angle", type=float, default=28.0)
    p.add_argument("--opacity", type=float, default=0.82,
                   help="0.0-1.0; default: 0.82")
    p.add_argument("--watermark-speed", type=float, default=70.0,
                   help="diagonal text speed in pixels/second (default: 70)")
    p.add_argument("--static-watermark", action="store_true",
                   help="freeze the diagonal watermark instead of flowing it")
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

    layer, alpha, wm_pad, wm_period, wm_vx, wm_vy = build_watermark(
        args.width, args.height, args.text,
        angle=args.angle, opacity=args.opacity,
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

                frame = apply_watermark(
                    frame, layer, alpha, wm_pad, wm_period, wm_vx, wm_vy,
                    frame_no=frame_no,
                    fps=args.fps,
                    speed=args.watermark_speed,
                    static=args.static_watermark,
                )
                cam.send(frame)

                if args.preview:
                    cv2.imshow("Watermarked webcam preview", frame)
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
