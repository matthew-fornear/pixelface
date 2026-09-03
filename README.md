# Face Protected Virtual Webcam

Creates a virtual webcam that tracks your face and reduces fine facial detail while keeping broad appearance, pose, and expression recognizable.

## Install

```powershell
python -m pip install opencv-python numpy pyvirtualcam cv2-enumerate-cameras
```

On Windows, install OBS Studio so the OBS Virtual Camera driver is available.

## Run

```powershell
python face_protected_webcam_randomized.py --auto-camera --preview
```

List cameras:

```powershell
python face_protected_webcam_randomized.py --list-cameras
```

Select one by name:

```powershell
python face_protected_webcam_randomized.py --camera-name "Logitech" --preview
```

## Methodology

The goal is to reduce the amount of fine facial information exposed to downstream capture or ML systems without fully hiding your identity from a human viewer.

The pipeline is:

1. Detect the face with OpenCV Haar cascade detectors.
2. Track the face between detections with pyramidal Lucas-Kanade optical flow.
3. Re-detect periodically to correct tracker drift.
4. Keep protecting the last tracked region during short detection failures.
5. Downsample the face region to remove fine texture.
6. Apply color quantization and mild color remapping.
7. Upscale with nearest-neighbor interpolation to create a coarse pixel representation.
8. Feather the protected region into the surrounding frame.
9. Send the processed frame through a virtual webcam.

The protection profile is randomized once per launch. It remains stable during the call so the video does not flicker.

Randomized properties include:

- Pixel resolution
- Pixel grid phase
- Color quantization levels
- Quantization offsets
- Hue shift
- Saturation
- Brightness
- Per-channel gains
- Pixel aspect ratio
- Edge feathering

This makes each session use a different transformation instead of exposing the same deterministic preprocessing every time.

## Tracking

The program does not rely on face detection every frame.

It uses face detection to acquire and periodically re-anchor the face location. Between detections, optical flow tracks facial feature movement and estimates an affine transform for the protected region.

If tracking temporarily fails, the previous face region remains protected while the detector attempts to reacquire it.

## Technology

### OpenCV

Used for:

- Webcam capture
- Haar cascade face detection
- Profile face detection
- Lucas-Kanade optical flow
- Feature detection
- Affine motion estimation
- Image resizing
- HSV color transformation
- Color quantization
- Edge feathering
- Preview rendering

### NumPy

Used for image arrays, color transforms, coordinate calculations, and pixel processing.

### pyvirtualcam

Publishes the processed frames as a virtual webcam that can be selected in video applications.

### cv2-enumerate-cameras

Enumerates Windows camera devices and their OpenCV backend mappings.

## Tuning

More facial detail:

```powershell
python face_protected_webcam_randomized.py --auto-camera --face-detail 40 --preview
```

Less facial detail:

```powershell
python face_protected_webcam_randomized.py --auto-camera --face-detail 24 --preview
```

Reproduce a protection profile:

```powershell
python face_protected_webcam_randomized.py --auto-camera --seed 123456 --preview
```

Disable randomization:

```powershell
python face_protected_webcam_randomized.py --auto-camera --no-randomize --preview
```

## Limitations

This is a privacy degradation technique, not a guarantee that ML systems cannot extract useful information.

A determined system may still recover broad identity, geometry, motion, expression, or other features from the processed video.

Reducing transmitted information is generally stronger than relying on a visible watermark or a fixed adversarial pattern.
