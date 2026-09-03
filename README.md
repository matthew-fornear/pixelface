# Watermarked Virtual Webcam

Adds repeated diagonal text to your webcam feed and exposes the result as a virtual camera.

## Install

```powershell
python -m pip install opencv-python numpy pyvirtualcam cv2-enumerate-cameras
```

On Windows, install OBS Studio so the OBS Virtual Camera driver is available.

## Test

```powershell
python watermarked_webcam_flowing_8.py --test-pattern --preview
```

Select the virtual camera in Zoom, Teams, Meet, Discord, OBS, or another video app.

## Run

Auto-select a physical camera:

```powershell
python watermarked_webcam_flowing_8.py --auto-camera --preview
```

List cameras:

```powershell
python watermarked_webcam_flowing_8.py --list-cameras
```

Select one by name:

```powershell
python watermarked_webcam_flowing_8.py --camera-name "Logitech" --preview
```

## Options

Change watermark speed:

```powershell
python watermarked_webcam_flowing_8.py --auto-camera --watermark-speed 45 --preview
```

Change opacity:

```powershell
python watermarked_webcam_flowing_8.py --auto-camera --opacity 0.9 --preview
```

Change text:

```powershell
python watermarked_webcam_flowing_8.py --auto-camera --text "NOT FOR MACHINE LEARNING RE-USE" --preview
```

## Notes

The watermark uses 8 parallel diagonal rows and continuously flows from top right to bottom left.

A visible watermark is a deterrent and notice. It cannot guarantee that video will never be altered or reused.
