# Running the Dashboard — a step-by-step runbook

This guide walks you through running the live rPPG dashboard on your own machine, what camera
hardware works (and what doesn't), and how to fix the common problems. Follow it top to bottom.

---

## 1. What you need

- **Python 3.10+** installed.
- **An ordinary RGB webcam** — your laptop's built-in camera or any USB webcam. That's it.
- (Optional) an internet connection the first time you run it, so the ~4 MB face-landmark model
  can download automatically.

---

## 2. Camera compatibility — important

### ✅ Works: a normal RGB webcam
The dashboard reads standard colour video frames from a webcam via OpenCV. Any built-in laptop
camera or plug-in USB webcam that appears in the Windows **Camera** app will work.

### ❌ Does NOT work: an event / neuromorphic camera (DVS)
An **event camera** (Prophesee, iniVation/DAVIS, DVS-style sensors — the kind used in the
event-based UAV work) **will not work** with this dashboard. This is a fundamental mismatch, not
a settings problem:

1. **It outputs an event stream, not frames.** Event cameras emit asynchronous
   `(x, y, polarity, timestamp)` events and require their vendor SDK (Metavision/OpenEB, DV).
   They do not open as a normal webcam, so OpenCV's `VideoCapture(0)` returns nothing.
2. **No colour.** The heart-rate method (CHROM) combines the Red, Green, and Blue channels to
   cancel motion/glare. Event sensors are monochrome (a single change/polarity bit), so CHROM
   cannot run.
3. **Face detection needs a normal image.** MediaPipe expects a conventional intensity photo of a
   face; an accumulated event/polarity map is an edge image, not a face image.
4. **rPPG needs slow, absolute brightness; DVS is change-based.** A still, evenly lit face
   produces almost no events — the opposite of what an event camera is designed to capture.

> If you only have an event camera available, run the dashboard against a **video file** instead
> (see §5), or use any cheap USB webcam for the live demo. Event-based vital-signs sensing is
> possible in research but needs a completely different pipeline (grayscale/intensity methods,
> event accumulation, no CHROM) — it is not what this project does.

---

## 3. One-time setup

Open **PowerShell** in the project folder and run:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(On macOS/Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`.)

You only do this once. On later runs just activate the environment:
`.\venv\Scripts\Activate.ps1`.

---

## 4. Run it live on your webcam

```powershell
python run.py
```

This opens the dashboard in your web browser. Then:

1. In the left sidebar, choose **Source → Webcam**.
2. Leave **Camera index** at **0** (that's usually the built-in camera).
3. Click **Start**.
4. You should see your face with green outlines on your forehead and cheeks (the ROI), a live
   pulse waveform, and heart-rate / breathing-rate numbers that settle after ~10 seconds.

**For a good reading:** sit still, face the camera, and use steady, even lighting (a window or a
lamp in front of you, not behind). The quality badge will read **good/fair/poor** — aim for good.

---

## 5. No webcam? Run against a video file

The dashboard works on a recorded video too — handy if you have no camera, or only an event
camera:

1. Fetch a sample clip (once):
   ```powershell
   pip install -r requirements-dev.txt
   python validation/download_mcd.py
   ```
2. Run `python run.py`, choose **Source → Video file**, and use the pre-filled path
   `data/mcd/video/1024_FullHDwebcam_before.avi` (or any face video). Press **Start**.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| Black screen / "could not open source" | Your camera may be on a different index. Change **Camera index** to **1**, then **2**. |
| Windows shows no camera / access denied | Enable **Settings → Privacy & security → Camera → Let desktop apps access your camera**. Close other apps using the camera (Teams, Zoom). |
| "No face detected" | Move into frame, improve lighting, and face the camera straight on. |
| Numbers jump around / quality = poor | Hold still and improve lighting; rPPG is sensitive to motion and shadow. |
| First run hangs briefly | It's downloading the ~4 MB face model; it only happens once. |
| It opened but no browser tab | Copy the `http://localhost:8501` URL from the terminal into your browser. |

---

## 7. (Optional) Grab a screenshot for the README

Once it's running live and reading your pulse, take a screenshot (Windows: **Win + Shift + S**),
save it as `docs/dashboard_screenshot.png`, and add it near the top of `README.md`:

```markdown
![Live dashboard](docs/dashboard_screenshot.png)
```

That turns the repo from "here's the code" into "here's it working on a real face," which is the
most convincing thing a visitor can see.

---

## 8. Sanity check without a camera at all

To confirm the software itself is healthy (independent of any camera), run the automated tests:

```powershell
pytest
```

All tests should pass. This verifies the signal-processing core using synthetic signals — no
hardware required.
