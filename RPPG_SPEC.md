# RPPG_SPEC.md — Contactless Vital Signs Monitor (Webcam rPPG)

## 1. Objective

Build a complete, working application that extracts **heart rate** and **breathing rate**
from an ordinary laptop webcam using **remote photoplethysmography (rPPG)** — no wearable
hardware, no cost. The end state is a runnable local app with a live dashboard showing
real-time HR/RR estimates, plus a documented signal-processing pipeline and a validation
report comparing the tool's output against a reference (phone pulse-oximeter app or manual
count) across at least 5 trials.

This project exists to demonstrate mastery of the same signal-processing skill set used in
contactless health monitoring research (CARG lab, Professor Bolic) — face/skin ROI
extraction, temporal signal filtering, and frequency-domain vital sign extraction — so it
must be built to a standard suitable for showing a research supervisor, not a toy demo.

## 2. Scope

**In scope**
- Real-time webcam capture and face/ROI detection
- Green-channel (or CHROM/POS algorithm) rPPG signal extraction
- Bandpass filtering + FFT-based heart rate estimation (0.75–4 Hz / 45–240 bpm)
- Breathing rate estimation via chest/shoulder motion or low-frequency rPPG envelope
  (0.1–0.5 Hz / 6–30 breaths/min)
- Live matplotlib/Streamlit dashboard showing waveform + current HR/RR
- Signal quality indicator (flags motion artifacts / poor lighting)
- A validation script that logs estimated vs. reference HR over multiple trials and reports
  mean absolute error
- README with setup instructions, architecture diagram (ASCII or Mermaid), and results

**Out of scope (do not build)**
- Mobile app version
- Cloud deployment
- Clinical-grade accuracy claims — this is a portfolio/research project, not a medical device
- Any pathology classification (that lives in the radar project)

## 3. Tech Stack (100% free/open-source)

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Camera capture | OpenCV (`cv2.VideoCapture`) |
| Face/landmark detection | MediaPipe Face Mesh |
| Signal processing | NumPy, SciPy (`scipy.signal` for Butterworth bandpass, FFT) |
| Dashboard | Streamlit (simplest path to a live, presentable UI) |
| Testing | pytest for the signal-processing unit tests (synthetic sine-wave signals with known frequency) |
| Packaging | `requirements.txt` + a single `run.py` entry point |

No paid APIs, no cloud services, no GPU required.

## 4. Architecture

```
Webcam (cv2) 
   -> Face Mesh landmarks (MediaPipe)
      -> ROI selection (forehead + cheeks polygon)
         -> Spatial averaging -> raw RGB signal per frame
            -> Signal buffer (rolling N-second window, e.g. 10s)
               -> Detrend + bandpass filter (Butterworth, 0.75-4Hz for HR)
                  -> CHROM or POS combination (RGB -> single pulse signal)
                     -> FFT -> peak frequency -> HR (bpm)
               -> Separate low-freq path (0.1-0.5Hz) -> RR (breaths/min)
                  -> Streamlit dashboard (waveform plot + live HR/RR numbers + signal quality)
```

## 5. Repository Structure

```
rppg-vitals/
├── README.md
├── requirements.txt
├── run.py                     # entry point: launches Streamlit app
├── src/
│   ├── capture.py              # webcam frame capture loop
│   ├── face_roi.py             # MediaPipe face mesh + ROI extraction
│   ├── signal_pipeline.py      # filtering, CHROM/POS, FFT extraction
│   ├── breathing.py            # low-frequency breathing rate extraction
│   ├── quality.py              # signal quality / motion artifact detection
│   └── dashboard.py            # Streamlit UI
├── tests/
│   ├── test_signal_pipeline.py # synthetic sine wave -> verify frequency recovery
│   └── test_breathing.py
├── validation/
│   ├── validation_log.csv      # trial results: estimated vs reference HR/RR
│   └── validation_report.md    # methodology + MAE results + discussion
└── docs/
    └── architecture.md         # pipeline diagram + explanation of CHROM/POS method
```

## 6. Build Phases (do these in order, commit after each)

**Phase 1 — Capture & ROI**
Implement webcam capture and MediaPipe face mesh. Draw the selected ROI (forehead +
cheeks, avoiding eyes/mouth) on screen as a live sanity check. Commit: "Phase 1: face
detection and ROI extraction working."

**Phase 2 — Raw signal extraction**
Average RGB values within the ROI per frame, buffer into a rolling window. Plot the raw
signal live to confirm it's noisy but periodic. Commit: "Phase 2: raw RGB signal buffer."

**Phase 3 — Signal processing pipeline**
Implement detrending, Butterworth bandpass filter, and the CHROM (or POS) algorithm to
combine RGB channels into a clean pulse signal. Implement FFT-based peak detection for HR.
Write unit tests using synthetic sine waves at known frequencies (e.g., generate a 72bpm =
1.2Hz sine wave, verify the pipeline recovers 72 ± 2 bpm). Commit only once these tests pass.

**Phase 4 — Breathing rate**
Add the low-frequency extraction path for RR. Test similarly with synthetic 15
breaths/min signal.

**Phase 5 — Signal quality + dashboard**
Add a quality flag (e.g., low variance = poor signal, large frame-to-frame ROI shift =
motion artifact). Build the Streamlit dashboard: live camera feed with ROI overlay, live
waveform plot, current HR/RR numbers, quality indicator.

**Phase 6 — Validation**
Run 5+ trials against a reference (a phone pulse oximeter app, or manual 30-second pulse
count doubled). Log estimated vs. reference in `validation_log.csv`. Compute mean absolute
error. Write `validation_report.md` summarizing methodology, results, and known limitations
(lighting sensitivity, motion sensitivity, skin tone considerations — note this honestly,
it's a known limitation of RGB-camera rPPG in the literature).

**Phase 7 — Documentation & polish**
Write the full README (setup, run instructions, screenshot/gif of the dashboard,
architecture explanation, validation summary, and a "Future Work" section referencing how
this connects to radar-based contactless monitoring). Clean up code, add docstrings.

## 7. Definition of Done

- `python run.py` launches a working Streamlit dashboard on a fresh clone + `pip install -r requirements.txt`
- All pytest tests pass
- Validation report exists with real logged trials and an honest MAE figure
- README is complete enough that a stranger could run it in under 5 minutes
- Code is committed incrementally to git with descriptive messages (not one giant commit)

## 8. Resume bullet (for reference, do not put in README as-is)

"Built a real-time contactless vital signs monitor extracting heart rate and breathing rate
from webcam video using remote photoplethysmography (CHROM algorithm, Butterworth
filtering, FFT-based frequency extraction); validated against reference measurements with
documented mean absolute error."
