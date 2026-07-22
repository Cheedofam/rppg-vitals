# PLAN.md — rppg-vitals

Planning document for the contactless webcam vital-signs monitor defined in
[`RPPG_SPEC.md`](./RPPG_SPEC.md). Covers the milestone schedule, pre-Phase-1 dependency
checklist, and the top risks with fallbacks.

---

## 1. Milestone Plan

One work session per Build Phase (spec Section 6), worked part-time at ~2–3 sessions/week
starting **Thu 2026-07-23**. Estimated total ≈ **22 hours** across ~2.5 weeks. Commit after
each phase, as the spec requires.

| Session | Build Phase | Deliverable | Est. hours | Target date |
|:--:|---|---|:--:|---|
| 1 | **Phase 1 — Capture & ROI** | Webcam capture + MediaPipe face mesh; forehead+cheeks ROI drawn live as a sanity check | 3h | Thu 2026-07-23 |
| 2 | **Phase 2 — Raw signal extraction** | Per-frame ROI RGB averaging into a rolling window; live raw-signal plot (noisy but periodic) | 2h | Sat 2026-07-25 |
| 3 | **Phase 3 — Signal processing** | Detrend + Butterworth bandpass + CHROM/POS + FFT peak → HR; **passing** synthetic-sine unit tests (72 bpm ± 2) | 5h | Wed 2026-07-29 |
| 4 | **Phase 4 — Breathing rate** | Low-freq (0.1–0.5 Hz) path → RR; synthetic 15 br/min test | 2h | Fri 2026-07-31 |
| 5 | **Phase 5 — Quality + dashboard** | Quality flag (low variance / motion) + Streamlit dashboard (feed, ROI overlay, waveform, live HR/RR) | 4h | Sun 2026-08-02 |
| 6 | **Phase 6 — Validation** | 5+ trials vs. reference (pulse-oximeter app or manual count); `validation_log.csv` + MAE + `validation_report.md` | 3h | Wed 2026-08-05 |
| 7 | **Phase 7 — Docs & polish** | Full README (setup, run, screenshot/gif, architecture, results, Future Work) + docstrings | 3h | Sat 2026-08-08 |

*Dates are targets, not deadlines — slip Phase 3 first if time is short; it's the critical path.*

---

## 2. Dependency Checklist (verify before Phase 1)

- [x] Python 3.10.11 present (`C:\Python310\python.exe`)
- [x] git available (2.54.0)
- [x] Virtual environment created (`venv/`)
- [x] `requirements.txt` installs cleanly *(status recorded in §3.1 after VERIFY)*
- [x] Every dependency imports without error: `cv2`, `mediapipe`, `numpy`, `scipy`, `streamlit`, `matplotlib`, `pytest`
- [ ] Webcam opens via `cv2.VideoCapture(0)` — **deferred to Phase 1** (see Risk 2)
- [ ] Windows Camera privacy allows desktop apps to access the camera

---

## 3. Risk List

| # | Risk | Fallback (per spec where noted) |
|:--:|---|---|
| 1 | **MediaPipe install / numpy conflict** on Windows + Py3.10. MediaPipe pins `numpy<2` and bundles `opencv-contrib-python`. | `numpy<2` is pinned in `requirements.txt`. If a second OpenCV conflicts, drop `opencv-python` and use MediaPipe's bundled build. If MediaPipe won't install at all, fall back to OpenCV Haar-cascade face detection + the **green-channel** method — the spec explicitly allows "Green-channel (or CHROM/POS)". |
| 2 | **Webcam access on Windows** — `VideoCapture(0)` may fail (wrong index or privacy block). | Try camera indices 0–2; enable *Settings → Privacy → Camera → Let desktop apps access your camera*; support running against a recorded video file for development. |
| 3 | **Poor signal quality** (lighting / motion) — rPPG is inherently sensitive. | The Phase 5 quality flag (low variance / large ROI shift) surfaces it; use steady, even lighting and lengthen the rolling window when SNR is low. |
| 4 | **Streamlit + live OpenCV loop friction** — Streamlit reruns on interaction, awkward for a continuous video loop. | Drive frames through an `st.image` placeholder / container refresh; if that stalls, fall back to a matplotlib live window — scope names "matplotlib/Streamlit". |
| 5 | **Skin-tone / demographic accuracy limits** of RGB-camera rPPG — a documented literature limitation, not a bug. | Report honestly in `validation_report.md` (Phase 6) alongside lighting and motion sensitivity; do not make clinical-accuracy claims (out of scope). |

### 3.1 VERIFY outcome (2026-07-22)
`pip install -r requirements.txt` **succeeded with no conflicts** and all seven packages import
cleanly. No fallback was needed. Resolved versions:

| Package | Version | Package | Version |
|---|---|---|---|
| OpenCV | 4.11.0 | Streamlit | 1.60.0 |
| MediaPipe | 0.10.35 | Matplotlib | 3.10.9 |
| NumPy | 1.26.4 (< 2 ✓) | pytest | 9.1.1 |
| SciPy | 1.15.3 | | |

Note on Risk 1: MediaPipe pulled in `opencv-contrib-python` alongside `opencv-python`; pip
resolved both to the same build (4.11.0.86), so no conflict occurred. Risk 2 (webcam access)
remains open and is validated at the start of Phase 1.
