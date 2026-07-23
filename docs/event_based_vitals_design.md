# Event-Based Contactless Vitals — Design Note (feasibility sketch)

**Status:** design sketch, not built. This note explores whether — and how — the vital-signs
idea in this project could be moved from an ordinary RGB webcam to an **event / neuromorphic
camera** (DVS: Prophesee, iniVation/DAVIS). It exists to think through feasibility *before*
committing to a build, and to connect this project to event-based sensing work.

---

## 1. Why consider it

Event cameras have real advantages for health monitoring: microsecond temporal resolution,
very high dynamic range, low power, no motion blur, and they work in the dark and through large
lighting changes. If vital signs could be read from a DVS, you'd get a lighting-robust,
low-power contactless monitor — attractive for elderly-care / sleep settings.

The catch is that the *signal we currently use* (a ~1% skin-colour change) is a poor match for
how a DVS works. So the design question isn't "port the code" — it's **"what is the right signal
to look for on an event sensor?"**

---

## 2. The core feasibility problem

An event camera fires an event at a pixel only when the **log-intensity change** since its last
event exceeds a **contrast threshold** `C`. Commercial sensors sit around `C ≈ 15–50%` (tunable,
sometimes down to ~10%). The rPPG pulse is a **~1% reflectance modulation** at ~1 Hz.

> **A 1% change is far below the DVS contrast threshold.** A perfectly still, evenly lit face
> therefore produces essentially **no pulse-driven intensity events**. This kills the naive plan
> ("reconstruct frames, then run CHROM"): you cannot reconstruct a modulation that never crossed
> threshold, and there is no colour for CHROM anyway.

So intensity/colour rPPG is the *wrong* target for a DVS. The signal that *does* match the sensor
is **motion**.

---

## 3. Three candidate signal sources

| Source | What it is | DVS match | Verdict |
|---|---|---|---|
| **Colour/intensity pulse** (classic rPPG) | ~1% skin-colour change | sub-threshold, monochrome | ❌ infeasible on a plain DVS |
| **Pulsatile micro-motion (BCG)** | head/skin moves ~sub-mm with each beat as blood is ejected (ballistocardiography) | motion at ~1 Hz — DVS excels at motion | ⚠️ **most promising for heart rate** |
| **Respiration motion** | chest/shoulder moves mm–cm per breath | large, slow motion — easy events | ✅ feasible for breathing rate |

The key pivot: **video-based pulse detection from head motion is already established** (Balakrishnan
et al., *Detecting Pulse from Head Motions in Video*, CVPR 2013). Event cameras are, if anything,
*better* suited to capturing that subtle periodic motion than frame cameras. So the recommended
target is **motion-based**: respiration from torso motion, heart rate from head/facial
micro-motion.

---

## 4. Recommended pipeline

Front-end changes (event → a 1-D signal); **the whole back-end is reused unchanged** from this
project (band-pass → FFT → parabolic peak → spectral-SNR quality).

```
DVS event stream (x, y, polarity, t_µs)
  → ROI selection            face region (HR) and/or chest region (RR)
    → motion signal          per-time-bin activity in the ROI  (two options below)
      → band-pass            0.7–2 Hz (resting HR) or 0.1–0.5 Hz (RR)
        → FFT + parabolic    dominant frequency → rate
          → spectral-SNR     quality gate (reuse quality.py)
```

**Two ways to build the motion signal:**

- **Option A — event-frame accumulation (reuses existing know-how).** Accumulate events into
  short-window frames (~20–50 ms), exactly like an event→frame converter. Track the ROI's vertical
  motion by event centroid or optical flow → a 1-D displacement/velocity signal. Familiar and
  visualisable, at the cost of some temporal smearing.
- **Option B — direct event-rate signal (more "neuromorphic").** Skip frames: bin events in the
  ROI into ~10–20 ms bins (fs ≈ 50–100 Hz) and take the signed polarity sum (direction of motion)
  or the event count (motion energy) as the 1-D signal. Higher temporal fidelity, no reconstruction.

Both feed the identical FFT rate estimator already written in `src/signal_pipeline.py`.

---

## 5. Parameters and design decisions (first-pass)

| Parameter | Suggested value | Rationale |
|---|---|---|
| Accumulation window (Opt. A) | 20–50 ms | dense enough for motion, short vs a 1 Hz beat |
| Event-rate bin (Opt. B) | 10–20 ms → fs 50–100 Hz | Nyquist ≫ 4 Hz HR; smooth rate signal |
| HR band | 0.7–2 Hz (42–120 bpm) | narrower than optical (motion BCG is noisier); resting range |
| RR band | 0.1–0.5 Hz | same as this project |
| ROI (HR) | forehead/upper face | rigid, moves with head BCG, few confounds |
| ROI (RR) | chest/shoulder | largest respiratory motion |
| Polarity | signed sum for motion direction; \|events\| for energy | direction aids BCG; energy aids RR |
| Denoising | nearest-neighbour / refractory filter | remove isolated background-activity noise events |

**Confounds to gate out:** blinking (~0.2–0.4 Hz), talking, ambient vibration. The existing
spectral-SNR quality metric flags weak/ambiguous peaks; a stillness requirement (or IMU/gyro
compensation) helps for HR.

---

## 6. Validation strategy (no hardware needed to start)

There is no ready public event-rPPG dataset, but you don't need a DVS to test the idea:

1. **Simulate events from existing RGB rPPG video.** Feed a clip from this project's datasets
   (MCD-rPPG / UBFC-rPPG, which have ground-truth PPG) through an event-camera simulator —
   **v2e** (Hu et al.) or **ESIM** (Rebecq et al.) — to synthesise a realistic event stream.
2. Run the Option A/B front-end + the reused FFT back-end on the synthetic events.
3. Compare the recovered rate against the clip's contact-PPG ground truth, using the same
   MAE / RMSE / Pearson metrics as this project's `validation/`.

This mirrors the *radar-vitals-sim* philosophy: validate the signal processing in simulation
first, then move to hardware. Later, a real DVS + finger pulse-oximeter gives an on-hardware check.

---

## 7. Honest feasibility verdict

- **Respiration rate:** high confidence — respiratory motion is large and event-friendly.
- **Heart rate (motion/BCG):** moderate — plausible given the video-BCG literature and the DVS's
  motion strength, but the ~1 Hz head micro-motion is small and easily buried by any voluntary
  motion; needs a still subject and careful ROI/denoising. This is the interesting research risk.
- **Heart rate (colour/intensity):** low — sub-threshold and monochrome on a plain DVS. A hybrid
  **DAVIS** grayscale APS frame could support ordinary low-frame-rate grayscale rPPG, but that's
  "a slow grayscale camera," not an event method, and still can't use CHROM.

---

## 8. What to reuse vs build new

**Reuse (already written / already known):**
- FFT rate extraction, band-pass, spectral-SNR quality gate — `src/signal_pipeline.py`,
  `src/quality.py`.
- Validation harness pattern (per-trial CSV + report + figure) — `validation/`.
- Event→frame accumulation and event-`.npy` handling — from the event-based UAV codebase.

**Build new:**
- ROI selection on event/accumulated data (no colour face-mesh; use motion + an initial coarse
  detector or a reconstructed keyframe).
- The event→1-D-motion front-end (Options A/B).
- A v2e/ESIM simulation harness.

---

## 9. Suggested de-risking experiment (½–1 day)

Before any real build, run the smallest test that answers the feasibility question:

1. Take **one** MCD/UBFC clip with a known heart rate.
2. Synthesise events with **v2e**.
3. Build the Option B ROI event-rate signal; band-pass 0.7–2 Hz; FFT.
4. **Does a peak appear near the true heart rate?**
   - *Yes* → motion-BCG on events is viable; proceed to a fuller build + more subjects.
   - *No* → the pulse is genuinely sub-threshold; restrict scope to **respiration + fall** on
     events (still a solid, lighting-robust contribution) and document HR as out of reach.

Either outcome is a publishable, honest result — and a distinctive third portfolio piece that
sits exactly at the intersection of contactless vitals and neuromorphic sensing.

---

## 10. Scope note

This is a *design sketch only*. Nothing here is implemented in this repository. It is recorded as
a future research direction connecting `rppg-vitals` (optical) and `radar-vitals-sim` (radar) to a
third, event-based modality.

**References:** Balakrishnan, Durand & Guttag, *Detecting Pulse from Head Motions in Video*, CVPR
2013 · Hu, Liu, Delbruck, *v2e: From Video Frames to Realistic DVS Events*, CVPRW 2021 · Rebecq et
al., *ESIM: an Open Event Camera Simulator*, CoRL 2018 · Gallego et al., *Event-based Vision: A
Survey*, IEEE TPAMI 2020.
