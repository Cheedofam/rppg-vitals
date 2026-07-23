# rPPG Vitals — Technical Deep Dive

A plain-language explanation of what this project does, the ideas behind it, and how it is
built. It assumes **no prior knowledge** — every technical term is defined the first time it
appears. If you can follow a recipe, you can follow this.

---

## 1. What this project does

It measures a person's **heart rate** (beats per minute) and **breathing rate** (breaths per
minute) using nothing but a normal **webcam** — no chest strap, no finger clip, no wearable.

You point a camera at someone's face, and the software prints their pulse. It does this by
noticing that skin very slightly changes colour every time the heart beats. The technique has a
name: **rPPG**, short for *remote photoplethysmography*. Let's unpack that word:

- *photo* = light, *plethysmo* = volume change, *graphy* = recording.
- So "photoplethysmography" = **recording volume changes using light.**
- The "remote" part just means **from a distance** (a camera), instead of a sensor touching the
  skin.

---

## 2. The big idea, explained simply

### 2.1 Why skin changes colour with your pulse
Your blood contains **haemoglobin**, the molecule that carries oxygen. Haemoglobin **absorbs
light** — especially green light. Every time your heart beats, it pushes a little pulse of blood
into the tiny vessels just under your skin. More blood means more haemoglobin means **slightly
less light bounces back** to the camera. So with each heartbeat, your skin gets a tiny bit
darker, then lighter, then darker again.

This change is **tiny** — about a 1% change in brightness, far too small for your eyes to see.
But a camera measures brightness as numbers, and math can find a 1% wiggle that repeats
regularly.

### 2.2 The one hard problem: the signal is buried in noise
The pulse is a 1% wiggle. Sitting on top of it are **much bigger** unwanted changes:

- **Lighting drift:** lights flicker, clouds pass, the camera auto-adjusts brightness. These
  cause slow, large brightness changes.
- **Movement:** if the head shifts even slightly, the amount of light hitting the skin changes
  far more than 1%.
- **Random camera noise:** every sensor has grain.

"**Noise**" is the general engineering word for *any unwanted part of a measurement*. "**Signal**"
is the part you actually want (here, the pulse). **The entire project is one long effort to pull
the small signal out of the larger noise.** Keep that in mind — every step below exists for that
reason.

---

## 3. A guided tour of how it works

Here is the whole assembly line. Each box hands its output to the next.

```
Webcam → find the face → pick skin patches → average their colour each frame
   → collect a few seconds of colour → clean it up → combine R,G,B into one pulse
   → find the repeating frequency → HEART RATE
   → (a slower version of the same) → BREATHING RATE
   → check the signal quality → show it all on a live dashboard
```

Now each stage in plain terms (the file that does it is named in brackets).

### 3.1 Get pictures from the camera  ·  `capture.py`
The software grabs frames (single pictures) from the webcam, one after another, and records the
**exact time** each frame arrived.

- **Why record the time?** Webcams claim "30 frames per second" but actually drift a bit. Later
  math needs the *true* rate, or the heart-rate number comes out slightly wrong. So we measure
  the real timing instead of trusting the label. (This "measure, don't assume" habit shows up
  again.)
- It can also read from a **video file** instead of a live camera — handy for testing on
  recordings where the true answer is known.

### 3.2 Find the face and pick skin patches  ·  `face_roi.py`
The software uses **MediaPipe** (a free, pre-trained face-tracking tool from Google) to locate
468 points on the face — corners of eyes, edges of lips, and so on. From those points it draws
a **region of interest (ROI)** — engineer-speak for "the specific area we care about" — over the
**forehead and both cheeks.**

- **Why forehead and cheeks?** They are flat, rich in blood vessels (so the pulse is strong), and
  away from the eyes (which glare) and the mouth (which moves).
- It then computes the **average colour** of that skin area for each frame: one red number, one
  green, one blue. Averaging over thousands of skin pixels is itself a noise-reduction trick —
  random speckle in individual pixels cancels out when you average many of them.

### 3.3 Keep a rolling few seconds of colour  ·  `SignalBuffer` in `signal_pipeline.py`
The software remembers the last ~30 seconds of those colour values (with their timestamps). From
this memory it can grab "the last 10 seconds" to compute heart rate, or "the last 30 seconds"
for the slower breathing rate.

### 3.4 Clean up and combine the colours into one pulse  ·  `signal_pipeline.py`
This is the clever core. Two steps:

1. **Detrend:** remove the slow drift. "**Detrending**" just means subtracting the gradual
   up-or-down slope so the wiggle is centred around zero (like tilting a wobbly line back to
   flat).
2. **CHROM:** combine the red, green, and blue channels in a special way that **cancels out the
   movement/glare** and keeps the pulse. This is a published method (its inventors are de Haan &
   Jeanne, 2013). The intuition: glare from the skin is *white* (equal amounts of red, green,
   blue), while the pulse is *coloured* (mostly green). By mixing the channels with the right
   recipe, the white glare subtracts to nothing and the coloured pulse survives. (More detail in
   §4.3.)

The result is a single clean **pulse waveform** — a line that goes up and down with each
heartbeat.

### 3.5 Turn the pulse into a number (heart rate)  ·  `signal_pipeline.py`
A pulse that repeats, say, 72 times a minute has a **frequency** of 72 beats per minute. To find
that frequency, the software uses an **FFT**.

- **FFT** = *Fast Fourier Transform*. In plain terms: a math tool that takes a wiggly line and
  tells you **which repeating rhythms it contains and how strong each one is.** Feed it the pulse
  waveform and it reports "there's a strong rhythm at 1.2 cycles per second" — multiply by 60 and
  that's **72 beats per minute.**

### 3.6 Breathing rate  ·  `breathing.py`
Breathing also changes the skin signal, but slowly (about 0.1–0.5 cycles per second = 6–30
breaths per minute). The software runs the *same* FFT idea on the slow part of the signal. Using
one tool for both jobs keeps the code simple.

### 3.7 Is this estimate trustworthy?  ·  `quality.py`
Not every moment gives a clean reading (someone might turn their head). The software rates each
estimate **good / fair / poor** based on two checks:

- **Sharpness of the pulse (SNR):** if the FFT shows one clear spike, the pulse is real; if it's
  a mushy blur, it's noise. "**SNR**" = *signal-to-noise ratio*, a measure of how much stronger
  the wanted signal is than the noise.
- **Movement:** how much the face patch jumped between frames.

This label is important because — as the testing showed — the readings are accurate when the
signal is sharp and unreliable when it isn't. Flagging the difference is what keeps the tool
honest.

### 3.8 The live dashboard  ·  `dashboard.py`, `run.py`
Finally, a simple web app (built with **Streamlit**, a tool for making quick data dashboards in
Python) shows the camera feed with the skin patches outlined, the live pulse waveform, the
current heart and breathing rates, and the quality badge. To keep the displayed number steady,
it shows a smoothed average of the last several readings, trusting the sharp ones more.

---

## 4. The key ideas, defined simply

These ideas are general — they show up in audio, sensors, and medical devices everywhere.

### 4.1 Sampling rate and the speed limit ("Nyquist")
Taking one picture per frame is called **sampling**. The **sampling rate** is how many samples
per second (here ~30). There is a rule — the **Nyquist limit** — that says you can only detect
rhythms **up to half your sampling rate**. At 30 samples/second you can see rhythms up to 15 per
second, which is plenty for a heart beating a few times per second.

### 4.2 A band-pass filter
A **filter** removes parts of a signal. A **band-pass filter** keeps only rhythms inside a chosen
range ("band") and throws away everything faster or slower. We keep 0.75–4 cycles/second (the
possible heart-rate range) and discard the slow lighting drift and fast noise. The specific
filter used is a **Butterworth** filter (a standard, smooth type). A neat detail: the software
runs the filter **forwards and then backwards**, which cancels the small time-delay a filter
normally adds — so the cleaned waveform still lines up with real time.

### 4.3 CHROM, a little deeper
Raw signal = pulse + glare/movement + drift. The movement is the biggest troublemaker. CHROM's
trick: **glare is white** (equal red, green, blue) because it's just reflected light, whereas the
**pulse is coloured** (blood mostly affects green). CHROM builds two specific colour mixtures
designed so the white part cancels, band-passes them, and combines them with a scaling factor
that erases any leftover movement. What's left is a clean pulse. In the project's demo picture,
the raw green channel looks like random noise while the CHROM output shows an obvious heartbeat —
a good visual of why this step matters.

### 4.4 Getting a precise number from a short recording
There's a trade-off: a **longer** recording gives a **more precise** frequency but makes the app
feel **laggy**; a **shorter** recording is snappy but blurry. The software gets the best of both:
it uses a short 10-second window (snappy) and then sharpens the FFT result with two math tricks —
**zero-padding** (adding zeros to smooth the frequency read-out) and **parabolic interpolation**
(fitting a little curve to the top of the peak to pinpoint it between the coarse steps). Together
these pin the heart rate down to better than 1 beat per minute without needing a long, laggy
window.

There's also a smaller helper called a **window function** (specifically a **Hann window**): it
gently fades the recording's start and end to zero before the FFT, which stops the FFT from
smearing energy into wrong frequencies (a problem called **spectral leakage**).

---

## 5. How the project was built

### 5.1 Test-first development (TDD)
**TDD** = *Test-Driven Development*: you write an automatic check *before* the code, then write
code until the check passes. For example: "make a fake pulse that is exactly 72 beats per minute,
run it through the pipeline, and confirm the answer is 72 (± 2)." Because the fake signal's true
answer is known, a passing test *proves the math is correct* — independent of any camera or
dataset. There are 22 such automatic checks, and they all pass.

### 5.2 Built in phases, saved step by step
The work went in seven phases (camera → raw colour → the math core → breathing → quality +
dashboard → validation → documentation). Each phase ended with its tests passing and a saved
checkpoint (a **git commit** — a labelled snapshot of the code). The saved history reads like the
story of the build.

### 5.3 Separating the "math" from the "hardware"
All the number-crunching lives in **plain functions** that don't touch the camera or the screen.
The webcam and dashboard are thin wrappers around them. This separation is why the core could be
fully tested with no camera at all, and validated on recorded video.

### 5.4 Real problems that came up (and how they were handled)
- **The planned dataset was unavailable.** The standard test dataset (called UBFC-rPPG) was
  blocked by a download limit. The fix: switch to a different, reliably hosted dataset
  (MCD-rPPG) that also happened to include both heart-rate *and* breathing references.
- **A library changed.** The face-tracking tool had removed an old programming interface, so the
  code was updated to its newer one, and the required model file now downloads automatically the
  first time you run it — so a fresh copy "just works."
- **The reference number was misleading.** The dataset listed a single pulse value that didn't
  match the video; the fix was to compare against the dataset's *contact sensor waveform* instead
  (a proper, time-aligned truth). Lesson: always double-check what you're comparing against.

---

## 6. How well it works (and how to read the results)

The project was tested on 5 people from the dataset, comparing its webcam heart rate against a
medical contact sensor.

| Situation | Average error | What it means |
|---|---|---|
| All readings | ~12 beats/min | includes bad moments (movement, poor light) |
| Only sharp-signal readings | **~2 beats/min** | when the pulse is clean, it's genuinely accurate |

The takeaway: **the error shrinks as the signal gets cleaner**, which proves both that the method
works and that the quality check correctly spots the trustworthy moments. The project reports the
bad numbers too, and openly notes the limitations: webcam pulse-reading struggles in poor light
and with motion, and is known to be **less reliable for darker skin tones** (because the green
signal is weaker), and it is **not a medical device**.

---

## 7. Why certain choices were made

| Choice | Reason in one line |
|---|---|
| Use CHROM instead of just the green channel | it cancels movement/glare, giving a much cleaner pulse |
| Use a 10-second window | long enough for a steady reading, short enough to feel live |
| Sharpen the FFT with zero-padding + curve-fitting | precise number without a long, laggy recording |
| Filter forwards *and* backwards | keeps the waveform lined up with real time |
| Measure the real frame timing | webcam speed drifts; assuming 30 fps would bias the result |
| Keep all math in plain, testable functions | can be checked without any camera |
| Only trust sharp-signal readings | the numbers are reliable only when the pulse is clear |

---

## 8. Glossary (plain definitions)

- **rPPG / PPG** — measuring the blood-volume pulse using light; "remote" means via a camera.
- **Signal / noise** — the part you want / everything you don't.
- **ROI (region of interest)** — the specific skin area the software watches.
- **Channel (R/G/B)** — the red, green, or blue brightness of the image.
- **CHROM** — a method that mixes the colour channels to cancel movement and keep the pulse.
- **Filter / band-pass** — math that removes unwanted parts / keeps only a chosen rhythm range.
- **Butterworth filter** — a common, smooth type of filter.
- **FFT (Fast Fourier Transform)** — turns a wiggly signal into "which rhythms it contains."
- **Frequency** — how many times per second something repeats (×60 gives per-minute).
- **Sampling rate / Nyquist** — samples per second / the top rhythm you can detect (half of it).
- **Window function / spectral leakage** — a fade applied before the FFT / the smearing it
  prevents.
- **SNR (signal-to-noise ratio)** — how much stronger the wanted signal is than the noise.
- **MAE (mean absolute error)** — the average size of the mistakes.
- **TDD** — writing the automatic test before the code.
- **git commit** — a saved, labelled snapshot of the code.
- **Streamlit** — a Python tool for quickly building a web dashboard.

---

## 9. How to run it

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                                   # run the automatic checks
python run.py                            # open the live dashboard (webcam, or a video file)
pip install -r requirements-dev.txt
python validation/download_mcd.py        # fetch the test dataset
python validation/run_validation.py      # reproduce the accuracy numbers and figure
```

To read the code, start with `src/signal_pipeline.py` (the math core), then `src/face_roi.py`,
`src/quality.py`, and `src/dashboard.py`. Every idea in this document lives in that code, with
comments explaining each step.
