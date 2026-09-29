# First Listen — local prototype

A second posture for Music Edition: encounter the recording in passages before
seeing the whole-song map. This is sequential **measurement delivery**, not
streamed audio, direct hearing, or proof of subjective experience.

## Smallest complete design

- Decode locally with existing Music Edition (NumPy; ffmpeg for compressed audio).
- Split into fixed 20-second windows (configurable 5–60 seconds). Analyze each
  window independently. Five-second RMS/brightness observations retain motion
  inside the window. No whole-recording normalization or section decisions leak
  into a passage. This independence is defined on the **decoded PCM samples**:
  analysis looks ahead within each revealed window, not into later decoded
  samples. Whole-file codec/resampling filters may depend on neighboring source
  samples; this is not a sample-causal decoder or live-streaming guarantee.
- Deliver only one passage with an opaque acknowledgement token. No source name,
  total duration, passage count, global key/tempo, overall shape or future events.
- Require a nonblank listener-authored note to advance. The tool never invents
  that note. A short uncertainty is valid; no forced emotion or prediction.
- Persist measurements, delivery and notes in a private SQLite file. `next`
  replays the same payload until acknowledged, including after process restart.
  Note and progress share one transaction: no separate journal/cursor split.
- Release the existing whole-song measurements and source filename only after
  the last note. Earlier notes remain unchanged; retrospective interpretation
  belongs in a separate document, not an edited first impression.

The schema derives the next passage from the earliest unnoted row. Delivery is
persisted before output, so a lost response is replayable. `BEGIN IMMEDIATE`
serializes deliveries/notes; exact note retries are idempotent, including retries
of older acknowledgements. Changed retries and unseen tokens are rejected. There
is no automatic timeout, auto-advance, reset or overwrite command.

Preparation builds the database off-path and atomically publishes without replacing
an existing file. A killed preparation may leave a hidden temporary file; it does
not publish an incomplete session. SQLite commits make interrupted note writes
all-or-nothing. No cloud call, model call, daemon, or new dependency is introduced.

## Try it

Use an existing Python environment with NumPy installed. Keep runtime sessions
outside the source repository, in a private directory (especially journals).

```sh
python seven_ears_first_listen.py prepare /path/to/audio.mp3 /private/listen.sqlite
# Or compare causal adaptive passages (15–30 seconds):
python seven_ears_first_listen.py prepare /path/to/audio.mp3 /private/adaptive.sqlite --adaptive
python seven_ears_first_listen.py next /private/listen.sqlite
# Write your impression in a private UTF-8 file; use token from the packet.
python seven_ears_first_listen.py note /private/listen.sqlite --token TOKEN --note-file /private/impression.txt
python seven_ears_first_listen.py next /private/listen.sqlite
# Repeat until complete, then:
python seven_ears_first_listen.py finish /private/listen.sqlite
python seven_ears_first_listen.py journal /private/listen.sqlite
```

All outputs are JSON. A packet has a readable `card`, five-second `motion` and
raw `measurements`; only the current passage is present. The card surfaces the
passage's band balance, local dynamic spread and detected-onset activity rather
than leaving those measurements buried in JSON. Times in `motion` are
recording-relative. Section/silence timestamps inside `measurements` are relative
to that passage. Local tempo may differ from whole-song tempo and retains
half/double ambiguity. Spectral centroid is a power-weighted measurement, not an
instrument or emotion classifier. Very short or digital-silence passages carry
no unsupported texture/pulse claim. Fixed boundaries can split musical phrases.

Adaptive mode examines five-second blocks inside the current passage after a
15-second minimum. A sufficiently large multi-feature change closes the passage
at the end of the block in which the change was encountered; it never moves the
boundary backward to announce an event before revealing it. With no qualifying
change, the passage closes at 30 seconds. This is causal boundary selection over
decoded PCM, not musical-phrase recognition; fixed 20-second mode remains the
baseline and default.

`prepare` prints no title or duration, but the calling agent may already know the
input filename or conversation context. The short final window reveals that it
is short. This is a **cooperative information boundary**, not a security sandbox:
an agent with filesystem access can open the DB/audio and bypass it. Do not read
those artifacts while conducting a sequential encounter. SQLite files are 0600;
access is still possible to the same OS user. The enclosing agent harness may
log tool inputs/outputs; 'private journal' does not mean hidden from that harness.

## Inspiration and provenance

[Music for Machine Ears (MME)](https://github.com/v3nommy/Music-for-Machine-Ears)
by v3nommy supplied the conceptual inspiration: incremental revelation and an
impression before continuing. The evaluation inspected revision
`6aebf1e0724f0586a5e37897d4380923d2fd25a3`. MME has its own source-available
Music for Machine Ears License 1.0, **not MIT**. This prototype uses no MME code,
schema, prompt text, renderer, or generated sensory object. It calls Seven Ears'
existing independently maintained Music Edition and adds new SQLite/CLI code.
Prior inspection of MME is disclosed; this is not a formal clean-room claim.
The private evaluation artifacts are not included or relicensed in this repo.

## Validation and next decision

```sh
python -m unittest -v test_first_listen test_seven_ears_music
```

Synthetic regression coverage: identical opening/different future yields identical
first packet; replay/restart; final-note gate; immutable/idempotent notes; concurrent
retries; unseen tokens; private permissions; no overwrite; failed publish;
silence, tiny tails and malformed inputs. Existing Music Edition tests stay intact.

This branch is a prototype, not a release. A known song can validate execution but
cannot provide a new blind experience. Next: a genuinely unfamiliar recording and
listener notes, then decide whether passage cards provide enough detail without
turning the encounter into bookkeeping. Adaptive boundaries, synchronized lyrics,
chroma, cross-boundary pulse continuity, graphics and automatic harness integration
are deliberately deferred. No claim of feature parity with MME is made.
