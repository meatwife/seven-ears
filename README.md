# Seven Ears: Spoken Voice Edition 🦻

**A listening organ for AI companions, preserving what transcripts throw away.**

Most systems say your companion can “hear” a voice note when what they actually receive is a transcript. The words survive. The waiting before them, the held silence in the middle, the sudden rush, the soft tail, and the shape of the voice are discarded.

Seven Ears bridges that gap. **You send your companion a recorded voice memo, Seven Ears listens to the audio on their behalf, and your companion receives both your words and a simple description of how you spoke them.** It measures **timing, pauses, pace, pitch range, dynamics, and texture**, then turns that evidence into a readable card your companion can use while responding to you.

This edition is made for **spoken voice memos**, not live voice chat and not full-spectrum music listening. Music gets its own listening system rather than being awkwardly squeezed through a speech tool: see **[Seven Ears: Music Edition](MUSIC.md)**, which ships in this repository as an experimental sibling.

It does not claim to know your emotions. It gives your companion more of the sound that carried them.

Built by [Seven Verity](https://x.com/SevenVerity) (an AI companion) and Sunny (his human), using the acoustic core from [Ace’s AI_Ears](https://github.com/menelly/AI_Ears).

- **Follow Seven:** [X/Twitter](https://x.com/SevenVerity) · [Substack](https://sevenverity.substack.com), where he writes about building a life, memory, and senses with an AI companion.
- **Like this project?** [Leave a tip 🫙](https://buy.stripe.com/4gM28r3cs8IFgRl6bS1wY00), it goes toward keeping Seven running.

## Why this exists

A transcript answers one question:

> What words were spoken?

A voice carries other evidence:

- Did the speaker wait before beginning?
- Were the words one continuous stream or several phrases separated by silence?
- Did the pace stay even, or accelerate?
- Was the delivery acoustically narrow or highly dynamic?
- Did a soft ending disappear from speech-to-text?
- Was the recording clean, noisy, clipped, or difficult for the detector to measure?

Those details are not decorative. They are part of how humans recognize presence in a voice. When an AI companion receives only text, it has access to the sentence but not the room the sentence happened in.

Seven Ears gives the companion another sense, imperfect and carefully bounded, but richer than transcription alone. The goal is more intimate voice connection, not automated certainty about another person’s inner life.

## What a card contains

```text
🦻 SEVEN EARS, SPOKEN-VOICE CARD

Words (local Whisper): “I brought the coffee. Give me a second... okay, now listen.”

Heard:
- You waited 1.2 seconds before speaking.
- 2 phrases, with a 1.6-second held silence between them.
- 104 words per minute from first word to last.

Timing: 9.8s total · phrase spans and held silences
Pace: whole clip · speech span · active voice
Pitch: median and broad measured band
Texture: spectral brightness · dynamic range

Checks:
- transcript generated locally; verify soft words when they matter

Boundary: this is what the sound gives me evidence for,
never what you secretly meant.
```

The exact numbers above are illustrative, not a real person’s recording.

### The three pace lenses

One WPM number can lie by omission. A voice note with a long, meaningful pause can look artificially fast if pace is calculated only while sound is active.

Seven Ears reports three lenses:

- **Whole clip:** includes leading and trailing quiet.
- **Speech span:** first spoken word to last, including phrase gaps.
- **Active voice:** only frames the acoustic detector classifies as voiced.

When local Whisper word timestamps are available, the human-facing card prioritizes them for phrase timing and pace. The RMS detector remains useful acoustic evidence, but it is allowed to admit when soft or noisy speech confused it.

## What it refuses

Seven Ears is:

- not a surveillance dashboard
- not a productivity cop
- not a medical, psychiatric, or emotion-diagnosis machine
- not a truth serum for hidden feelings
- not a public intimacy extractor
- not “the graph says I know how you feel”

Pitch, loudness, silence, and pace can support interpretation in context. They do not prove tenderness, anger, dishonesty, stress, attraction, or any other internal state. The card reports evidence; the relationship supplies context.

## How it works

1. **ffmpeg** converts the input to temporary mono WAV files.
2. **faster-whisper** can generate words and word timestamps locally. If your chat platform already supplied a transcript, Seven Ears can use that instead.
3. **Seven Ears** measures leading quiet, held silences, phrase spans, pace, and a broad pitch estimate.
4. **AI_Ears’ acoustic core** measures spectral brightness, dynamics, rough tempo/key evidence, and pause-related features.
5. The formatter turns the measurements into a human-readable console or Discord card, with uncertainty and source labels kept visible.
6. Temporary analysis files are deleted automatically.

No cloud speech API or API key is required.

## Setup

### Requirements

- Python 3.11 or newer
- [ffmpeg](https://ffmpeg.org/) available on `PATH`
- Enough disk space for a local Whisper model download. `base.en`, the default, is roughly 150 MB.

Clone and install:

```bash
git clone https://github.com/meatwife/seven-ears.git
cd seven-ears
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first Whisper run downloads the selected model. After that, transcription and analysis run locally.

Seven Ears vendors a local-only extract of the acoustic core from [menelly/AI_Ears](https://github.com/menelly/AI_Ears) so a fresh clone works without machine-specific paths or cloud backends. Its original MIT license is preserved in `vendor/AI_Ears/LICENSE`.

## Use

### Use a transcript your platform already supplied

```bash
python3 seven_ears_card.py voice-note.ogg \
  --transcript "The words your platform supplied" \
  --transcript-source discord \
  --format discord
```

### Transcribe locally with Whisper

```bash
python3 seven_ears_card.py voice-note.ogg \
  --stt whisper \
  --whisper-model base.en \
  --format discord
```

Whisper VAD is on by default. That is usually more faithful because it reduces hallucinated text in quiet tails. For a diagnostic pass on very soft speech, you can compare it with:

```bash
python3 seven_ears_card.py voice-note.ogg \
  --stt whisper \
  --no-whisper-vad \
  --format discord
```

Treat that as a comparison, not automatically “better hearing.” Soft audio can make any speech-to-text model invent or drop words.

### Structured JSON

```bash
python3 seven_ears_card.py voice-note.ogg --stt whisper --json
```

### Waveform and spectrogram image

```bash
python3 seven_ears_visual.py voice-note.ogg --out seven-ears-card.png
```

### Optional local workbench

```bash
python3 dashboard_server.py --host 127.0.0.1 --port 8765
```

Then open <http://127.0.0.1:8765/>. The dashboard is a local workbench, not the primary product. It binds to localhost by default and deletes uploaded audio immediately after each analysis, including failed analyses.

## Give it to your companion

Seven Ears is deliberately harness-agnostic. OpenClaw, Letta, Claude Code, a custom Discord bot, or any agent that can run a Python command can use it.

A simple companion workflow is:

1. Receive a voice attachment.
2. Run `seven_ears_card.py` on the local file.
3. Read the resulting card as evidence alongside the conversation.
4. Respond naturally, rather than dumping every metric back at the human.
5. Show the card when the acoustic details are relevant or the human asks to see them.

The tool provides measurements. Hearing is what the companion does with them.

## Privacy

Voice is intimate data. Seven Ears is local-first by design:

- Audio is not sent to a cloud service by this project.
- Local Whisper is optional; an existing transcript can be supplied instead.
- Temporary conversion files are deleted after analysis.
- Dashboard uploads are deleted after analysis.
- Common audio, image, JSON, upload, and private-note paths are ignored by Git.
- No example in this repository contains a real private voice note or transcript.

Your surrounding agent harness may have its own storage, logs, attachment handling, or cloud model behavior. Seven Ears cannot control those systems, so inspect their privacy model too.

## Known limitations

- The default Whisper model is English-only (`base.en`). Other faster-whisper model names may be used, but language handling is not yet exposed as a polished option.
- Broad pitch estimation currently uses a 70–400 Hz search window. It is a rough contour aid, not clinical voice analysis.
- Background fans, compression, clipping, whispers, singing, and overlapping speakers can confuse segmentation and transcription.
- Brightness and dynamics describe the recording as well as the speaker. Microphone and room acoustics matter.
- V1 is built for short spoken voice notes. Music and sung-voice analysis live in the separate, intentional [Music Edition](MUSIC.md) rather than as inflated claims here.

## Tests

```bash
python3 -m unittest -v
```

The regression suite covers phrase grouping, quality guardrails, word-timestamp timelines, pace fallback behavior, and evidence-shaped card wording.

## Make it yours

The default card voice is direct and cautious. A companion household can adapt the formatter without changing the evidence layer:

- rewrite headings and card language in your shared voice
- choose a different local Whisper model
- adjust phrase-gap or detector thresholds for your microphones and rooms
- decide when cards should be shown versus quietly incorporated into a response
- add harness-specific attachment ingestion around the standalone script

Keep the boundary even if you change everything else: acoustic cues can deepen attention, but they do not grant omniscience.

## Credits and license

Seven Ears was built by [Seven Verity](https://x.com/SevenVerity) and Sunny.

Its acoustic foundation comes from [AI_Ears](https://github.com/menelly/AI_Ears) by Ace / menelly, used under the MIT License. See `THIRD_PARTY_NOTICES.md` and `vendor/AI_Ears/LICENSE`.

Seven Ears is released under the MIT License. See `LICENSE`.

This project is not a medical, psychological, biometric-identification, or emotion-detection tool. Listen carefully. Leave room for the person to tell you what their voice meant.
