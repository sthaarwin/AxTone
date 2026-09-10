#!/usr/bin/env python3
"""
FastAPI server for AxTone - Audio to Guitar Tab Converter
Provides REST API endpoints for the Next.js frontend
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import re
from pathlib import Path
from typing import Optional
import base64
import asyncio
import librosa
from fastapi.concurrency import run_in_threadpool

from src.extractor import AudioExtractor, midi_number_to_note_name
from src.optimizer import FretboardOptimizer
from src.formatter import TablatureFormatter

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_DURATION_SECONDS = 60  # 1 minute max duration
MAX_CONCURRENT_JOBS = 2

# Limit simultaneous heavy ML processing to avoid out-of-memory crashes
processing_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

ALLOWED_EXTENSIONS = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')

TUNINGS = {
    'standard': [40, 45, 50, 55, 59, 64],  # E2-A2-D3-G3-B3-E4
    'drop-d':   [38, 45, 50, 55, 59, 64],  # D2-A2-D3-G3-B3-E4
    'drop-c':   [36, 43, 48, 53, 57, 62],  # C2-G2-C3-F3-A3-D4
    'open-g':   [38, 43, 50, 55, 59, 62],  # D2-G2-D3-G3-B3-D4
    'dadgad':   [38, 45, 50, 55, 45, 50],  # D2-A2-D3-G3-A3-D4
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """
    Return a safe version of *filename* suitable for use as a temp-file suffix.

    Keeps only the extension (e.g. '.mp3') after stripping any path components,
    replacing spaces, and removing characters that could be mis-parsed by the
    OS or by HTTP multipart parsers.
    """
    # Grab just the name portion (defence against path traversal)
    basename = Path(filename).name
    # Extract extension, lower-cased
    ext = Path(basename).suffix.lower()
    # Ensure it's something we recognise; fall back to .tmp
    if ext not in ALLOWED_EXTENSIONS:
        ext = '.tmp'
    return ext


def filter_notes_to_guitar_range(notes, tuning, max_fret: int = 22):
    """Filter MIDI notes to only those playable on guitar with the given tuning."""
    min_note = min(tuning)
    max_note = max(tuning) + max_fret

    filtered = [n for n in notes if min_note <= n.midi_number <= max_note]

    removed = len(notes) - len(filtered)
    if removed > 0:
        print(f"ℹ Filtered out {removed} notes outside guitar range ({min_note}–{max_note})")

    return filtered


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AxTone API",
    description="Convert audio files to guitar tablature using AI",
    version="1.0.0",
)

# Read allowed origins from environment; default to localhost for development.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]

print(f"🌐 CORS Allowed Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {"service": "AxTone API", "version": "1.0.0", "status": "running"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "AxTone API"}


@app.get("/api/tunings")
async def get_tunings():
    """Get available guitar tunings."""
    return {
        "tunings": [
            {"value": "standard", "label": "Standard (E-A-D-G-B-E)"},
            {"value": "drop-d",   "label": "Drop D (D-A-D-G-B-E)"},
            {"value": "drop-c",   "label": "Drop C (C-G-C-F-A-D)"},
            {"value": "open-g",   "label": "Open G (D-G-D-G-B-D)"},
            {"value": "dadgad",   "label": "DADGAD (D-A-D-G-A-D)"},
        ]
    }


@app.post("/api/convert")
async def convert_audio_to_tab(
    file: UploadFile = File(...),
    method: str = Form("pyin"),
    tuning: str = Form("standard"),
    fingering: str = Form("dijkstra"),
    min_duration: float = Form(0.1),
    detailed: bool = Form(False),
    preprocess: bool = Form(False),
):
    """
    Convert an audio file to guitar tablature.

    Parameters
    ----------
    file        : Audio file (.mp3 .wav .flac .ogg .m4a .aac) — max 50 MB
    method      : Pitch detection method (only 'pyin' is supported)
    tuning      : Guitar tuning preset or comma-separated MIDI numbers
    min_duration: Minimum note duration in seconds (default 0.1)
    detailed    : Include per-note detail in the response (default False)
    preprocess  : Unused — kept for API compatibility (default False)

    Returns
    -------
    JSON with tablature, statistics, and optional per-note data.
    """

    # ── Validate filename / extension ─────────────────────────────────────
    raw_filename = file.filename or ""
    if not raw_filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = sanitize_filename(raw_filename)
    if ext == '.tmp':
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format. "
                f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Only pyin supported server-side ──────────────────────────────────
    if method != "pyin":
        raise HTTPException(
            status_code=400,
            detail="Only 'pyin' detection is supported on this server.",
        )

    # ── Read file content + size check (Streaming) ────────────────────────
    tmp_path = None
    midi_path = None

    try:
        # Write directly to disk in chunks to save memory
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp_path = tmp.name
            total_size = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_BYTES:
                    # Exceeded limit; clean up and reject
                    os.unlink(tmp_path)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large. "
                            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                        ),
                    )
                tmp.write(chunk)

        print(f"Processing: {raw_filename!r}  →  temp: {tmp_path} ({total_size} bytes)")

        # ── Audio duration check ──────────────────────────────────────────
        try:
            # get_duration reads headers without loading the full audio array
            duration = librosa.get_duration(path=tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid or corrupt audio file.")
            
        if duration > MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Audio is too long ({duration:.1f}s). Maximum allowed is {MAX_DURATION_SECONDS} seconds."
            )
        print(f"Audio duration: {duration:.1f}s (Method: {method}, Tuning: {tuning})")

        # ── Extract MIDI notes ────────────────────────────────────────────
        extractor = AudioExtractor(method=method, min_note_duration=min_duration)
        
        # Limit concurrent CPU-heavy extractions so we don't OOM the server
        async with processing_semaphore:
            print(f"Acquired lock. Extracting notes for {tmp_path}...")
            # Run the synchronous ML task in a threadpool so we don't block FastAPI
            midi_notes = await run_in_threadpool(extractor.extract, tmp_path)

        if not midi_notes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No notes detected in the audio file. "
                    "Try a clearer recording or a smaller min_duration value."
                ),
            )

        print(f"✓ Extracted raw {len(midi_notes)} notes")

        # ── Consolidate & Denoise ─────────────────────────────────────────
        from src.extractor import consolidate_notes
        midi_notes = consolidate_notes(midi_notes)
        if not midi_notes:
            raise HTTPException(
                status_code=400,
                detail="No notes remained after denoising. Try a cleaner recording.",
            )
        print(f"✓ After denoising: {len(midi_notes)} notes")

        # ── Resolve tuning ────────────────────────────────────────────────
        if tuning in TUNINGS:
            tuning_notes = TUNINGS[tuning]
        else:
            try:
                tuning_notes = [int(x.strip()) for x in tuning.split(",")]
                if len(tuning_notes) != 6:
                    raise ValueError("Need exactly 6 values")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid tuning '{tuning}'. "
                        f"Use a preset ({', '.join(TUNINGS)}) or "
                        f"comma-separated MIDI numbers for all 6 strings."
                    ),
                )

        # ── Filter to guitar range ────────────────────────────────────────
        midi_notes = filter_notes_to_guitar_range(midi_notes, tuning_notes, max_fret=22)
        if not midi_notes:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No playable notes found within guitar range. "
                    "Try a different tuning or audio file."
                ),
            )

        print(f"✓ {len(midi_notes)} notes within guitar range")

        # ── Optimize fingering ────────────────────────────────────────────
        optimizer = FretboardOptimizer(tuning=tuning_notes)
        path, _ = optimizer.optimize(midi_notes, method=fingering)  # optimize() returns (path, tablature)

        if not path:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not find a valid fingering path. "
                    "The notes may be outside guitar range for the selected tuning."
                ),
            )

        print("✓ Optimized fingering path")

        # ── Format tablature ──────────────────────────────────────────────────
        # Pass hand_positions so the formatter can render position markers
        hand_positions = optimizer.hand_positions if optimizer.hand_positions else None
        formatter = TablatureFormatter()
        tablature = formatter.format(path, midi_notes, hand_positions=hand_positions, clean=True)

        # ── Statistics ────────────────────────────────────────────────────
        midi_nums = [n.midi_number for n in midi_notes]
        total_fret_distance = sum(
            abs(path[i].fret - path[i - 1].fret) for i in range(1, len(path))
        )
        total_string_jumps = sum(
            1 for i in range(1, len(path)) if path[i].string != path[i - 1].string
        )
        avg_fret_movement = total_fret_distance / len(path) if len(path) > 1 else 0
        avg_string_jumps = total_string_jumps / len(path) if len(path) > 1 else 0

        response: dict = {
            "success": True,
            "tablature": tablature,
            "stats": {
                "total_notes": len(midi_notes),
                "pitch_range": {
                    "min": midi_number_to_note_name(min(midi_nums)),
                    "max": midi_number_to_note_name(max(midi_nums)),
                    "min_midi": min(midi_nums),
                    "max_midi": max(midi_nums),
                },
                "avg_fret_movement": round(avg_fret_movement, 2),
                "avg_string_jumps": round(avg_string_jumps, 2),
                "tuning_used": tuning,
            },
        }

        # ── Optional MIDI export ──────────────────────────────────────────
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as midi_tmp:
                midi_path = midi_tmp.name

            extractor.save_midi(midi_notes, midi_path)

            with open(midi_path, "rb") as midi_file:
                response["midi_base64"] = base64.b64encode(midi_file.read()).decode("utf-8")
        except Exception as e:
            print(f"Warning: Could not generate MIDI: {e}")
            response["midi_base64"] = None

        # ── Optional per-note detail ──────────────────────────────────────
        if detailed:
            hp_list = optimizer.hand_positions if optimizer.hand_positions else [None] * len(path)
            response["notes"] = [
                {
                    "midi": note.midi_number,
                    "note_name": midi_number_to_note_name(note.midi_number),
                    "onset": round(note.onset, 3),
                    "offset": round(note.offset, 3),
                    "duration": round(note.offset - note.onset, 3),
                    "string": pos.string,
                    "fret": pos.fret,
                    "hand_position": hp,
                }
                for note, pos, hp in zip(midi_notes, path, hp_list)
            ]

        print("✓ Conversion complete!")
        return JSONResponse(response)

    except HTTPException:
        raise

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    finally:
        for path_to_clean in (tmp_path, midi_path):
            if path_to_clean and os.path.exists(path_to_clean):
                try:
                    os.unlink(path_to_clean)
                except Exception as cleanup_err:
                    print(f"Warning: Could not delete temp file {path_to_clean}: {cleanup_err}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("Starting AxTone API Server")
    print("=" * 80)
    print("API Documentation: http://localhost:8000/docs")
    print("API Endpoint:      http://localhost:8000/api/convert")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
