#!/usr/bin/env python3
"""
FastAPI server for AxTone - Audio to Guitar Tab Converter
Provides REST API endpoints for the Next.js frontend
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import tempfile
import os
from pathlib import Path
from typing import Optional
import base64

from src.extractor import AudioExtractor, midi_number_to_note_name
from src.optimizer import FretboardOptimizer
from src.formatter import TablatureFormatter


def filter_notes_to_guitar_range(notes, tuning, max_fret=22):
    """
    Filter MIDI notes to only include those playable on guitar with the given tuning.
    
    Args:
        notes: List of MidiNote objects
        tuning: List of open string MIDI numbers
        max_fret: Maximum fret number
    
    Returns:
        Filtered list of MidiNote objects
    """
    min_note = min(tuning)  # Lowest open string
    max_note = max(tuning) + max_fret  # Highest fret on highest string
    
    filtered = [note for note in notes if min_note <= note.midi_number <= max_note]
    
    removed_count = len(notes) - len(filtered)
    if removed_count > 0:
        print(f"ℹ Filtered out {removed_count} notes outside guitar range ({min_note}-{max_note})")
    
    return filtered

app = FastAPI(
    title="AxTone API",
    description="Convert audio files to guitar tablature using AI",
    version="1.0.0"
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGINS
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preset tunings
TUNINGS = {
    'standard': [40, 45, 50, 55, 59, 64],  # E2-A2-D3-G3-B3-E4
    'drop-d': [38, 45, 50, 55, 59, 64],    # D2-A2-D3-G3-B3-E4
    'drop-c': [36, 43, 48, 53, 57, 62],    # C2-G2-C3-F3-A3-D4
    'open-g': [38, 43, 50, 55, 59, 62],    # D2-G2-D3-G3-B3-D4
    'dadgad': [38, 45, 50, 55, 45, 50],    # D2-A2-D3-G3-A3-D4
}


@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "service": "AxTone API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "AxTone API"
    }


@app.get("/api/tunings")
async def get_tunings():
    """Get available guitar tunings"""
    return {
        "tunings": [
            {"value": "standard", "label": "Standard (E-A-D-G-B-E)"},
            {"value": "drop-d", "label": "Drop D (D-A-D-G-B-E)"},
            {"value": "drop-c", "label": "Drop C (C-G-C-F-A-D)"},
            {"value": "open-g", "label": "Open G (D-G-D-G-B-D)"},
            {"value": "dadgad", "label": "DADGAD (D-A-D-G-A-D)"},
        ]
    }


@app.post("/api/convert")
async def convert_audio_to_tab(
    file: UploadFile = File(...),
    method: str = Form("pyin"),
    tuning: str = Form("standard"),
    min_duration: float = Form(0.1),
    detailed: bool = Form(False),
    preprocess: bool = Form(False)
):
    """
    Convert audio file to guitar tablature.
    
    Parameters:
    - file: Audio file (.mp3, .wav, .flac, .ogg, .m4a)
    - method: Pitch detection method ('basic_pitch' or 'pyin')
    - tuning: Guitar tuning preset (standard, drop-d, drop-c, open-g, dadgad)
    - min_duration: Minimum note duration in seconds (default: 0.1)
    - detailed: Include detailed note information (default: False)
    - preprocess: Preprocess audio (normalize and trim) (default: False)
    
    Returns:
    - JSON with tablature, statistics, and optional detailed note info
    """
    
    # Validate file type
    allowed_extensions = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')
    if not file.filename or not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Validate method
    if method not in ['basic_pitch', 'pyin']:
        raise HTTPException(
            status_code=400,
            detail="Invalid method. Must be 'basic_pitch' or 'pyin'"
        )
    
    tmp_path = None
    
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        print(f"Processing file: {file.filename}")
        print(f"Method: {method}, Tuning: {tuning}, Min Duration: {min_duration}")
        
        # Extract MIDI notes
        extractor = AudioExtractor(method=method, min_note_duration=min_duration)
        midi_notes = extractor.extract(tmp_path)
        
        if not midi_notes:
            raise HTTPException(
                status_code=400,
                detail="No notes detected in audio file. Try using a clearer recording or adjusting the min_duration parameter."
            )
        
        print(f"✓ Extracted {len(midi_notes)} notes")
        
        # Get tuning
        if tuning in TUNINGS:
            tuning_notes = TUNINGS[tuning]
        else:
            try:
                # Try parsing as comma-separated MIDI numbers
                tuning_notes = [int(x.strip()) for x in tuning.split(',')]
                if len(tuning_notes) != 6:
                    raise ValueError("Custom tuning must have 6 strings")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tuning: {tuning}. Use preset names or comma-separated MIDI numbers."
                )
        
        # Filter notes to guitar range (IMPORTANT for basic_pitch which may detect out-of-range notes)
        midi_notes = filter_notes_to_guitar_range(midi_notes, tuning_notes, max_fret=22)
        
        if not midi_notes:
            raise HTTPException(
                status_code=400,
                detail="No playable notes found within guitar range. Try a different audio file or tuning."
            )
        
        print(f"✓ {len(midi_notes)} notes within guitar range")
        
        # Optimize fingering
        optimizer = FretboardOptimizer(tuning=tuning_notes)
        path = optimizer.optimize(midi_notes)
        
        if not path:
            raise HTTPException(
                status_code=400,
                detail="Could not find valid fingering path. Notes might be outside guitar range or incompatible with the selected tuning."
            )
        
        print(f"✓ Optimized fingering path")
        
        # Format tablature (clean format without headers for web display)
        formatter = TablatureFormatter()
        tablature = formatter.format(path, midi_notes, clean=True)
        
        # Calculate statistics
        midi_nums = [n.midi_number for n in midi_notes]
        min_note = min(midi_nums)
        max_note = max(midi_nums)
        
        # Calculate average fret movement and string jumps
        total_fret_distance = 0
        total_string_jumps = 0
        
        for i in range(1, len(path)):
            prev_pos = path[i-1]
            curr_pos = path[i]
            
            # Fret distance
            total_fret_distance += abs(curr_pos.fret - prev_pos.fret)
            
            # String jumps
            if prev_pos.string != curr_pos.string:
                total_string_jumps += 1
        
        avg_fret_movement = total_fret_distance / len(path) if len(path) > 1 else 0
        avg_string_jumps = total_string_jumps / len(path) if len(path) > 1 else 0
        
        # Prepare response
        response = {
            "success": True,
            "tablature": tablature,
            "stats": {
                "total_notes": len(midi_notes),
                "pitch_range": {
                    "min": midi_number_to_note_name(min_note),
                    "max": midi_number_to_note_name(max_note),
                    "min_midi": min_note,
                    "max_midi": max_note
                },
                "avg_fret_movement": round(avg_fret_movement, 2),
                "avg_string_jumps": round(avg_string_jumps, 2),
                "tuning_used": tuning
            }
        }
        
        # Generate and include MIDI file as base64
        midi_path = None
        try:
            # Create temporary MIDI file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mid') as midi_tmp:
                midi_path = midi_tmp.name
            
            # Save MIDI using extractor
            extractor.save_midi(midi_notes, midi_path)
            
            # Read and encode as base64
            with open(midi_path, 'rb') as midi_file:
                midi_data = base64.b64encode(midi_file.read()).decode('utf-8')
                response["midi_base64"] = midi_data
        except Exception as e:
            print(f"Warning: Could not generate MIDI: {e}")
            response["midi_base64"] = None
        finally:
            # Cleanup MIDI temp file
            if midi_path and os.path.exists(midi_path):
                try:
                    os.unlink(midi_path)
                except:
                    pass
        
        # Add detailed note information if requested
        if detailed:
            response["notes"] = [
                {
                    "midi": note.midi_number,
                    "note_name": midi_number_to_note_name(note.midi_number),
                    "onset": round(note.onset, 3),
                    "offset": round(note.offset, 3),
                    "duration": round(note.offset - note.onset, 3),
                    "string": pos.string,
                    "fret": pos.fret
                }
                for note, pos in zip(midi_notes, path)
            ]
        
        print(f"✓ Conversion complete!")
        return JSONResponse(response)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log the error and return a generic error response
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {tmp_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("Starting AxTone API Server")
    print("=" * 80)
    print("API Documentation: http://localhost:8000/docs")
    print("API Endpoint: http://localhost:8000/api/convert")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
