# 🎸 AxTone - Full Stack Setup Guide

## Quick Start (Easiest Method)

```bash
# From the project root
./start.sh
```

This will start both the Python backend and Next.js frontend automatically!

- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

Press `Ctrl+C` to stop both servers.

---

## Manual Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Python API Backend

```bash
python api.py
```

The API will run on `http://localhost:8000`

### 3. Start the Next.js Frontend (in a new terminal)

```bash
cd frontend/axtone
npm install  # First time only
npm run dev
```

The frontend will run on `http://localhost:3000`

---

## How to Use

1. **Open your browser** to http://localhost:3000
2. **Upload an audio file** (.mp3, .wav, .flac, etc.)
3. **Wait for processing** (typically 10-30 seconds)
4. **View your guitar tab!**
5. **Download** the tablature as a .txt file

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js Frontend                     │
│                  (http://localhost:3000)                │
│                                                         │
│  ┌──────────────┐     ┌───────────────┐               │
│  │ File Upload  │────▶│  Processing   │               │
│  │  Component   │     │   Animation   │               │
│  └──────────────┘     └───────────────┘               │
│         │                      │                        │
│         ▼                      ▼                        │
│  ┌────────────────────────────────────┐                │
│  │      Tab Display & Statistics      │                │
│  └────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
                         │
                         │ HTTP POST /api/convert
                         │ (audio file + settings)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│                (http://localhost:8000)                  │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                │
│  │   Extract    │────▶│   Optimize   │                │
│  │ MIDI Notes   │     │  Fingering   │                │
│  │  (librosa,   │     │  (Dijkstra   │                │
│  │ basic-pitch) │     │  Algorithm)  │                │
│  └──────────────┘     └──────────────┘                │
│         │                      │                        │
│         ▼                      ▼                        │
│  ┌────────────────────────────────────┐                │
│  │      Format as Guitar Tab          │                │
│  └────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### Python Backend (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/health` | GET | Health check |
| `/api/tunings` | GET | List available tunings |
| `/api/convert` | POST | Convert audio to tab |
| `/docs` | GET | Interactive API documentation |

### POST /api/convert Parameters

```typescript
{
  file: File              // Audio file (.mp3, .wav, etc.)
  method: string          // 'pyin' or 'basic_pitch'
  tuning: string          // 'standard', 'drop-d', 'drop-c', etc.
  min_duration: number    // Minimum note duration (default: 0.1)
  detailed: boolean       // Include detailed note info (default: false)
  preprocess: boolean     // Preprocess audio (default: false)
}
```

### Response Format

```json
{
  "success": true,
  "tablature": "e|---|---|\nB|---|---|...",
  "stats": {
    "total_notes": 42,
    "pitch_range": {
      "min": "E3",
      "max": "B4"
    },
    "avg_fret_movement": 2.3,
    "avg_string_jumps": 0.8
  }
}
```

---

## Configuration

### Environment Variables

Edit `frontend/axtone/.env.local`:

```env
PYTHON_API_URL=http://localhost:8000
```

### Tuning Options

Available presets:
- `standard` - E-A-D-G-B-E (default)
- `drop-d` - D-A-D-G-B-E
- `drop-c` - C-G-C-F-A-D
- `open-g` - D-G-D-G-B-D
- `dadgad` - D-A-D-G-A-D

### Pitch Detection Methods

- **pyin**: Fast, signal processing based (default)
- **basic_pitch**: Neural network, more accurate but slower

---

## Troubleshooting

### "Failed to connect to processing server"

✅ **Solution**: Make sure Python API is running
```bash
python api.py
```

### "No notes detected in audio file"

✅ **Solutions**:
- Use a clearer audio recording with distinct pitch
- Try the `basic_pitch` method for better accuracy
- Adjust `min_duration` to a lower value (e.g., 0.05)

### CORS errors in browser console

✅ **Solution**: The API is configured for `localhost:3000`. For production, update `api.py`:
```python
allow_origins=[
    "http://localhost:3000",
    "https://your-production-domain.com"  # Add this
]
```

### Port already in use

✅ **Solution**:
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

---

## Tech Stack

### Frontend
- **Framework**: Next.js 14+ (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI
- **Audio Processing**: librosa
- **Pitch Detection**: basic-pitch, PYIN
- **Algorithm**: Custom Dijkstra implementation
- **MIDI**: pretty-midi, mido

---

## Development

### Run Tests

```bash
# Python tests
pytest tests/

# Frontend (if you add tests later)
cd frontend/axtone
npm test
```

### View API Documentation

Start the Python backend and visit:
http://localhost:8000/docs

This provides interactive Swagger UI documentation.

### Hot Reload

Both servers support hot reload:
- Python: Uvicorn auto-reloads on file changes
- Next.js: Fast Refresh on save

---

## Deployment

### Python Backend

Deploy to:
- **Railway**: `railway up`
- **Render**: Connect GitHub repo
- **Google Cloud Run**: `gcloud run deploy`
- **AWS Lambda**: Use Mangum adapter

### Next.js Frontend

Deploy to:
- **Vercel**: `vercel deploy` (recommended)
- **Netlify**: Connect GitHub repo
- **Cloudflare Pages**: Connect GitHub repo

### Environment Variables

Production environment:
1. Update `PYTHON_API_URL` in frontend to your deployed API URL
2. Update `allow_origins` in `api.py` to include your frontend domain

---

## License

See LICENSE file in the project root.

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Happy Tabbing! 🎸**
