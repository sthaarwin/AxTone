'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { Guitar } from 'lucide-react'

interface Note {
  midi: number
  onset: number
  offset: number
  string: number
  fret: number
}

interface InteractiveFretboardProps {
  isPlaying: boolean
  onPlayingChange: (playing: boolean) => void
  notes: Note[]
  audioUrl?: string | null
}

const STRING_META = [
  { name: 'E', color: '#ef4444' },  // 0 = low E  (top of canvas)
  { name: 'A', color: '#f97316' },
  { name: 'D', color: '#eab308' },
  { name: 'G', color: '#22c55e' },
  { name: 'B', color: '#06b6d4' },
  { name: 'e', color: '#8b5cf6' },  // 5 = high e (bottom of canvas)
]

const FRETS = 22
const MARKER_FRETS = [3, 5, 7, 9, 12, 15, 17, 19, 21]

function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12)
}

interface ActivePluck {
  source: AudioBufferSourceNode
  gain: GainNode
  baseFreq: number
}

/** Karplus-Strong plucked-string synthesis. Sounds like a real guitar. */
function synthesizeString(ctx: AudioContext, frequency: number, duration: number, volume: number = 0.8): ActivePluck {
  const sr = ctx.sampleRate
  const N = Math.max(2, Math.round(sr / frequency))
  const totalSamples = Math.min(sr * 4, Math.max(sr, Math.ceil(sr * (duration + 1.5))))

  const buf = ctx.createBuffer(1, totalSamples, sr)
  const data = buf.getChannelData(0)

  // Noise excitation: one period of white noise (the "pick" attack)
  for (let i = 0; i < N; i++) data[i] = Math.random() * 2 - 1

  // Karplus-Strong feedback: 2-point averaging = lowpass + delay = decaying tone
  for (let i = N; i < totalSamples; i++) {
    data[i] = 0.4985 * (data[i - N] + (i - N - 1 >= 0 ? data[i - N - 1] : 0))
  }

  const source = ctx.createBufferSource()
  source.buffer = buf

  const gain = ctx.createGain()
  const now = ctx.currentTime
  const baseGain = 0.5 * volume // Adjust base volume with slider
  gain.gain.setValueAtTime(baseGain, now)
  gain.gain.setValueAtTime(baseGain, now + Math.max(0.01, duration * 0.7))
  gain.gain.exponentialRampToValueAtTime(0.001, now + duration + 0.8)

  source.connect(gain)
  gain.connect(ctx.destination)
  source.start(now)
  source.stop(now + duration + 1.5)
  
  return { source, gain, baseFreq: frequency }
}

export default function InteractiveFretboard({
  isPlaying,
  onPlayingChange,
  notes,
  audioUrl
}: InteractiveFretboardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const audioElRef = useRef<HTMLAudioElement>(null)
  const [highlightedNote, setHighlightedNote] = useState<Note | null>(null)
  const [playOriginal, setPlayOriginal] = useState(false)
  const [synthVolume, setSynthVolume] = useState(0.8)
  
  const audioCtxRef = useRef<AudioContext | null>(null)
  const startAudioTimeRef = useRef<number>(0)
  const lastPlayedIdxRef = useRef<number>(-1)
  const rafRef = useRef<number>(0)
  const activeStringsRef = useRef<(ActivePluck | null)[]>(new Array(6).fill(null))

  // ── Canvas drawing ──────────────────────────────────────────────────────────
  const drawFretboard = useCallback((highlighted: Note | null) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    const padL = 52
    const padR = 16
    const padT = 28
    const padB = 16

    const nStrings = STRING_META.length
    const strGap = (H - padT - padB) / (nStrings - 1)
    const fretGap = (W - padL - padR) / FRETS

    // Background
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, W, H)

    // Position dot markers (background colour, drawn before strings)
    ctx.fillStyle = '#1e293b'
    MARKER_FRETS.forEach(f => {
      const x = padL + (f - 0.5) * fretGap
      const midY = padT + strGap * (nStrings - 1) / 2
      if (f === 12) {
        ctx.beginPath(); ctx.arc(x, midY - strGap * 0.65, 5, 0, Math.PI * 2); ctx.fill()
        ctx.beginPath(); ctx.arc(x, midY + strGap * 0.65, 5, 0, Math.PI * 2); ctx.fill()
      } else {
        ctx.beginPath(); ctx.arc(x, midY, 6, 0, Math.PI * 2); ctx.fill()
      }
      // Fret number label
      ctx.fillStyle = '#475569'
      ctx.font = '9px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(String(f), x, padT - 10)
      ctx.fillStyle = '#1e293b'
    })

    // Fret bars
    for (let f = 0; f <= FRETS; f++) {
      const x = padL + f * fretGap
      ctx.strokeStyle = f === 0 ? '#94a3b8' : '#334155'
      ctx.lineWidth = f === 0 ? 3 : 1
      ctx.beginPath()
      ctx.moveTo(x, padT)
      ctx.lineTo(x, H - padB)
      ctx.stroke()
    }

    // String lines & highlight
    STRING_META.forEach((str, si) => {
      const y = padT + si * strGap
      // String thickness: low E (~2.2px) → high e (~0.6px)
      const thickness = 2.2 - si * 0.27
      const isHighlighted = highlighted !== null && highlighted.string === si

      ctx.strokeStyle = isHighlighted ? str.color : `${str.color}55`
      ctx.lineWidth = thickness
      ctx.beginPath()
      ctx.moveTo(padL, y)
      ctx.lineTo(W - padR, y)
      ctx.stroke()

      // String label
      ctx.fillStyle = '#94a3b8'
      ctx.font = `bold 12px monospace`
      ctx.textAlign = 'right'
      ctx.fillText(str.name, padL - 6, y + 4)

      // Highlighted note circle
      if (isHighlighted && highlighted) {
        const fret = highlighted.fret
        // Open string: draw on the nut side; otherwise midpoint between fret-1 and fret
        const cx = fret === 0 ? padL - 14 : padL + (fret - 0.5) * fretGap
        const cy = y

        // Glow
        ctx.shadowColor = str.color
        ctx.shadowBlur = 20
        ctx.fillStyle = str.color
        ctx.beginPath()
        ctx.arc(cx, cy, 11, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0

        // Fret number label inside dot
        ctx.fillStyle = '#0f172a'
        ctx.font = 'bold 9px monospace'
        ctx.textAlign = 'center'
        ctx.fillText(String(fret), cx, cy + 3)
      }
    })
  }, [])

  // Redraw whenever highlight changes
  useEffect(() => { drawFretboard(highlightedNote) }, [highlightedNote, drawFretboard])

  // Manage original audio toggle during playback
  useEffect(() => {
    if (!audioElRef.current) return
    if (!playOriginal && !audioElRef.current.paused) {
      audioElRef.current.pause()
    } else if (playOriginal && isPlaying && audioCtxRef.current) {
      // Sync it up when toggled on mid-playback
      const elapsed = audioCtxRef.current.currentTime - startAudioTimeRef.current
      if (notes.length > 0) {
         audioElRef.current.currentTime = notes[0].onset + elapsed
         audioElRef.current.play().catch(() => {})
      }
    }
  }, [playOriginal, isPlaying, notes])

  // ── Playback loop ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isPlaying || notes.length === 0) {
      cancelAnimationFrame(rafRef.current)
      setHighlightedNote(null)
      lastPlayedIdxRef.current = -1
      // Stop all active strings
      activeStringsRef.current.forEach(active => {
        if (active) active.source.stop()
      })
      activeStringsRef.current = new Array(6).fill(null)
      // Stop original audio
      if (audioElRef.current) {
        audioElRef.current.pause()
      }
      return
    }

    // Init / resume AudioContext
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext()
    const actx = audioCtxRef.current
    if (actx.state === 'suspended') actx.resume()

    // Align AudioContext time so the first note starts immediately
    startAudioTimeRef.current = actx.currentTime - notes[0].onset
    lastPlayedIdxRef.current = -1

    // Play original audio synced up
    if (playOriginal && audioElRef.current) {
      audioElRef.current.currentTime = notes[0].onset
      audioElRef.current.play()
    }

    const tick = () => {
      if (!audioCtxRef.current) return
      const elapsed = audioCtxRef.current.currentTime - startAudioTimeRef.current

      // Advance monotonically through notes, playing each one exactly once
      let idx = lastPlayedIdxRef.current
      while (idx + 1 < notes.length && notes[idx + 1].onset <= elapsed + 0.02) {
        idx++
        const note = notes[idx]
        
        let isSlide = false
        if (idx > 0) {
          const prevNote = notes[idx - 1]
          if (note.string === prevNote.string && (note.onset - prevNote.offset) < 0.05) {
            isSlide = true
          }
        }

        const active = activeStringsRef.current[note.string]
        
        if (isSlide && active) {
          // Slide existing pluck! Sweep playback rate to match target pitch
          const targetFreq = midiToFrequency(note.midi)
          const ratio = targetFreq / active.baseFreq
          const now = actx.currentTime
          
          active.source.playbackRate.setValueAtTime(active.source.playbackRate.value, now)
          active.source.playbackRate.linearRampToValueAtTime(ratio, now + 0.08) // 80ms glide
          
          // Extend decay
          const newDuration = Math.max(0.08, note.offset - note.onset)
          active.gain.gain.cancelScheduledValues(now)
          active.gain.gain.setValueAtTime(active.gain.gain.value, now)
          active.gain.gain.setValueAtTime(0.3 * synthVolume, now + 0.05) // bump volume slightly for slide impact
          active.gain.gain.exponentialRampToValueAtTime(0.001, now + newDuration + 0.8)
        } else {
          // Normal pluck
          if (active) active.source.stop(actx.currentTime) // mute previous note on this string
          activeStringsRef.current[note.string] = synthesizeString(actx, midiToFrequency(note.midi), Math.max(0.08, note.offset - note.onset), synthVolume)
        }
        
        setHighlightedNote(note)
        lastPlayedIdxRef.current = idx
      }

      // Auto-stop when song ends
      if (elapsed > notes[notes.length - 1].offset + 0.8) {
        onPlayingChange(false)
        setHighlightedNote(null)
        lastPlayedIdxRef.current = -1
        return  // don't request next frame
      }

      // Clear highlight when current note has ended
      const cur = lastPlayedIdxRef.current >= 0 ? notes[lastPlayedIdxRef.current] : null
      if (cur && elapsed > cur.offset + 0.05) setHighlightedNote(null)

      rafRef.current = requestAnimationFrame(tick)
    }

    tick()
    return () => cancelAnimationFrame(rafRef.current)
  }, [isPlaying, notes, onPlayingChange])

  // ── JSX ─────────────────────────────────────────────────────────────────────
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
      <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Guitar className="w-4 h-4 text-cyan-400" />
        Interactive Fretboard
      </h3>
      {audioUrl && <audio ref={audioElRef} src={audioUrl} className="hidden" />}
      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={920}
          height={220}
          className="w-full rounded-lg bg-slate-950 border border-slate-800"
        />
      </div>
      
      <div className="flex flex-wrap items-center justify-between gap-4 mt-4 bg-slate-950/50 p-3 rounded-lg border border-slate-800/50">
        <p className="text-xs text-slate-400 font-medium">
          {isPlaying
            ? '♪ Playing'
            : notes.length > 0
              ? `${notes.length} notes ready`
              : 'Press play above'}
        </p>

        <div className="flex items-center gap-6">
          {/* Synth Volume Slider */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">Guitar Synth:</span>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={synthVolume}
              onChange={(e) => setSynthVolume(parseFloat(e.target.value))}
              className="w-20 accent-cyan-500 h-1 bg-slate-700 rounded-full appearance-none"
            />
          </div>

          {/* Original Audio Toggle */}
          {audioUrl && (
            <label className="flex items-center gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={playOriginal}
                onChange={(e) => setPlayOriginal(e.target.checked)}
                className="w-4 h-4 rounded border-slate-700 bg-slate-800 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900"
              />
              <span className="text-xs text-slate-300 font-medium">Play Original Audio</span>
            </label>
          )}
        </div>
      </div>
    </div>
  )
}
