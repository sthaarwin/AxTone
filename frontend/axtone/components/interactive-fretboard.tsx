'use client'

import { useEffect, useRef, useState } from 'react'
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
  notes: Note[]
}

const STRINGS = [
  { name: 'E', color: '#ef4444', frequency: 82.41 },
  { name: 'A', color: '#f97316', frequency: 110.0 },
  { name: 'D', color: '#eab308', frequency: 146.83 },
  { name: 'G', color: '#22c55e', frequency: 196.0 },
  { name: 'B', color: '#06b6d4', frequency: 246.94 },
  { name: 'e', color: '#8b5cf6', frequency: 329.63 },
]

const FRETS = 24

// Convert MIDI note number to frequency
function midiToFrequency(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12)
}

export default function InteractiveFretboard({ isPlaying, notes }: InteractiveFretboardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [highlightedFret, setHighlightedFret] = useState<{ string: number; fret: number } | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const startTimeRef = useRef<number>(0)
  const animationFrameRef = useRef<number>(0)

  // Play a note with Web Audio API
  const playNote = (frequency: number, duration: number) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext()
    }

    const ctx = audioContextRef.current
    const now = ctx.currentTime

    // Create oscillator for the note
    const oscillator = ctx.createOscillator()
    const gainNode = ctx.createGain()

    oscillator.connect(gainNode)
    gainNode.connect(ctx.destination)

    // Guitar-like sound using triangle wave
    oscillator.type = 'triangle'
    oscillator.frequency.setValueAtTime(frequency, now)

    // ADSR envelope for guitar-like attack
    gainNode.gain.setValueAtTime(0, now)
    gainNode.gain.linearRampToValueAtTime(0.3, now + 0.01) // Quick attack
    gainNode.gain.exponentialRampToValueAtTime(0.1, now + 0.1) // Decay
    gainNode.gain.exponentialRampToValueAtTime(0.05, now + duration) // Sustain
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + duration + 0.1) // Release

    oscillator.start(now)
    oscillator.stop(now + duration + 0.1)
  }

  // Animation and playback loop
  useEffect(() => {
    if (!isPlaying || notes.length === 0) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      setHighlightedFret(null)
      return
    }

    startTimeRef.current = Date.now() / 1000

    const animate = () => {
      const currentTime = Date.now() / 1000 - startTimeRef.current

      // Find the note that should be playing now
      const currentNote = notes.find(
        note => currentTime >= note.onset && currentTime < note.offset
      )

      if (currentNote) {
        setHighlightedFret({ string: currentNote.string, fret: currentNote.fret })
        
        // Play the note sound (only when transitioning to a new note)
        const prevNote = notes.find(
          note => currentTime - 0.05 >= note.onset && currentTime - 0.05 < note.offset
        )
        
        if (!prevNote || prevNote !== currentNote) {
          const frequency = midiToFrequency(currentNote.midi)
          const duration = currentNote.offset - currentNote.onset
          playNote(frequency, Math.min(duration, 1))
        }
      } else if (currentTime > notes[notes.length - 1].offset) {
        // Song finished, loop back
        startTimeRef.current = Date.now() / 1000
      }

      animationFrameRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }
  }, [isPlaying, notes])

  // Draw fretboard
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const padding = 40
    const width = canvas.width
    const height = canvas.height
    const stringSpacing = (height - 2 * padding) / (STRINGS.length - 1)
    const fretSpacing = (width - 2 * padding) / FRETS

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw frets
    ctx.strokeStyle = '#334155'
    ctx.lineWidth = 1

    for (let i = 0; i <= FRETS; i++) {
      const x = padding + i * fretSpacing
      ctx.beginPath()
      ctx.moveTo(x, padding)
      ctx.lineTo(x, height - padding)
      ctx.stroke()
    }

    // Draw strings
    STRINGS.forEach((string, stringIndex) => {
      const y = padding + stringIndex * stringSpacing

      // String line
      ctx.strokeStyle = string.color
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(padding, y)
      ctx.lineTo(width - padding, y)
      ctx.stroke()

      // String label
      ctx.fillStyle = '#cbd5e1'
      ctx.font = '14px monospace'
      ctx.textAlign = 'right'
      ctx.fillText(string.name, padding - 15, y + 5)

      // Draw fret dots
      ctx.fillStyle = '#64748b'
      for (let fret = 0; fret <= FRETS; fret++) {
        const x = padding + fret * fretSpacing
        const dotSize = 3

        // Highlight if this is the current playing note
        if (
          isPlaying &&
          highlightedFret &&
          stringIndex === highlightedFret.string &&
          fret === highlightedFret.fret
        ) {
          ctx.fillStyle = string.color
          ctx.fillRect(x - 5, y - 5, 10, 10)
        }
      }
    })

  }, [isPlaying, highlightedFret])

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
      <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Guitar className="w-4 h-4 text-cyan-400" />
        Interactive Fretboard
      </h3>
      <div className="overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={800}
          height={280}
          className="w-full rounded-lg bg-slate-950 border border-slate-800 min-w-full"
        />
      </div>
      <p className="text-xs text-slate-500 mt-3">
        {isPlaying 
          ? notes.length > 0 
            ? '♪ Playing with sound - watch and listen!' 
            : '♪ Playing...' 
          : notes.length > 0
            ? `Ready to play ${notes.length} notes. Press play to hear them!`
            : 'Press play to see notes highlighted on the fretboard'}
      </p>
    </div>
  )
}
