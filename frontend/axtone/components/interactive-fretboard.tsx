'use client'

import { useEffect, useRef, useState } from 'react'
import { Guitar } from 'lucide-react'

interface InteractiveFretboardProps {
  isPlaying: boolean
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

export default function InteractiveFretboard({ isPlaying }: InteractiveFretboardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [highlightedFret, setHighlightedFret] = useState<{ string: number; fret: number } | null>(null)

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

    // Animate highlighted note when playing
    if (isPlaying) {
      const time = Date.now() / 1000
      const notePattern = [
        { string: 1, fret: 2 },
        { string: 2, fret: 3 },
        { string: 3, fret: 2 },
        { string: 1, fret: 5 },
        { string: 2, fret: 7 },
        { string: 3, fret: 5 },
      ]
      
      const index = Math.floor(time * 2) % notePattern.length
      setHighlightedFret(notePattern[index])
    }
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
          ? '♪ Playing - watch as notes highlight on the fretboard' 
          : 'Press play to see notes highlighted on the fretboard'}
      </p>
    </div>
  )
}
