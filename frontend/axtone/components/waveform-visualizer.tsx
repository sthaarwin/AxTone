'use client'

import { useEffect, useRef } from 'react'

interface WaveformVisualizerProps {
  isPlaying: boolean
}

export default function WaveformVisualizer({ isPlaying }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#0f172a'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform bars
    const barCount = 60
    const barWidth = width / barCount
    const gap = 2

    for (let i = 0; i < barCount; i++) {
      // Generate random bar heights with variation
      const randomness = Math.sin(i * 0.3 + Date.now() / 1000) * 0.3 + 0.7
      const baseHeight = (Math.sin(i / 10) * 0.5 + 0.5) * randomness
      
      let barHeight = baseHeight * (height - 20)
      
      // Animated playback effect
      if (isPlaying) {
        barHeight *= (0.5 + Math.sin(Date.now() / 100 - i * 0.1) * 0.5)
      }

      const x = i * barWidth + gap / 2
      const y = (height - barHeight) / 2

      // Gradient color
      const gradient = ctx.createLinearGradient(x, y, x, y + barHeight)
      gradient.addColorStop(0, '#22d3ee')
      gradient.addColorStop(1, '#0ea5e9')

      ctx.fillStyle = gradient
      ctx.fillRect(x, y, barWidth - gap, barHeight)
    }

    // Draw center line
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.5)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, height / 2)
    ctx.lineTo(width, height / 2)
    ctx.stroke()
  }, [isPlaying])

  return (
    <canvas
      ref={canvasRef}
      width={400}
      height={150}
      className="w-full rounded-lg bg-slate-950 border border-slate-800"
    />
  )
}
