'use client'

import { useState } from 'react'
import { Play, Pause, Download, FileDown, Music as MusicIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import WaveformVisualizer from './waveform-visualizer'

interface ResultViewProps {
  fileName: string
  isPlaying: boolean
  onPlayingChange: (playing: boolean) => void
  tablature: string
  midiBase64?: string
  stats: {
    total_notes: number
    pitch_range: {
      min: string
      max: string
    }
    avg_fret_movement?: number
    avg_string_jumps?: number
    tuning_used?: string
  }
}

export default function ResultView({ 
  fileName, 
  isPlaying, 
  onPlayingChange,
  tablature,
  stats,
  midiBase64
}: ResultViewProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(tablature)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownloadTab = () => {
    const element = document.createElement('a')
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(tablature))
    element.setAttribute('download', fileName.replace(/\.[^/.]+$/, '') + '_tab.txt')
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  const handleDownloadMidi = () => {
    if (!midiBase64) return
    
    // Convert base64 to blob
    const byteCharacters = atob(midiBase64)
    const byteNumbers = new Array(byteCharacters.length)
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i)
    }
    const byteArray = new Uint8Array(byteNumbers)
    const blob = new Blob([byteArray], { type: 'audio/midi' })
    
    // Download
    const url = window.URL.createObjectURL(blob)
    const element = document.createElement('a')
    element.setAttribute('href', url)
    element.setAttribute('download', fileName.replace(/\.[^/.]+$/, '') + '.mid')
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    window.URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Your Tab</h2>
          <p className="text-slate-400 text-sm mt-1">{fileName}</p>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => onPlayingChange(!isPlaying)}
            className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white border-0"
            size="lg"
          >
            {isPlaying ? (
              <>
                <Pause className="w-4 h-4 mr-2" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Play
              </>
            )}
          </Button>
          <Button
            onClick={handleDownloadTab}
            variant="outline"
            size="lg"
            className="border-slate-700 text-slate-300 hover:bg-slate-800 bg-transparent"
          >
            <FileDown className="w-4 h-4 mr-2" />
            Download Tab
          </Button>
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Stats Panel */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
            <MusicIcon className="w-4 h-4 text-cyan-400" />
            Statistics
          </h3>
          <div className="space-y-3">
            {stats.tuning_used && (
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Tuning</span>
                <span className="text-white font-semibold uppercase">{stats.tuning_used}</span>
              </div>
            )}
            <div className="flex justify-between items-center">
              <span className="text-slate-400 text-sm">Total Notes</span>
              <span className="text-white font-semibold">{stats.total_notes}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-slate-400 text-sm">Pitch Range</span>
              <span className="text-white font-semibold">
                {stats.pitch_range.min} - {stats.pitch_range.max}
              </span>
            </div>
            {stats.avg_fret_movement !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Avg Fret Movement</span>
                <span className="text-white font-semibold">{stats.avg_fret_movement.toFixed(2)}</span>
              </div>
            )}
            {stats.avg_string_jumps !== undefined && (
              <div className="flex justify-between items-center">
                <span className="text-slate-400 text-sm">Avg String Jumps</span>
                <span className="text-white font-semibold">{stats.avg_string_jumps.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>

        {/* Guitar Tab */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <MusicIcon className="w-4 h-4 text-cyan-400" />
              Tablature Preview
            </h3>
            <Button
              onClick={handleCopy}
              variant="ghost"
              size="sm"
              className="text-slate-400 hover:text-slate-200 hover:bg-slate-800 text-xs"
            >
              {copied ? 'Copied!' : 'Copy'}
            </Button>
          </div>
          <div className="font-mono text-xs text-slate-300 leading-relaxed overflow-x-auto whitespace-pre">
            {tablature}
          </div>
        </div>
      </div>

      {/* Download Options */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-slate-200 mb-4">Export Options</h3>
        <div className="flex flex-col sm:flex-row gap-3">
          <Button
            onClick={handleDownloadTab}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100"
          >
            <FileDown className="w-4 h-4 mr-2" />
            Download Tab (.txt)
          </Button>
          <Button
            onClick={handleDownloadMidi}
            disabled={!midiBase64}
            className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4 mr-2" />
            Download MIDI (.mid)
          </Button>
        </div>
        {!midiBase64 && (
          <p className="text-xs text-slate-500 mt-2">MIDI export is unavailable for this conversion</p>
        )}
      </div>
    </div>
  )
}
