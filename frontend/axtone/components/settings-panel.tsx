'use client'

import { Settings, Music } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface SettingsPanelProps {
  tuning: string
  method: string
  fingering: string
  onTuningChange: (tuning: string) => void
  onMethodChange: (method: string) => void
  onFingeringChange: (fingering: string) => void
}

const TUNINGS = [
  { value: 'standard', label: 'Standard (E-A-D-G-B-E)' },
  { value: 'drop-d', label: 'Drop D (D-A-D-G-B-E)' },
  { value: 'drop-c', label: 'Drop C (C-G-C-F-A-D)' },
  { value: 'open-g', label: 'Open G (D-G-D-G-B-D)' },
  { value: 'dadgad', label: 'DADGAD (D-A-D-G-A-D)' },
]

const METHODS = [
  { value: 'pyin', label: 'PYIN', description: 'Fast & accurate signal processing' },
]

const FINGERINGS = [
  { value: 'dijkstra', label: 'Smart (Dijkstra)', description: 'Ergonomic hand-position routing' },
  { value: 'naive', label: 'Naive', description: 'Just pick the lowest possible fret' },
]

export default function SettingsPanel({ 
  tuning, 
  method, 
  fingering,
  onTuningChange, 
  onMethodChange,
  onFingeringChange
}: SettingsPanelProps) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 backdrop-blur-sm">
      <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <Settings className="w-4 h-4 text-cyan-400" />
        Conversion Settings
      </h3>
      
      <div className="space-y-5">
        {/* Tuning Selector */}
        <div className="space-y-2">
          <label className="text-sm text-slate-400">Guitar Tuning</label>
          <select 
            value={tuning}
            onChange={(e) => onTuningChange(e.target.value)}
            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
          >
            {TUNINGS.map(t => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        {/* Method Selector */}
        <div className="space-y-2">
          <label className="text-sm text-slate-400">Detection Method</label>
          <div className="space-y-2">
            {METHODS.map(m => (
              <label 
                key={m.value}
                className={`flex items-start p-3 rounded-lg border cursor-pointer transition-all ${
                  method === m.value
                    ? 'border-cyan-500 bg-cyan-500/10'
                    : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                }`}
              >
                <input
                  type="radio"
                  name="method"
                  value={m.value}
                  checked={method === m.value}
                  onChange={(e) => onMethodChange(e.target.value)}
                  className="mt-1 mr-3 accent-cyan-500"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium text-slate-200">{m.label}</div>
                  <div className="text-xs text-slate-400">{m.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Fingering Selector */}
        <div className="space-y-2">
          <label className="text-sm text-slate-400">Fingering Optimization</label>
          <div className="space-y-2">
            {FINGERINGS.map(f => (
              <label 
                key={f.value}
                className={`flex items-start p-3 rounded-lg border cursor-pointer transition-all ${
                  fingering === f.value
                    ? 'border-purple-500 bg-purple-500/10'
                    : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                }`}
              >
                <input
                  type="radio"
                  name="fingering"
                  value={f.value}
                  checked={fingering === f.value}
                  onChange={(e) => onFingeringChange(e.target.value)}
                  className="mt-1 mr-3 accent-purple-500"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium text-slate-200">{f.label}</div>
                  <div className="text-xs text-slate-400">{f.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Info */}
        <div className="mt-4 p-3 rounded-lg bg-slate-800/50 border border-slate-700">
          <p className="text-xs text-slate-400">
            <Music className="w-3 h-3 inline mr-1" />
            Settings apply to the next conversion
          </p>
        </div>
      </div>
    </div>
  )
}
