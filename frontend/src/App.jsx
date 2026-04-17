import { useState, useRef } from 'react'
import './App.css'

const REASON_LABELS = {
  hate_speech: 'Hate Speech',
  explicit_content: 'Explicit / Vulgar Language',
  sexual_innuendo: 'Sexual Innuendo / Double Meaning',
  offensive_language: 'Offensive Language / Slurs',
  sarcasm_or_targeted_speech: 'Sarcasm / Targeted Speech',
  potentially_targeted_hate: 'Targeted Speech (High Risk)',
  potentially_targeted: 'Potentially Targeted',
  safe: 'Clean'
}

function SignalPill({ label, active, color }) {
  return (
    <span className={`signal-pill ${active ? 'signal-active' : 'signal-inactive'}`}
      style={active ? { borderColor: color, color } : {}}>
      {active ? '⚠ ' : '✓ '}{label}
    </span>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setResult(null)
      setError(null)
    }
  }

  const handleScan = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: formData })
      if (!response.ok) throw new Error('Prediction failed')
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      setFile(droppedFile)
      setPreview(URL.createObjectURL(droppedFile))
      setResult(null)
      setError(null)
    }
  }

  const isHateful = result?.classification === 'HATEFUL'
  const accentColor = isHateful ? 'var(--accent-hate)' : 'var(--accent-safe)'

  const textReason = result?.text_analysis?.reason || ''
  const textReasonLabel = REASON_LABELS[textReason.split(':')[0]] || textReason

  return (
    <div className="container">
      <header>
        <h1>MemeModerator AI</h1>
        <p className="subtitle">Multimodal detection — hate speech, sarcasm, double meaning &amp; visual content</p>
      </header>

      <main className="main-card">
        {/* ── LEFT: Upload ── */}
        <div className="upload-section">
          <div
            className={`dropzone ${file ? 'active' : ''}`}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current.click()}
          >
            <span className="upload-icon">🖼️</span>
            <p>{file ? file.name : 'Drag & drop meme or click to browse'}</p>
            <input type="file" className="file-input" ref={fileInputRef}
              onChange={handleFileChange} accept="image/*" />
          </div>

          {preview && (
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-img" />
            </div>
          )}

          <button className="btn-scan" onClick={handleScan} disabled={!file || loading}>
            {loading ? (
              <div className="loading">
                <div className="spinner"></div>
                Analyzing Content...
              </div>
            ) : 'Analyze Meme'}
          </button>

          {error && <p style={{ color: '#ef4444', textAlign: 'center' }}>{error}</p>}
        </div>

        {/* ── RIGHT: Results ── */}
        <div className={`results-section ${result || loading ? 'visible' : ''}`}>
          {!result && !loading && (
            <div className="loading" style={{ opacity: 0.4 }}>
              Waiting for analysis...
            </div>
          )}

          {loading && (
            <div className="loading">
              <div className="spinner"></div>
              <span>Scanning image &amp; text...</span>
            </div>
          )}

          {result && (
            <>
              {/* ── Verdict badge ── */}
              <div>
                <span className={`result-badge ${isHateful ? 'badge-hate' : 'badge-safe'}`}>
                  {isHateful ? '🚫 Harmful Content Detected' : '✅ Safe Content'}
                </span>
              </div>

              {/* ── Confidence bar ── */}
              <div className="score-container">
                <div className="score-header">
                  <span className="score-label">Confidence Score</span>
                  <span className="score-value">{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="progress-bar-bg">
                  <div className="progress-bar-fill"
                    style={{ width: `${result.confidence * 100}%`, backgroundColor: accentColor }} />
                </div>
              </div>

              {/* ── Analysis breakdown ── */}
              <div className="analysis-grid">
                {/* Visual */}
                <div className={`analysis-card ${result.visual_analysis?.harmful ? 'card-harm' : 'card-ok'}`}>
                  <div className="analysis-card-header">
                    <span className="analysis-icon">{result.visual_analysis?.harmful ? '👁️‍🗨️' : '🖼️'}</span>
                    <span className="analysis-label">Visual Analysis</span>
                    <span className={`mini-badge ${result.visual_analysis?.harmful ? 'mini-hate' : 'mini-safe'}`}>
                      {result.visual_analysis?.harmful ? 'FLAGGED' : 'CLEAR'}
                    </span>
                  </div>
                  <p className="analysis-detail">
                    {result.visual_analysis?.harmful
                      ? result.visual_analysis.category
                      : 'No harmful imagery detected'}
                  </p>
                  <div className="mini-bar-bg">
                    <div className="mini-bar-fill"
                      style={{
                        width: `${(result.visual_analysis?.score || 0) * 100}%`,
                        backgroundColor: result.visual_analysis?.harmful ? 'var(--accent-hate)' : 'var(--accent-safe)'
                      }} />
                  </div>
                </div>

                {/* Text */}
                <div className={`analysis-card ${result.text_analysis?.harmful ? 'card-harm' : 'card-ok'}`}>
                  <div className="analysis-card-header">
                    <span className="analysis-icon">{result.text_analysis?.harmful ? '💬⚠️' : '💬'}</span>
                    <span className="analysis-label">Text Analysis</span>
                    <span className={`mini-badge ${result.text_analysis?.harmful ? 'mini-hate' : 'mini-safe'}`}>
                      {result.text_analysis?.harmful ? 'FLAGGED' : 'CLEAR'}
                    </span>
                  </div>
                  <p className="analysis-detail">{textReasonLabel}</p>
                  <div className="mini-bar-bg">
                    <div className="mini-bar-fill"
                      style={{
                        width: `${(result.text_analysis?.score || 0) * 100}%`,
                        backgroundColor: result.text_analysis?.harmful ? 'var(--accent-hate)' : 'var(--accent-safe)'
                      }} />
                  </div>
                </div>
              </div>

              {/* ── Extracted text ── */}
              <div className="ocr-box">
                <span className="ocr-title">EXTRACTED TEXT</span>
                <p className="ocr-content">
                  {result.extracted_text || 'No text detected in image.'}
                </p>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
