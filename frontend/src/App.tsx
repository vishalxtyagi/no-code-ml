import React, { useState } from 'react';
import {
  getDownloadUrl,
  PipelineResult,
  runPipeline,
  uploadDataset,
  UploadResponse,
} from './api/client';
import FeatureChart from './components/FeatureChart';
import ModelTable from './components/ModelTable';
import UploadZone from './components/UploadZone';

const card: React.CSSProperties = {
  background: '#fff',
  border: '1px solid #e5e7eb',
  borderRadius: 12,
  padding: 24,
  marginBottom: 24,
  boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
};

const sectionTitle: React.CSSProperties = {
  fontSize: 18,
  fontWeight: 700,
  marginBottom: 16,
  color: '#111827',
};

export default function App() {
  const [uploadLoading, setUploadLoading] = useState(false);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadInfo, setUploadInfo] = useState<UploadResponse | null>(null);
  const [targetCol, setTargetCol] = useState<string>('');
  const [tune, setTune] = useState(true);
  const [result, setResult] = useState<PipelineResult | null>(null);

  const handleUpload = async (file: File) => {
    setError(null);
    setResult(null);
    setUploadInfo(null);
    setUploadLoading(true);
    try {
      const info = await uploadDataset(file);
      setUploadInfo(info);
      setTargetCol(info.suggested_target);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Upload failed');
    } finally {
      setUploadLoading(false);
    }
  };

  const handleRun = async () => {
    if (!uploadInfo) return;
    setError(null);
    setPipelineLoading(true);
    try {
      const res = await runPipeline(uploadInfo.job_id, targetCol, tune);
      setResult(res);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Pipeline failed');
    } finally {
      setPipelineLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f1f5f9', fontFamily: 'Inter, system-ui, sans-serif' }}>
      <header style={{ background: '#4f46e5', color: '#fff', padding: '20px 32px', boxShadow: '0 2px 8px rgba(0,0,0,0.15)' }}>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>⚡ AutoML Platform</h1>
        <p style={{ margin: '4px 0 0', opacity: 0.85, fontSize: 14 }}>
          Upload a CSV → instantly get the best model + insights
        </p>
      </header>

      <main style={{ maxWidth: 960, margin: '0 auto', padding: '32px 16px' }}>
        <div style={card}>
          <div style={sectionTitle}>1. Upload Dataset</div>
          <UploadZone onUpload={handleUpload} loading={uploadLoading} />
        </div>

        {error && (
          <div style={{ background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 8, padding: '12px 16px', marginBottom: 24, color: '#b91c1c' }}>
            ❌ {error}
          </div>
        )}

        {uploadInfo && !result && (
          <div style={card}>
            <div style={sectionTitle}>2. Configure Pipeline</div>

            <div style={{ marginBottom: 16 }}>
              <p style={{ margin: '0 0 8px', fontWeight: 600 }}>
                Detected problem type:{' '}
                <span style={{ color: '#4f46e5' }}>{uploadInfo.detected_problem_type.toUpperCase()}</span>
              </p>
              <p style={{ margin: 0, color: '#6b7280', fontSize: 13 }}>
                Dataset: {uploadInfo.profile.shape.rows.toLocaleString()} rows × {uploadInfo.profile.shape.columns} columns
              </p>
            </div>

            <label style={{ display: 'block', marginBottom: 16 }}>
              <span style={{ fontWeight: 600, display: 'block', marginBottom: 6 }}>Target column (to predict):</span>
              <select
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                style={{ padding: '8px 12px', borderRadius: 8, border: '1px solid #d1d5db', fontSize: 14, width: '100%', maxWidth: 360 }}
              >
                {uploadInfo.columns.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20, cursor: 'pointer' }}>
              <input type="checkbox" checked={tune} onChange={(e) => setTune(e.target.checked)} />
              <span style={{ fontSize: 14 }}>Enable Optuna hyperparameter tuning (recommended)</span>
            </label>

            {uploadInfo.preprocessing_suggestions.length > 0 && (
              <div style={{ background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 8, padding: '12px 16px', marginBottom: 20 }}>
                <strong style={{ fontSize: 13 }}>🔍 Preprocessing plan:</strong>
                <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 13, color: '#166534' }}>
                  {uploadInfo.preprocessing_suggestions.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={pipelineLoading}
              style={{
                background: pipelineLoading ? '#9ca3af' : '#4f46e5',
                color: '#fff',
                border: 'none',
                borderRadius: 8,
                padding: '12px 28px',
                fontSize: 15,
                fontWeight: 700,
                cursor: pipelineLoading ? 'not-allowed' : 'pointer',
              }}
            >
              {pipelineLoading ? '⏳ Running pipeline…' : '🚀 Run AutoML Pipeline'}
            </button>
            {pipelineLoading && (
              <p style={{ color: '#6b7280', fontSize: 13, marginTop: 8 }}>
                Training models and tuning hyperparameters. This may take a moment…
              </p>
            )}
          </div>
        )}

        {result && (
          <>
            <div style={{ ...card, borderLeft: '4px solid #4f46e5' }}>
              <div style={sectionTitle}>🏆 Best Model: {result.best_model}</div>
              <p style={{ margin: '0 0 12px', color: '#374151', fontSize: 14 }}>
                <strong>Problem type:</strong> {result.problem_type} &nbsp;|&nbsp;
                <strong>Target:</strong> {result.target_col} &nbsp;|&nbsp;
                <strong>Dataset:</strong> {result.dataset_shape.rows.toLocaleString()} rows × {result.dataset_shape.columns} cols
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12 }}>
                {Object.entries(result.best_model_scores).map(([k, v]) => (
                  <div key={k} style={{ background: '#eef2ff', borderRadius: 8, padding: '8px 16px', minWidth: 100, textAlign: 'center' }}>
                    <div style={{ fontSize: 11, color: '#6366f1', textTransform: 'uppercase', letterSpacing: 1 }}>{k}</div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: '#312e81' }}>
                      {['accuracy', 'f1_weighted', 'precision', 'recall'].includes(k)
                        ? `${(v * 100).toFixed(1)}%`
                        : v.toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 20 }}>
                <a
                  href={getDownloadUrl(result.job_id)}
                  download
                  style={{ display: 'inline-block', background: '#059669', color: '#fff', borderRadius: 8, padding: '10px 20px', fontSize: 14, fontWeight: 600, textDecoration: 'none' }}
                >
                  ⬇️ Download Trained Model (.pkl)
                </a>
              </div>
            </div>

            <div style={card}>
              <div style={sectionTitle}>💡 Plain-English Explanation</div>
              <p style={{ margin: 0, lineHeight: 1.7, color: '#374151', fontSize: 15, whiteSpace: 'pre-wrap' }}>
                {result.explanation.replace(/\*\*/g, '')}
              </p>
            </div>

            <div style={card}>
              <div style={sectionTitle}>📊 Model Comparison</div>
              <ModelTable
                rows={result.comparison_table}
                bestModel={result.best_model}
                problemType={result.problem_type}
              />
            </div>

            {result.feature_importances.length > 0 && (
              <div style={card}>
                <div style={sectionTitle}>🔑 Feature Importance</div>
                <FeatureChart importances={result.feature_importances} />
              </div>
            )}

            <div style={{ textAlign: 'center', marginTop: 8 }}>
              <button
                onClick={() => { setResult(null); setUploadInfo(null); }}
                style={{ background: 'transparent', border: '1px solid #d1d5db', borderRadius: 8, padding: '10px 24px', cursor: 'pointer', fontSize: 14, color: '#374151' }}
              >
                ↩ Analyze another dataset
              </button>
            </div>
          </>
        )}
      </main>
      <style>{`
        * { box-sizing: border-box; }
        body { margin: 0; }
      `}</style>
    </div>
  );
}
