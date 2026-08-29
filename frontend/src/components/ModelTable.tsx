import React from 'react';
import { ModelScore } from '../api/client';

interface Props {
  rows: ModelScore[];
  bestModel: string;
  problemType: 'classification' | 'regression';
}

const MetricBadge: React.FC<{ value: number | undefined; label: string }> = ({ value, label }) => {
  if (value === undefined) return null;
  return (
    <span style={{ marginRight: 8 }}>
      <span style={{ color: '#6b7280', fontSize: 12 }}>{label}: </span>
      <strong>{(value * (label === 'RMSE' || label === 'MAE' || label === 'MSE' ? 1 : 100)).toFixed(
        label === 'RMSE' || label === 'MAE' || label === 'MSE' ? 4 : 1
      )}{label === 'RMSE' || label === 'MAE' || label === 'MSE' ? '' : '%'}</strong>
    </span>
  );
};

const ModelTable: React.FC<Props> = ({ rows, bestModel, problemType }) => {
  const isClassification = problemType === 'classification';

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
        <thead>
          <tr style={{ background: '#f3f4f6' }}>
            <th style={thStyle}>Model</th>
            {isClassification ? (
              <>
                <th style={thStyle}>Accuracy</th>
                <th style={thStyle}>F1 (weighted)</th>
                <th style={thStyle}>Precision</th>
                <th style={thStyle}>Recall</th>
              </>
            ) : (
              <>
                <th style={thStyle}>RMSE</th>
                <th style={thStyle}>MAE</th>
                <th style={thStyle}>R²</th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={row.model}
              style={{
                background: row.model === bestModel ? '#eef2ff' : i % 2 === 0 ? '#fff' : '#f9fafb',
                fontWeight: row.model === bestModel ? 700 : 400,
              }}
            >
              <td style={tdStyle}>
                {row.model === bestModel && <span style={{ marginRight: 6 }}>🏆</span>}
                {row.model}
              </td>
              {isClassification ? (
                <>
                  <td style={tdStyle}>{fmtPct(row.accuracy)}</td>
                  <td style={tdStyle}>{fmtPct(row.f1_weighted)}</td>
                  <td style={tdStyle}>{fmtPct(row.precision)}</td>
                  <td style={tdStyle}>{fmtPct(row.recall)}</td>
                </>
              ) : (
                <>
                  <td style={tdStyle}>{fmtNum(row.rmse)}</td>
                  <td style={tdStyle}>{fmtNum(row.mae)}</td>
                  <td style={tdStyle}>{fmtNum(row.r2)}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const thStyle: React.CSSProperties = {
  padding: '10px 14px',
  textAlign: 'left',
  fontWeight: 600,
  borderBottom: '2px solid #e5e7eb',
};

const tdStyle: React.CSSProperties = {
  padding: '10px 14px',
  borderBottom: '1px solid #e5e7eb',
};

const fmtPct = (v?: number) => (v !== undefined ? `${(v * 100).toFixed(1)}%` : '—');
const fmtNum = (v?: number) => (v !== undefined ? v.toFixed(4) : '—');

export default ModelTable;
