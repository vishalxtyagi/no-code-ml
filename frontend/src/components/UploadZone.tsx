import React, { useCallback, useState } from 'react';

interface Props {
  onUpload: (file: File) => void;
  loading: boolean;
}

const UploadZone: React.FC<Props> = ({ onUpload, loading }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onUpload(file);
    },
    [onUpload]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${dragOver ? '#4f46e5' : '#d1d5db'}`,
        borderRadius: 12,
        padding: '48px 32px',
        textAlign: 'center',
        cursor: loading ? 'not-allowed' : 'pointer',
        background: dragOver ? '#eef2ff' : '#f9fafb',
        transition: 'all 0.2s',
      }}
      onClick={() => !loading && document.getElementById('csv-input')?.click()}
    >
      <input
        id="csv-input"
        type="file"
        accept=".csv"
        style={{ display: 'none' }}
        onChange={handleChange}
        disabled={loading}
      />
      <div style={{ fontSize: 48, marginBottom: 12 }}>📂</div>
      {loading ? (
        <p style={{ color: '#6b7280', margin: 0 }}>Uploading…</p>
      ) : (
        <>
          <p style={{ fontWeight: 600, fontSize: 16, margin: '0 0 8px' }}>
            Drop your CSV here or click to browse
          </p>
          <p style={{ color: '#6b7280', margin: 0, fontSize: 14 }}>
            Only .csv files are supported
          </p>
        </>
      )}
    </div>
  );
};

export default UploadZone;
