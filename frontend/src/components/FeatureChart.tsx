import React from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { FeatureImportance } from '../api/client';

interface Props {
  importances: FeatureImportance[];
}

const COLORS = [
  '#4f46e5', '#7c3aed', '#a855f7', '#c084fc',
  '#818cf8', '#6366f1', '#8b5cf6', '#9333ea',
];

const FeatureChart: React.FC<Props> = ({ importances }) => {
  const data = importances
    .slice(0, 10)
    .map((fi) => ({ ...fi, pct: (fi.importance * 100).toFixed(1) }));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} layout="vertical" margin={{ left: 20, right: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <YAxis type="category" dataKey="feature" width={140} tick={{ fontSize: 12 }} />
        <Tooltip formatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} />
        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default FeatureChart;
