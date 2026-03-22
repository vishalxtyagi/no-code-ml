import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('./api/client', () => ({
  uploadDataset: jest.fn(),
  runPipeline: jest.fn(),
  getResults: jest.fn(),
  getDownloadUrl: jest.fn(() => '/api/download-model/test-id'),
}));

test('renders AutoML Platform header', () => {
  render(<App />);
  expect(screen.getByText(/AutoML Platform/i)).toBeInTheDocument();
});

test('renders upload section', () => {
  render(<App />);
  expect(screen.getByText(/Upload Dataset/i)).toBeInTheDocument();
});

test('renders drop zone hint text', () => {
  render(<App />);
  expect(screen.getByText(/Drop your CSV here/i)).toBeInTheDocument();
});
