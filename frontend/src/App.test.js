import { render, screen } from '@testing-library/react';
import App, { sourceHref, sourceLabel, sourceSnippet } from './App';

test('renders the PaperPilot workspace', () => {
  render(<App />);
  expect(screen.getByText(/PaperPilot/i)).toBeInTheDocument();
  expect(screen.getByText(/Unlock Your Documents/i)).toBeInTheDocument();
});

test('source cards prefer pdf urls and abstracts over Unknown', () => {
  const academic = {
    kind: "academic",
    title: "Attention Is All You Need",
    abstract: "We propose the Transformer.",
    pdf_url: "https://arxiv.org/pdf/1706.03762",
  };
  expect(sourceHref(academic)).toBe("https://arxiv.org/pdf/1706.03762");
  expect(sourceLabel(academic)).toBe("Attention Is All You Need");
  expect(sourceSnippet(academic)).toBe("We propose the Transformer.");

  const library = { kind: "library", source: "rag.pdf", text: "BM25 hybrid search" };
  expect(sourceHref(library)).toBe("");
  expect(sourceLabel(library)).toBe("rag.pdf");
  expect(sourceSnippet(library)).toBe("BM25 hybrid search");
});

test('renders the PaperPilot workspace', () => {
  render(<App />);
  expect(screen.getByText(/PaperPilot/i)).toBeInTheDocument();
  expect(screen.getByText(/Unlock Your Documents/i)).toBeInTheDocument();
});
