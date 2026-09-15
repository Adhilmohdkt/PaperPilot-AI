import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the PaperPilot workspace', () => {
  render(<App />);
  expect(screen.getByText(/PaperPilot/i)).toBeInTheDocument();
  expect(screen.getByText(/Unlock Your Documents/i)).toBeInTheDocument();
});
