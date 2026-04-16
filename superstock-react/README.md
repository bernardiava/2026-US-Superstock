# 🚀 SuperStock Dashboard - React + Alpaca

Real-time US stock market dashboard built with React, TypeScript, and Alpaca Trade API.

## Features

- **Real-time Market Data**: Live stock prices via Alpaca API
- **Multi-symbol Tracking**: Monitor multiple stocks simultaneously
- **Price Change Indicators**: Visual color-coded price movements (green/red)
- **Market Status**: Shows if US market is open or closed
- **Responsive Design**: Dark mode Bloomberg Terminal-style UI
- **Custom Symbol Management**: Add/remove stocks dynamically

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **@alpacahq/alpaca-trade-api** for market data
- **CSS-in-JS** inline styling

## Prerequisites

1. Node.js 18+ installed
2. Alpaca Markets account (free paper trading available)
   - Get your API keys from: https://app.alpaca.markets/paper/dashboard/overview

## Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## Usage

1. Open the app in your browser (usually http://localhost:5173)
2. Click "⚙️ Config" to expand the configuration panel
3. Enter your Alpaca API Key and API Secret
4. Watch real-time stock data stream in!
5. Add custom symbols using the input field

## Default Symbols

The dashboard comes pre-loaded with popular tech stocks:
- NVDA (NVIDIA)
- TSLA (Tesla)
- AMD (Advanced Micro Devices)
- META (Meta Platforms)
- AAPL (Apple)

## Project Structure

```
superstock-react/
├── src/
│   ├── App.tsx          # Main dashboard component
│   ├── App.css          # Styles
│   ├── main.tsx         # Entry point
│   └── assets/          # Static assets
├── package.json
└── README.md
```

## API Integration

This app uses Alpaca's Market Data API to fetch real-time trades:
- `getLatestTrades()` - Fetches most recent trade for each symbol
- Updates every 1 second via polling interval
- Paper trading mode enabled by default (safe for testing)

## Disclaimer

**For educational purposes only.** This is not financial advice. 
Data provided by Alpaca Markets API. Past performance does not guarantee future results.

## License

MIT
