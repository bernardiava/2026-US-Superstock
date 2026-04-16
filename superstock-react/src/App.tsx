import { useState, useEffect, useRef } from 'react';
import Alpaca from '@alpacahq/alpaca-trade-api';
import './App.css';

// Types
interface StockTick {
  symbol: string;
  price: number;
  timestamp: number;
  volume: number;
  change: number;
  changePercent: number;
}

// Custom hook for Alpaca data stream
const useAlpacaDataStream = (apiKey: string, apiSecret: string, symbols: string[]) => {
  const [ticks, setTicks] = useState<Record<string, StockTick>>({});
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const alpacaRef = useRef<Alpaca | null>(null);

  useEffect(() => {
    if (!apiKey || !apiSecret) {
      setError('API credentials required');
      return;
    }

    try {
      alpacaRef.current = new Alpaca({
        keyId: apiKey,
        secretKey: apiSecret,
        paper: true,
        verbose: false,
      });

      const fetchLatestTrades = async () => {
        try {
          const latestTrades = await alpacaRef.current!.getLatestTrades(symbols);
          
          const newTicks: Record<string, StockTick> = {};
          symbols.forEach(symbol => {
            const trade = latestTrades[symbol];
            if (trade) {
              const prevTick = ticks[symbol];
              const prevPrice = prevTick?.price || trade.p;
              const change = trade.p - prevPrice;
              const changePercent = (change / prevPrice) * 100;

              newTicks[symbol] = {
                symbol,
                price: trade.p,
                timestamp: trade.t,
                volume: trade.s,
                change,
                changePercent,
              };
            }
          });
          
          setTicks(prev => ({ ...prev, ...newTicks }));
          setConnected(true);
          setError(null);
        } catch (err: any) {
          setError(err.message);
          setConnected(false);
        }
      };

      fetchLatestTrades();
      const interval = setInterval(fetchLatestTrades, 1000);

      return () => clearInterval(interval);
    } catch (err: any) {
      setError(err.message);
    }
  }, [apiKey, apiSecret, symbols.join(',')]);

  return { ticks, connected, error };
};

// Price Display Component
const PriceDisplay = ({ price, prevPrice, fontSize = '2.5rem' }: { price: number; prevPrice: number; fontSize?: string }) => {
  const color = price > prevPrice ? '#00ff00' : price < prevPrice ? '#ff0000' : '#ffffff';
  
  return (
    <div style={{ 
      fontSize, 
      fontWeight: 'bold', 
      color, 
      fontVariantNumeric: 'tabular-nums',
      transition: 'color 0.1s ease' 
    }}>
      ${price.toFixed(2)}
    </div>
  );
};

// Stock Card Component
const StockCard = ({ tick, prevTick }: { tick: StockTick; prevTick?: StockTick }) => {
  if (!tick) return null;
  const prevPrice = prevTick?.price || tick.price;

  return (
    <div style={{
      backgroundColor: '#1f2833',
      padding: '1.5rem',
      borderRadius: '8px',
      borderLeft: '4px solid #45a29e',
      marginBottom: '1rem'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 style={{ margin: 0, color: '#66fcf1', fontSize: '1.5rem' }}>{tick.symbol}</h3>
        <span style={{ 
          backgroundColor: tick.change >= 0 ? 'rgba(0,255,0,0.2)' : 'rgba(255,0,0,0.2)',
          color: tick.change >= 0 ? '#00ff00' : '#ff0000',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '0.9rem'
        }}>
          {tick.change >= 0 ? '+' : ''}{tick.changePercent.toFixed(2)}%
        </span>
      </div>
      
      <PriceDisplay price={tick.price} prevPrice={prevPrice} />
      
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '1rem' }}>
        <div>
          <div style={{fontSize: '0.75rem', color: '#8892b0'}}>CHANGE</div>
          <div style={{fontSize: '1rem', fontWeight: '600', color: tick.change >= 0 ? '#00ff00' : '#ff0000'}}>
            {tick.change >= 0 ? '+' : ''}{tick.change.toFixed(2)} USD
          </div>
        </div>
        <div>
          <div style={{fontSize: '0.75rem', color: '#8892b0'}}>VOLUME</div>
          <div style={{fontSize: '1rem', fontWeight: '600'}}>{tick.volume.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
};

// Main Dashboard Component
function App() {
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [symbols, setSymbols] = useState<string[]>(['NVDA', 'TSLA', 'AMD', 'META', 'AAPL']);
  const [customSymbol, setCustomSymbol] = useState('');
  const [showConfig, setShowConfig] = useState(false);
  
  const { ticks, connected, error } = useAlpacaDataStream(apiKey, apiSecret, symbols);
  const prevTicksRef = useRef<Record<string, StockTick>>({});
  
  useEffect(() => {
    prevTicksRef.current = ticks;
  }, [ticks]);

  const handleAddSymbol = () => {
    if (customSymbol && !symbols.includes(customSymbol.toUpperCase())) {
      setSymbols([...symbols, customSymbol.toUpperCase()]);
      setCustomSymbol('');
    }
  };

  const handleRemoveSymbol = (symbolToRemove: string) => {
    setSymbols(symbols.filter(s => s !== symbolToRemove));
  };

  const marketStatus = (() => {
    const hour = new Date().getUTCHours();
    const day = new Date().getUTCDay();
    if (day === 0 || day === 6) return 'CLOSED';
    return hour >= 9 && hour < 16 ? 'OPEN' : 'CLOSED';
  })();

  return (
    <div style={{
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      backgroundColor: '#0b0c10',
      color: '#c5c6c7',
      minHeight: '100vh',
      padding: '2rem'
    }}>
      <header style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '2rem',
        paddingBottom: '1rem',
        borderBottom: '1px solid #1f2833'
      }}>
        <div>
          <h1 style={{margin: 0, fontSize: '2rem', color: '#66fcf1'}}>🚀 SuperStock Dashboard</h1>
          <p style={{margin: '5px 0 0', color: '#8892b0', fontSize: '0.9rem'}}>
            Real-time US Market Data via Alpaca
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          <div style={{
            backgroundColor: marketStatus === 'OPEN' ? '#00ff00' : '#ff0000',
            color: '#000',
            padding: '5px 12px',
            borderRadius: '4px',
            fontWeight: 'bold',
            fontSize: '0.9rem'
          }}>
            MARKET {marketStatus}
          </div>
          
          <div style={{
            backgroundColor: connected ? '#00ff00' : '#ff0000',
            color: '#000',
            padding: '5px 12px',
            borderRadius: '4px',
            fontWeight: 'bold',
            fontSize: '0.9rem'
          }}>
            {connected ? '● LIVE' : '● OFFLINE'}
          </div>
          
          <button
            onClick={() => setShowConfig(!showConfig)}
            style={{
              backgroundColor: '#1f2833',
              color: '#66fcf1',
              border: '1px solid #45a29e',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            ⚙️ Config
          </button>
        </div>
      </header>

      {showConfig && (
        <div style={{
          backgroundColor: '#1f2833',
          padding: '1.5rem',
          borderRadius: '8px',
          marginBottom: '2rem'
        }}>
          <h3 style={{marginTop: 0, color: '#66fcf1'}}>Alpaca API Configuration</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <label style={{display: 'block', marginBottom: '5px', fontSize: '0.85rem'}}>API Key</label>
              <input
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter your Alpaca API Key"
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: '#0b0c10',
                  border: '1px solid #45a29e',
                  borderRadius: '4px',
                  color: '#c5c6c7'
                }}
              />
            </div>
            <div>
              <label style={{display: 'block', marginBottom: '5px', fontSize: '0.85rem'}}>API Secret</label>
              <input
                type="password"
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                placeholder="Enter your Alpaca API Secret"
                style={{
                  width: '100%',
                  padding: '10px',
                  backgroundColor: '#0b0c10',
                  border: '1px solid #45a29e',
                  borderRadius: '4px',
                  color: '#c5c6c7'
                }}
              />
            </div>
          </div>
          <p style={{fontSize: '0.8rem', color: '#8892b0', margin: 0}}>
            ℹ️ Get your API keys from <a href="https://app.alpaca.markets/paper/dashboard/overview" target="_blank" rel="noopener noreferrer" style={{color: '#66fcf1'}}>Alpaca Paper Trading Dashboard</a>
          </p>
        </div>
      )}

      {error && (
        <div style={{
          backgroundColor: 'rgba(255,0,0,0.1)',
          border: '1px solid #ff0000',
          color: '#ff0000',
          padding: '1rem',
          borderRadius: '8px',
          marginBottom: '2rem'
        }}>
          ⚠️ {error}
        </div>
      )}

      <div style={{
        marginBottom: '2rem',
        display: 'flex',
        gap: '10px',
        flexWrap: 'wrap',
        alignItems: 'center'
      }}>
        <div style={{display: 'flex', gap: '10px', flexWrap: 'wrap'}}>
          {symbols.map(symbol => (
            <div
              key={symbol}
              style={{
                backgroundColor: '#1f2833',
                padding: '8px 12px',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              <span style={{fontWeight: 'bold', color: '#66fcf1'}}>{symbol}</span>
              <button
                onClick={() => handleRemoveSymbol(symbol)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#ff0000',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  padding: '0 2px'
                }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
        
        <div style={{display: 'flex', gap: '5px'}}>
          <input
            type="text"
            value={customSymbol}
            onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
            onKeyPress={(e) => e.key === 'Enter' && handleAddSymbol()}
            placeholder="Add symbol..."
            style={{
              padding: '8px 12px',
              backgroundColor: '#1f2833',
              border: '1px solid #45a29e',
              borderRadius: '4px',
              color: '#c5c6c7',
              width: '120px'
            }}
          />
          <button
            onClick={handleAddSymbol}
            style={{
              backgroundColor: '#45a29e',
              color: '#0b0c10',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            +
          </button>
        </div>
      </div>

      <main style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))', gap: '2rem'}}>
        {symbols.map(symbol => (
          <StockCard
            key={symbol}
            tick={ticks[symbol]}
            prevTick={prevTicksRef.current[symbol]}
          />
        ))}
      </main>

      <footer style={{
        marginTop: '3rem',
        paddingTop: '1rem',
        borderTop: '1px solid #1f2833',
        textAlign: 'center',
        fontSize: '0.85rem',
        color: '#8892b0'
      }}>
        <p>
          Data provided by Alpaca Markets API • For educational purposes only • Not financial advice
        </p>
        <p style={{marginTop: '5px'}}>
          Built with React + TypeScript + Alpaca Trade API
        </p>
      </footer>
    </div>
  );
}

export default App;
