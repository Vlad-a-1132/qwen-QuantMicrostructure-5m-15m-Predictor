"""
Async WebSocket ingestion for Binance L2 Orderbook + Trades.
Reconstructs orderbook and aggregates into 1s windows.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import websockets
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBookLevel:
    price: float
    quantity: float


@dataclass
class OrderBook:
    timestamp: float
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


@dataclass
class Trade:
    timestamp: float
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell, False = buy


@dataclass
class AggregatedWindow:
    timestamp: float
    trades: List[Trade] = field(default_factory=list)
    orderbook_snapshot: Optional[OrderBook] = None


class BinanceIngestion:
    """
    Async WebSocket client for Binance Futures L2 data.
    Implements orderbook reconstruction and trade aggregation.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        ws_url: str = "wss://fstream.binance.com/ws",
        max_l2_levels: int = 20,
        on_window_complete: Optional[Callable[[AggregatedWindow], None]] = None
    ):
        self.symbol = symbol.lower()
        self.ws_url = ws_url
        self.max_l2_levels = max_l2_levels
        self.on_window_complete = on_window_complete
        
        self.orderbook = OrderBook(timestamp=0)
        self.current_window: Optional[AggregatedWindow] = None
        self.window_trades: List[Trade] = []
        
        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._last_window_ts: float = 0
        
    async def connect(self):
        """Establish WebSocket connection."""
        streams = [
            f"{self.symbol}@depth20@100ms",
            f"{self.symbol}@trade"
        ]
        sub_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        self._ws = await websockets.connect(
            f"{self.ws_url}/stream",
            ping_interval=30,
            ping_timeout=10
        )
        await self._ws.send(json.dumps(sub_message))
        logger.info(f"Connected to Binance WS, subscribed to {streams}")
        
    async def _process_depth(self, data: dict):
        """Process L2 depth update and reconstruct orderbook."""
        ts = data.get('E', time.time() * 1000) / 1000
        
        bids = [
            OrderBookLevel(price=float(p), quantity=float(q))
            for p, q in data.get('bids', [])[:self.max_l2_levels]
        ]
        asks = [
            OrderBookLevel(price=float(p), quantity=float(q))
            for p, q in data.get('asks', [])[:self.max_l2_levels]
        ]
        
        self.orderbook = OrderBook(
            timestamp=ts,
            bids=bids,
            asks=asks
        )
        
    async def _process_trade(self, data: dict):
        """Process trade and add to current window."""
        ts = data.get('T', time.time() * 1000) / 1000
        
        trade = Trade(
            timestamp=ts,
            price=float(data['p']),
            quantity=float(data['q']),
            is_buyer_maker=data['m']
        )
        
        self.window_trades.append(trade)
        
    async def _check_window_boundary(self):
        """Check if we crossed 1s boundary and emit window."""
        current_ts = time.time()
        current_second = int(current_ts)
        
        if current_second > self._last_window_ts and self.window_trades:
            window = AggregatedWindow(
                timestamp=self._last_window_ts,
                trades=self.window_trades.copy(),
                orderbook_snapshot=self.orderbook
            )
            
            if self.on_window_complete:
                self.on_window_complete(window)
            
            self.window_trades = []
            self._last_window_ts = current_second
            
    async def run(self):
        """Main ingestion loop."""
        self._running = True
        self._last_window_ts = int(time.time())
        
        await self.connect()
        
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=5.0
                )
                data = json.loads(message)
                
                if 'data' not in data:
                    continue
                    
                payload = data['data']
                stream = data.get('stream', '')
                
                if 'depth' in stream:
                    await self._process_depth(payload)
                elif 'trade' in stream:
                    await self._process_trade(payload)
                    
                await self._check_window_boundary()
                
            except asyncio.TimeoutError:
                await self._check_window_boundary()
            except websockets.ConnectionClosed:
                logger.warning("WebSocket closed, reconnecting...")
                await asyncio.sleep(1)
                await self.connect()
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                await asyncio.sleep(1)
                
    def stop(self):
        """Stop ingestion."""
        self._running = False
        
    def get_current_orderbook(self) -> OrderBook:
        """Get latest orderbook snapshot."""
        return self.orderbook
    
    def get_mid_price(self) -> Optional[float]:
        """Get current mid-price."""
        return self.orderbook.mid_price


async def main():
    """Example usage."""
    def on_window(window: AggregatedWindow):
        print(f"Window {window.timestamp}: {len(window.trades)} trades, "
              f"mid={window.orderbook_snapshot.mid_price if window.orderbook_snapshot else 'N/A'}")
    
    ingestion = BinanceIngestion(
        symbol="BTCUSDT",
        max_l2_levels=20,
        on_window_complete=on_window
    )
    
    try:
        await ingestion.run()
    except KeyboardInterrupt:
        ingestion.stop()


if __name__ == "__main__":
    asyncio.run(main())
