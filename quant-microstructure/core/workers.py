"""
Async Workers Module.
Runs parallel 5m and 15m prediction pipelines with independent configurations.
Each worker has its own delta, embargo, rolling_window, and slippage settings.
"""

import asyncio
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Callable
import logging
from dataclasses import dataclass

# Use absolute imports for direct script execution
try:
    from .ingestion import BinanceIngestion, AggregatedWindow, OrderBook
    from .features import FeatureEngine, FeatureSet
    from .model import MicrostructureModel, SignalPrediction
    from .theta_optimizer import ThetaOptimizer, ThetaResult
    from .risk_sim import RiskSimulator, PositionSide
except ImportError:
    from ingestion import BinanceIngestion, AggregatedWindow, OrderBook
    from features import FeatureEngine, FeatureSet
    from model import MicrostructureModel, SignalPrediction
    from theta_optimizer import ThetaOptimizer, ThetaResult
    from risk_sim import RiskSimulator, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for a single timeframe worker."""
    name: str
    delta_minutes: int
    embargo_minutes: int
    theta_window_hours: int
    min_K_ratio: float
    slippage_bps: float
    rolling_recal_minutes: int
    
    @classmethod
    def from_dict(cls, name: str, config: dict) -> 'TimeframeConfig':
        return cls(
            name=name,
            delta_minutes=config.get('delta_minutes', 5),
            embargo_minutes=config.get('embargo_minutes', 7),
            theta_window_hours=config.get('theta_window_hours', 48),
            min_K_ratio=config.get('min_K_ratio', 2.6),
            slippage_bps=config.get('slippage_bps', 0.3),
            rolling_recal_minutes=config.get('rolling_recal_minutes', 60)
        )


@dataclass
class SignalOutput:
    """Signal output for publishing via ZeroMQ."""
    timestamp: float
    timeframe: str
    direction: int  # -1, 0, +1
    signal_value: float
    probability: float
    theta_star: float
    status: str  # 'WAITING', 'SIGNAL', 'EMBARGO'
    
    def to_dict(self) -> dict:
        return {
            'ts': self.timestamp,
            'tf': self.timeframe,
            'dir': self.direction,
            'S_t': self.signal_value,
            'p_t': self.probability,
            'θ*': self.theta_star,
            'status': self.status
        }


class TimeframeWorker:
    """
    Async worker for a single timeframe (5m or 15m).
    
    Pipeline: Ingestion → Features → ML → θ*-Opt → Risk Sim → Publish
    """
    
    def __init__(
        self,
        config: TimeframeConfig,
        model: MicrostructureModel,
        on_signal: Optional[Callable[[SignalOutput], None]] = None,
        on_metrics: Optional[Callable[[Dict], None]] = None
    ):
        """
        Args:
            config: Timeframe configuration
            model: Pre-loaded ML model
            on_signal: Callback for signal publishing
            on_metrics: Callback for metrics publishing
        """
        self.config = config
        self.model = model
        self.on_signal = on_signal
        self.on_metrics = on_metrics
        
        # Components
        self.feature_engine = FeatureEngine(window_size=100)
        self.theta_optimizer = ThetaOptimizer(
            max_win_rate=0.35,
            stop_loss_pct=0.002,
            take_profit_multiplier=config.min_K_ratio
        )
        self.risk_sim = RiskSimulator(
            max_exposure_pct=0.02,
            base_slippage_pct=config.slippage_bps / 10000
        )
        
        # State
        self._running = False
        self._current_candle_start: Optional[float] = None
        self._pending_signals: list = []
        self._last_recal_time: float = 0
        
        # Candle tracking
        self.delta_seconds = config.delta_minutes * 60
        self.embargo_seconds = config.embargo_minutes * 60
        
    async def process_window(self, window: AggregatedWindow):
        """Process aggregated window through the pipeline."""
        if not self._running:
            return
            
        if window.orderbook_snapshot is None:
            return
            
        ts = window.timestamp
        ob = window.orderbook_snapshot
        
        # Update feature engine
        bids = [(b.price, b.quantity) for b in ob.bids]
        asks = [(a.price, a.quantity) for a in ob.asks]
        
        trades = [
            (t.timestamp, t.price, t.quantity, t.is_buyer_maker)
            for t in window.trades
        ]
        
        features = self.feature_engine.compute_features(ts, bids, asks, trades)
        
        if features is None:
            return
            
        # Get ML prediction
        feature_array = features.to_array()
        prediction = self.model.predict(feature_array)
        
        if prediction is None:
            return
            
        prediction.timestamp = ts
        
        # Check candle boundaries and embargo
        await self._check_candle_boundary(ts, prediction)
        
    async def _check_candle_boundary(self, ts: float, prediction: SignalPrediction):
        """Check candle boundaries and handle embargo."""
        candle_start = int(ts / self.delta_seconds) * self.delta_seconds
        
        if self._current_candle_start != candle_start:
            # New candle started
            self._current_candle_start = candle_start
            
            # Process pending signals from previous candle (after embargo)
            await self._process_pending_signals()
            
            self._pending_signals = []
            
        # Add to pending (will be processed after embargo)
        self._pending_signals.append((ts, prediction))
        
    async def _process_pending_signals(self):
        """Process signals after embargo period."""
        if not self._pending_signals:
            return
            
        # Wait for embargo
        await asyncio.sleep(self.embargo_seconds)
        
        for ts, pred in self._pending_signals:
            # Compute outcome (would need actual future price)
            # For now, just emit signal
            signal = SignalOutput(
                timestamp=ts,
                timeframe=self.config.name,
                direction=pred.direction,
                signal_value=pred.signal_value,
                probability=pred.probability,
                theta_star=self.model.threshold,
                status='SIGNAL' if pred.direction != 0 else 'WAITING'
            )
            
            if self.on_signal:
                self.on_signal(signal)
                
        # Recalculate theta periodically
        if time.time() - self._last_recal_time > self.config.rolling_recal_minutes * 60:
            await self._recalibrate_theta()
            self._last_recal_time = time.time()
            
    async def _recalibrate_theta(self):
        """Recalibrate optimal theta threshold."""
        result = self.theta_optimizer.optimize()
        
        if result and result.is_valid:
            self.model.set_threshold(result.theta_star)
            logger.info(
                f"[{self.config.name}] Theta recalibrated: "
                f"θ*={result.theta_star:.3f}, W={result.win_rate:.2%}, "
                f"E[R]={result.expected_return:.4%}"
            )
            
            if self.on_metrics:
                self.on_metrics(result.to_dict())
                
    def start(self):
        """Start the worker."""
        self._running = True
        self._last_recal_time = time.time()
        logger.info(f"[{self.config.name}] Worker started")
        
    def stop(self):
        """Stop the worker."""
        self._running = False
        logger.info(f"[{self.config.name}] Worker stopped")


class ParallelWorkers:
    """
    Manages parallel 5m and 15m workers.
    Coordinates ingestion and distributes data to both workers.
    """
    
    def __init__(
        self,
        config_path: str,
        on_signal: Optional[Callable[[SignalOutput], None]] = None,
        on_metrics: Optional[Callable[[Dict], None]] = None
    ):
        """
        Args:
            config_path: Path to config.yaml
            on_signal: Callback for signal publishing
            on_metrics: Callback for metrics publishing
        """
        self.config_path = config_path
        self.on_signal = on_signal
        self.on_metrics = on_metrics
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Create workers
        self.workers: Dict[str, TimeframeWorker] = {}
        self._setup_workers()
        
        # Shared ingestion
        self.ingestion: Optional[BinanceIngestion] = None
        
        # State
        self._running = False
        
    def _setup_workers(self):
        """Initialize workers from config."""
        tf_configs = self.config.get('timeframes', {})
        
        for tf_name, tf_config in tf_configs.items():
            config = TimeframeConfig.from_dict(tf_name, tf_config)
            
            # Create model (would load from file in production)
            model_path = self.config.get('ml', {}).get('model_path')
            model = MicrostructureModel(
                model_path=model_path,
                threshold=0.3  # Default threshold
            )
            
            worker = TimeframeWorker(
                config=config,
                model=model,
                on_signal=self.on_signal,
                on_metrics=self.on_metrics
            )
            
            self.workers[tf_name] = worker
            logger.info(f"Created worker for {tf_name}")
            
    async def _on_window(self, window: AggregatedWindow):
        """Distribute window to all workers."""
        for worker in self.workers.values():
            await worker.process_window(window)
            
    async def run(self):
        """Run all workers in parallel."""
        self._running = True
        
        # Start workers
        for worker in self.workers.values():
            worker.start()
            
        # Setup shared ingestion
        exchange_config = self.config.get('exchange', {})
        
        def on_window_sync(window: AggregatedWindow):
            asyncio.create_task(self._on_window(window))
            
        self.ingestion = BinanceIngestion(
            symbol=exchange_config.get('symbol', 'BTCUSDT'),
            ws_url=exchange_config.get('ws_url', 'wss://fstream.binance.com/ws'),
            max_l2_levels=exchange_config.get('max_l2_levels', 20),
            on_window_complete=on_window_sync
        )
        
        logger.info("Starting parallel workers...")
        
        try:
            await self.ingestion.run()
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop all workers."""
        self._running = False
        
        if self.ingestion:
            self.ingestion.stop()
            
        for worker in self.workers.values():
            worker.stop()
            
        logger.info("All workers stopped")


async def main():
    """Example usage."""
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
    
    def on_signal_handler(signal: SignalOutput):
        print(f"SIGNAL [{signal.timeframe}]: dir={signal.direction}, "
              f"S_t={signal.signal_value:.3f}, p={signal.probability:.2f}")
              
    def on_metrics_handler(metrics: dict):
        print(f"METRICS: {metrics}")
    
    workers = ParallelWorkers(
        config_path=config_path,
        on_signal=on_signal_handler,
        on_metrics=on_metrics_handler
    )
    
    try:
        await workers.run()
    except KeyboardInterrupt:
        workers.stop()


if __name__ == "__main__":
    asyncio.run(main())
