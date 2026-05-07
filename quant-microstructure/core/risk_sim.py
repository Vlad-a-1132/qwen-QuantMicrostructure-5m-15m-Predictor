"""
Risk Simulation Module.
Slippage modeling, position sizing, and PnL tracking.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Represents a completed trade."""
    timestamp: float
    timeframe: str
    side: PositionSide
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    slippage_pct: float
    hold_time_seconds: float
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'timeframe': self.timeframe,
            'side': self.side.name,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'slippage_pct': self.slippage_pct,
            'hold_time_seconds': self.hold_time_seconds
        }


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    n_trades: int
    n_wins: int
    avg_win: float
    avg_loss: float
    
    def to_dict(self) -> dict:
        return {
            'total_pnl': self.total_pnl,
            'total_pnl_pct': self.total_pnl_pct,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'n_trades': self.n_trades,
            'n_wins': self.n_wins,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss
        }


class RiskSimulator:
    """
    Simulates trading with realistic slippage and position sizing.
    
    Implements:
    - Effective cost model: C_eff = fee + 0.5·spread + α·(V_order / V_book)
    - Kelly fraction position sizing
    - Drawdown monitoring with circuit breaker
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_exposure_pct: float = 0.02,
        fee_pct: float = 0.0004,
        base_slippage_pct: float = 0.0003,
        slippage_impact: float = 0.5,
        max_drawdown_stop: float = -0.05,
        kelly_multiplier: float = 0.5  # Fractional Kelly
    ):
        """
        Args:
            initial_capital: Starting capital
            max_exposure_pct: Maximum position size as % of capital
            fee_pct: Exchange fee percentage
            base_slippage_pct: Base slippage (spread component)
            slippage_impact: Price impact coefficient α
            max_drawdown_stop: Circuit breaker threshold
            kelly_multiplier: Kelly fraction multiplier (0.5 = half Kelly)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_exposure_pct = max_exposure_pct
        self.fee_pct = fee_pct
        self.base_slippage_pct = base_slippage_pct
        self.slippage_impact = slippage_impact
        self.max_drawdown_stop = max_drawdown_stop
        self.kelly_multiplier = kelly_multiplier
        
        # Tracking
        self._trades: List[Trade] = []
        self._equity_curve: List[float] = [initial_capital]
        self._peak_equity: float = initial_capital
        self._current_drawdown: float = 0.0
        self._max_drawdown: float = 0.0
        
        # For Kelly calculation
        self._win_history: List[float] = []
        
        # Circuit breaker
        self._is_stopped: bool = False
        
    def compute_effective_cost(
        self,
        spread_pct: float,
        order_volume: float,
        book_volume: float
    ) -> float:
        """
        Compute effective transaction cost.
        
        C_eff = fee + 0.5·spread + α·(V_order / V_book)
        """
        if book_volume == 0:
            book_volume = 1.0  # Avoid division by zero
            
        impact = self.slippage_impact * (order_volume / book_volume)
        
        cost = self.fee_pct + 0.5 * spread_pct + impact
        return cost
    
    def compute_kelly_fraction(self) -> float:
        """
        Compute Kelly-optimal position size.
        
        f* = (p·b - q) / b
        where p = win probability, q = 1-p, b = win/loss ratio
        """
        if len(self._win_history) < 10:
            # Not enough history, use fixed fraction
            return self.max_exposure_pct
            
        wins = [w for w in self._win_history if w > 0]
        losses = [w for w in self._win_history if w <= 0]
        
        if not wins or not losses:
            return self.max_exposure_pct * 0.5
            
        p = len(wins) / len(self._win_history)
        q = 1 - p
        
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return self.max_exposure_pct
            
        b = avg_win / avg_loss
        
        # Kelly formula
        kelly = (p * b - q) / b
        
        # Apply multiplier and bounds
        kelly = max(0, min(kelly * self.kelly_multiplier, self.max_exposure_pct))
        
        return kelly
    
    def compute_position_size(
        self,
        price: float,
        signal_strength: float = 1.0
    ) -> float:
        """
        Compute position size based on Kelly fraction and signal strength.
        
        Args:
            price: Current asset price
            signal_strength: Signal confidence (0 to 1)
            
        Returns:
            Quantity to trade
        """
        if self._is_stopped:
            return 0.0
            
        kelly_frac = self.compute_kelly_fraction()
        
        # Scale by signal strength
        effective_fraction = kelly_frac * min(abs(signal_strength), 1.0)
        
        # Dollar amount
        dollar_amount = self.capital * effective_fraction
        
        # Convert to quantity
        quantity = dollar_amount / price
        
        return quantity
    
    def simulate_entry(
        self,
        timestamp: float,
        price: float,
        side: PositionSide,
        signal_strength: float,
        spread_pct: float,
        book_volume: float
    ) -> Optional[Tuple[float, float]]:
        """
        Simulate entering a position.
        
        Returns:
            (quantity, effective_entry_price) or None if stopped
        """
        if self._is_stopped:
            logger.warning("Circuit breaker triggered, no new positions")
            return None
            
        quantity = self.compute_position_size(price, signal_strength)
        
        if quantity <= 0:
            return None
            
        # Compute slippage
        order_volume = quantity * price
        slippage = self.compute_effective_cost(spread_pct, order_volume, book_volume)
        
        # Slippage is unfavorable
        if side == PositionSide.LONG:
            entry_price = price * (1 + slippage)
        else:
            entry_price = price * (1 - slippage)
            
        return quantity, entry_price
    
    def simulate_exit(
        self,
        timestamp: float,
        timeframe: str,
        side: PositionSide,
        entry_timestamp: float,
        entry_price: float,
        quantity: float,
        exit_price_raw: float,
        spread_pct: float,
        book_volume: float
    ) -> Trade:
        """
        Simulate exiting a position.
        
        Returns:
            Trade object with realized PnL
        """
        # Compute slippage on exit
        order_volume = quantity * exit_price_raw
        slippage = self.compute_effective_cost(spread_pct, order_volume, book_volume)
        
        # Slippage is unfavorable
        if side == PositionSide.LONG:
            exit_price = exit_price_raw * (1 - slippage)
        else:
            exit_price = exit_price_raw * (1 + slippage)
            
        # Compute PnL
        if side == PositionSide.LONG:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
            
        pnl = pnl_pct * quantity * entry_price
        
        # Update capital
        old_capital = self.capital
        self.capital += pnl
        self._equity_curve.append(self.capital)
        
        # Track drawdown
        if self.capital > self._peak_equity:
            self._peak_equity = self.capital
            
        self._current_drawdown = (self.capital - self._peak_equity) / self._peak_equity
        self._max_drawdown = min(self._max_drawdown, self._current_drawdown)
        
        # Check circuit breaker
        if self._current_drawdown < self.max_drawdown_stop:
            self._is_stopped = True
            logger.warning(f"Circuit breaker triggered! Drawdown: {self._current_drawdown:.2%}")
            
        # Track win history for Kelly
        self._win_history.append(pnl_pct)
        
        # Create trade record
        hold_time = timestamp - entry_timestamp
        
        trade = Trade(
            timestamp=timestamp,
            timeframe=timeframe,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            slippage_pct=slippage,
            hold_time_seconds=hold_time
        )
        
        self._trades.append(trade)
        
        return trade
    
    def get_metrics(self) -> RiskMetrics:
        """Compute current risk metrics."""
        if not self._trades:
            return RiskMetrics(
                total_pnl=0.0,
                total_pnl_pct=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                n_trades=0,
                n_wins=0,
                avg_win=0.0,
                avg_loss=0.0
            )
            
        pnls = np.array([t.pnl_pct for t in self._trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        
        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital
        
        win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
        
        profit_factor = abs(np.sum(wins) / np.sum(losses)) if np.sum(losses) != 0 else float('inf')
        
        sharpe_ratio = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
        
        return RiskMetrics(
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self._max_drawdown,
            current_drawdown=self._current_drawdown,
            n_trades=len(self._trades),
            n_wins=len(wins),
            avg_win=avg_win,
            avg_loss=avg_loss
        )
    
    def reset(self):
        """Reset simulator to initial state."""
        self.capital = self.initial_capital
        self._trades = []
        self._equity_curve = [self.initial_capital]
        self._peak_equity = self.initial_capital
        self._current_drawdown = 0.0
        self._max_drawdown = 0.0
        self._win_history = []
        self._is_stopped = False
        
    def get_trade_history(self) -> List[Dict]:
        """Get list of all trades as dictionaries."""
        return [t.to_dict() for t in self._trades]
    
    def get_equity_curve(self) -> List[float]:
        """Get equity curve."""
        return self._equity_curve.copy()


def create_risk_simulator(config: Optional[Dict] = None) -> RiskSimulator:
    """Factory function to create risk simulator from config."""
    default_config = {
        'initial_capital': 10000.0,
        'max_exposure_pct': 0.02,
        'fee_pct': 0.0004,
        'base_slippage_pct': 0.0003,
        'slippage_impact': 0.5,
        'max_drawdown_stop': -0.05,
        'kelly_multiplier': 0.5
    }
    
    if config:
        default_config.update(config)
        
    return RiskSimulator(**default_config)


if __name__ == "__main__":
    # Example usage
    sim = RiskSimulator(
        initial_capital=10000.0,
        max_exposure_pct=0.02
    )
    
    # Simulate some trades
    import time
    ts = time.time()
    
    # Long trade
    entry = sim.simulate_entry(
        timestamp=ts,
        price=50000.0,
        side=PositionSide.LONG,
        signal_strength=0.8,
        spread_pct=0.0002,
        book_volume=1000000.0
    )
    
    if entry:
        qty, entry_price = entry
        print(f"Entered LONG: {qty} @ {entry_price}")
        
        # Exit after some time
        trade = sim.simulate_exit(
            timestamp=ts + 300,
            timeframe="5m",
            side=PositionSide.LONG,
            entry_timestamp=ts,
            entry_price=entry_price,
            quantity=qty,
            exit_price_raw=50200.0,
            spread_pct=0.0002,
            book_volume=1000000.0
        )
        
        print(f"Exited: PnL = {trade.pnl:.2f} ({trade.pnl_pct:.2%})")
        
        # Get metrics
        metrics = sim.get_metrics()
        print(f"Metrics: {metrics.to_dict()}")
