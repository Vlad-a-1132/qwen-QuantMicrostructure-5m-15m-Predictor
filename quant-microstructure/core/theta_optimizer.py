"""
Theta Optimizer Module.
Finds optimal threshold θ* with W(θ) ≤ 0.35 constraint.
Uses scipy.optimize for rolling calibration.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThetaResult:
    """Result of theta optimization."""
    theta_star: float
    win_rate: float
    expected_return: float
    sharpe_ratio: float
    k_ratio: float
    n_trades: int
    is_valid: bool  # True if W(θ) ≤ 0.35 and E[R] > 0
    
    def to_dict(self) -> dict:
        return {
            'theta_star': self.theta_star,
            'win_rate': self.win_rate,
            'expected_return': self.expected_return,
            'sharpe_ratio': self.sharpe_ratio,
            'k_ratio': self.k_ratio,
            'n_trades': self.n_trades,
            'is_valid': self.is_valid
        }


class ThetaOptimizer:
    """
    Optimizes decision threshold θ* to maximize risk-adjusted returns
    subject to WinRate ≤ 0.35 constraint.
    
    Based on the theorem that there exists θ* such that:
    - W(θ*) ≤ 0.35
    - E[R|θ*] > 0
    - ∂K(θ)/∂θ > 0 as θ → ∞
    """
    
    def __init__(
        self,
        min_win_rate: float = 0.25,
        max_win_rate: float = 0.35,
        stop_loss_pct: float = 0.002,  # 0.2%
        take_profit_multiplier: float = 2.6,  # K = 2.6 for 5m
        transaction_cost_pct: float = 0.0005  # 0.05%
    ):
        """
        Args:
            min_win_rate: Minimum acceptable win rate (default 0.25)
            max_win_rate: Maximum win rate constraint (default 0.35)
            stop_loss_pct: Stop loss as percentage (default 0.2%)
            take_profit_multiplier: Take profit = K × stop_loss
            transaction_cost_pct: Total transaction cost per trade
        """
        self.min_win_rate = min_win_rate
        self.max_win_rate = max_win_rate
        self.stop_loss = stop_loss_pct
        self.take_profit = take_profit_multiplier * stop_loss_pct
        self.transaction_cost = transaction_cost_pct
        
        # Historical data for optimization
        self._signals: List[float] = []
        self._outcomes: List[float] = []  # Actual returns
        
    def add_observation(self, signal: float, outcome: float):
        """
        Add observation for optimization.
        
        Args:
            signal: Predicted signal value S_t
            outcome: Actual return r_{t+Δ}
        """
        self._signals.append(signal)
        self._outcomes.append(outcome)
        
    def _compute_metrics(
        self, 
        theta: float
    ) -> Tuple[float, float, float, float, int]:
        """
        Compute trading metrics for given threshold.
        
        Returns:
            (win_rate, expected_return, sharpe_ratio, k_ratio, n_trades)
        """
        if len(self._signals) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
            
        signals = np.array(self._signals)
        outcomes = np.array(self._outcomes)
        
        # Find trades where |signal| > theta
        mask = np.abs(signals) > theta
        trade_signals = signals[mask]
        trade_outcomes = outcomes[mask]
        
        n_trades = len(trade_outcomes)
        
        if n_trades == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
            
        # Determine predicted direction
        directions = np.sign(trade_signals)
        
        # Check if outcome matches prediction (win)
        # Win if sign(outcome) == sign(signal) and |outcome| > costs
        net_outcomes = directions * trade_outcomes - self.transaction_cost
        wins = net_outcomes > 0
        
        win_rate = np.sum(wins) / n_trades
        
        # Compute PnL for each trade with fixed stop/take profit
        pnl_list = []
        for outcome in trade_outcomes:
            if outcome > 0:
                # Long trade
                pnl = min(outcome, self.take_profit) - self.transaction_cost
                if outcome < -self.stop_loss:
                    pnl = -self.stop_loss - self.transaction_cost
            else:
                # Short trade
                pnl = min(-outcome, self.take_profit) - self.transaction_cost
                if outcome > self.stop_loss:
                    pnl = -self.stop_loss - self.transaction_cost
            pnl_list.append(pnl)
            
        pnl_array = np.array(pnl_list)
        
        expected_return = np.mean(pnl_array)
        std_return = np.std(pnl_array)
        
        sharpe_ratio = expected_return / std_return if std_return > 0 else 0.0
        
        # Effective K ratio (actual average win / average loss)
        avg_win = np.mean(pnl_array[pnl_array > 0]) if np.any(pnl_array > 0) else 0
        avg_loss = abs(np.mean(pnl_array[pnl_array < 0])) if np.any(pnl_array < 0) else 1
        k_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return win_rate, expected_return, sharpe_ratio, k_ratio, n_trades
    
    def _objective(self, theta: float) -> float:
        """
        Objective function: maximize Sharpe ratio subject to constraints.
        
        Returns negative Sharpe ratio (for minimization).
        Applies heavy penalty if constraints violated.
        """
        win_rate, exp_ret, sharpe, k_ratio, n_trades = self._compute_metrics(theta)
        
        # Apply constraints as penalties
        penalty = 0.0
        
        # Win rate must be in [min_win_rate, max_win_rate]
        if win_rate < self.min_win_rate:
            penalty += 1000 * (self.min_win_rate - win_rate) ** 2
        elif win_rate > self.max_win_rate:
            penalty += 1000 * (win_rate - self.max_win_rate) ** 2
            
        # Expected return must be positive
        if exp_ret <= 0:
            penalty += 1000 * exp_ret ** 2
            
        # Must have minimum number of trades for statistical significance
        if n_trades < 10:
            penalty += 100 * (10 - n_trades)
            
        # Return negative Sharpe (we want to maximize)
        return -sharpe + penalty
    
    def optimize(self) -> Optional[ThetaResult]:
        """
        Find optimal threshold θ*.
        
        Returns:
            ThetaResult with optimal parameters or None if optimization fails
        """
        if len(self._signals) < 50:
            logger.warning("Insufficient data for optimization")
            return None
            
        # Search range for theta (typically 0 to 2 for normalized signals)
        bounds = (0.0, 2.0)
        
        try:
            result = minimize_scalar(
                self._objective,
                bounds=bounds,
                method='bounded',
                options={'xatol': 0.01}
            )
            
            if not result.success:
                logger.warning(f"Optimization failed: {result.message}")
                return None
                
            theta_star = result.x
            
            # Compute final metrics
            win_rate, exp_ret, sharpe, k_ratio, n_trades = self._compute_metrics(theta_star)
            
            is_valid = (
                self.min_win_rate <= win_rate <= self.max_win_rate and
                exp_ret > 0
            )
            
            return ThetaResult(
                theta_star=theta_star,
                win_rate=win_rate,
                expected_return=exp_ret,
                sharpe_ratio=sharpe,
                k_ratio=k_ratio,
                n_trades=n_trades,
                is_valid=is_valid
            )
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return None
    
    def rolling_optimize(
        self,
        window_size: int = 1000,
        step: int = 100
    ) -> List[ThetaResult]:
        """
        Perform rolling optimization to track theta drift.
        
        Args:
            window_size: Number of observations per window
            step: Step size between windows
            
        Returns:
            List of ThetaResult for each window
        """
        results = []
        
        for start in range(0, len(self._signals) - window_size, step):
            end = start + window_size
            
            # Extract window
            window_signals = self._signals[start:end]
            window_outcomes = self._outcomes[start:end]
            
            # Temporarily replace data
            old_signals = self._signals
            old_outcomes = self._outcomes
            self._signals = window_signals
            self._outcomes = window_outcomes
            
            # Optimize
            result = self.optimize()
            if result:
                results.append(result)
                
            # Restore data
            self._signals = old_signals
            self._outcomes = old_outcomes
            
        return results
    
    def clear_history(self):
        """Clear historical data."""
        self._signals = []
        self._outcomes = []
        
    def set_parameters(
        self,
        stop_loss_pct: Optional[float] = None,
        take_profit_multiplier: Optional[float] = None,
        transaction_cost_pct: Optional[float] = None
    ):
        """Update trading parameters."""
        if stop_loss_pct is not None:
            self.stop_loss = stop_loss_pct
        if take_profit_multiplier is not None:
            self.take_profit = take_profit_multiplier * self.stop_loss
        if transaction_cost_pct is not None:
            self.transaction_cost = transaction_cost_pct


def create_theta_optimizer(
    timeframe: str = "5m",
    config: Optional[Dict] = None
) -> ThetaOptimizer:
    """
    Factory function to create optimizer with timeframe-specific parameters.
    
    Args:
        timeframe: "5m" or "15m"
        config: Optional configuration dictionary
    """
    # Default parameters based on timeframe
    if timeframe == "5m":
        params = {
            'take_profit_multiplier': 2.6,
            'stop_loss_pct': 0.002,
            'transaction_cost_pct': 0.0008  # Higher for 5m
        }
    elif timeframe == "15m":
        params = {
            'take_profit_multiplier': 2.3,
            'stop_loss_pct': 0.003,
            'transaction_cost_pct': 0.0007  # Lower for 15m
        }
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
        
    # Override with config if provided
    if config:
        params.update(config)
        
    return ThetaOptimizer(**params)


if __name__ == "__main__":
    # Example usage
    optimizer = ThetaOptimizer(
        max_win_rate=0.35,
        stop_loss_pct=0.002,
        take_profit_multiplier=2.6
    )
    
    # Simulate some data
    np.random.seed(42)
    n_samples = 500
    
    # Generate signals and outcomes with some correlation
    signals = np.random.randn(n_samples)
    outcomes = 0.3 * signals + 0.7 * np.random.randn(n_samples) * 0.01
    
    for s, o in zip(signals, outcomes):
        optimizer.add_observation(s, o)
        
    result = optimizer.optimize()
    
    if result:
        print(f"Optimization Result:")
        print(f"  θ* = {result.theta_star:.3f}")
        print(f"  Win Rate = {result.win_rate:.2%}")
        print(f"  Expected Return = {result.expected_return:.4%}")
        print(f"  Sharpe Ratio = {result.sharpe_ratio:.2f}")
        print(f"  K Ratio = {result.k_ratio:.2f}")
        print(f"  Valid = {result.is_valid}")
