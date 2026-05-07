"""
Feature Engineering Module.
Computes microstructure features: OFI, Microprice Drift, Liquidity Vacuum, VPIN.
Uses Polars for vectorized operations without GIL blocks.
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for computed features at time t."""
    timestamp: float
    ofi: float  # Order Flow Imbalance
    microprice_drift: float  # μ_t
    liquidity_vacuum: float  # 𝒱_t
    vpin: float  # VPIN toxicity
    d_ofi_dt: float  # Rate of change of OFI
    d_mu_dt: float  # Rate of change of microprice drift
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'OFI': self.ofi,
            'microprice_drift': self.microprice_drift,
            'liquidity_vacuum': self.liquidity_vacuum,
            'VPIN': self.vpin,
            'dOFI_dt': self.d_ofi_dt,
            'dμ_dt': self.d_mu_dt
        }
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.ofi,
            self.microprice_drift,
            self.liquidity_vacuum,
            self.vpin,
            self.d_ofi_dt,
            self.d_mu_dt
        ])


class FeatureEngine:
    """
    Computes leading invariant features from orderbook and trade data.
    All features are designed to be predictive (leading) indicators.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of seconds to use for rolling calculations
        """
        self.window_size = window_size
        self._trade_history: List[Tuple[float, float, float, bool]] = []  # (ts, price, qty, is_buyer_maker)
        self._orderbook_history: List[Tuple[float, List[Tuple[float, float]], List[Tuple[float, float]]]] = []
        self._ofi_history: List[float] = []
        self._mu_history: List[float] = []
        
    def update_orderbook(
        self,
        timestamp: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ):
        """Update orderbook snapshot."""
        self._orderbook_history.append((timestamp, bids, asks))
        
        # Keep only recent history
        if len(self._orderbook_history) > self.window_size:
            self._orderbook_history.pop(0)
            
    def update_trade(self, timestamp: float, price: float, quantity: float, is_buyer_maker: bool):
        """Update trade history."""
        self._trade_history.append((timestamp, price, quantity, is_buyer_maker))
        
        # Keep only recent history
        if len(self._trade_history) > self.window_size * 10:  # Trades are more frequent
            self._trade_history.pop(0)
    
    def compute_ofi(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> float:
        """
        Compute Order Flow Imbalance.
        
        OFI_t = Σ v_i · s_i, where s_i ∈ {-1, +1}
        Positive OFI indicates buying pressure.
        """
        ofi = 0.0
        
        # Bid side: positive contribution (buying)
        for price, qty in bids:
            ofi += qty
            
        # Ask side: negative contribution (selling)
        for price, qty in asks:
            ofi -= qty
            
        self._ofi_history.append(ofi)
        if len(self._ofi_history) > self.window_size:
            self._ofi_history.pop(0)
            
        return ofi
    
    def compute_microprice_drift(
        self,
        trades: List[Tuple[float, float, float, bool]],
        mid_price: float
    ) -> float:
        """
        Compute Microprice Drift.
        
        μ_t = (Σ P_i · v_i) / (Σ v_i) - mid_t
        
        Measures deviation of volume-weighted trade price from mid-price.
        """
        if not trades:
            return 0.0
            
        total_value = 0.0
        total_volume = 0.0
        
        for ts, price, qty, _ in trades:
            total_value += price * qty
            total_volume += qty
            
        if total_volume == 0:
            return 0.0
            
        vwap = total_value / total_volume
        drift = vwap - mid_price
        
        self._mu_history.append(drift)
        if len(self._mu_history) > self.window_size:
            self._mu_history.pop(0)
            
        return drift
    
    def compute_liquidity_vacuum(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        prev_spread: Optional[float] = None
    ) -> float:
        """
        Compute Liquidity Vacuum Index.
        
        𝒱_t = max(q_top^b, q_top^a) / min(q_top^b, q_top^a) · Δspread / spread
        
        High values indicate thinning protection → high probability of breakout.
        """
        if not bids or not asks:
            return 0.0
            
        q_top_b = bids[0][1]  # Best bid quantity
        q_top_a = asks[0][1]  # Best ask quantity
        
        if q_top_b == 0 or q_top_a == 0:
            return float('inf')
            
        ratio = max(q_top_b, q_top_a) / min(q_top_b, q_top_a)
        
        spread = asks[0][0] - bids[0][0]
        
        if prev_spread is None or prev_spread == 0:
            delta_spread_ratio = 1.0
        else:
            delta_spread_ratio = abs(spread - prev_spread) / prev_spread
            
        return ratio * delta_spread_ratio
    
    def compute_vpin(
        self,
        trades: List[Tuple[float, float, float, bool]],
        n_buckets: int = 10
    ) -> float:
        """
        Compute VPIN (Volume-Synchronized Probability of Informed Trading).
        
        VPIN_t = (1/n) · Σ |V_buy,k - V_sell,k| / (V_buy,k + V_sell,k)
        
        Measures toxicity of order flow - high VPIN indicates informed trading.
        """
        if not trades or n_buckets <= 0:
            return 0.0
            
        # Bucket trades by volume
        bucket_size = sum(qty for _, _, qty, _ in trades) / n_buckets
        
        if bucket_size == 0:
            return 0.0
            
        vpin_sum = 0.0
        current_bucket_buy = 0.0
        current_bucket_sell = 0.0
        buckets_completed = 0
        
        for _, _, qty, is_buyer_maker in trades:
            if is_buyer_maker:
                current_bucket_sell += qty
            else:
                current_bucket_buy += qty
                
            total_bucket = current_bucket_buy + current_bucket_sell
            
            if total_bucket >= bucket_size:
                # Complete this bucket
                if total_bucket > 0:
                    diff = abs(current_bucket_buy - current_bucket_sell)
                    vpin_sum += diff / total_bucket
                    buckets_completed += 1
                    
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                
        if buckets_completed == 0:
            return 0.0
            
        return vpin_sum / buckets_completed
    
    def compute_features(
        self,
        timestamp: float,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        trades: Optional[List[Tuple[float, float, float, bool]]] = None
    ) -> Optional[FeatureSet]:
        """
        Compute all features for the current state.
        
        Returns None if insufficient data.
        """
        if not bids or not asks:
            return None
            
        # Get previous spread for delta calculation
        prev_spread = None
        if len(self._orderbook_history) >= 2:
            prev_bids = self._orderbook_history[-2][1]
            prev_asks = self._orderbook_history[-2][2]
            if prev_bids and prev_asks:
                prev_spread = prev_asks[0][0] - prev_bids[0][0]
                
        # Mid price
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Use provided trades or internal history
        if trades is None:
            trades = self._trade_history
            
        # Compute individual features
        ofi = self.compute_ofi(bids, asks)
        mu = self.compute_microprice_drift(trades, mid_price)
        vacuum = self.compute_liquidity_vacuum(bids, asks, prev_spread)
        vpin = self.compute_vpin(trades)
        
        # Compute derivatives (rate of change)
        d_ofi_dt = 0.0
        if len(self._ofi_history) >= 2:
            d_ofi_dt = self._ofi_history[-1] - self._ofi_history[-2]
            
        d_mu_dt = 0.0
        if len(self._mu_history) >= 2:
            d_mu_dt = self._mu_history[-1] - self._mu_history[-2]
            
        return FeatureSet(
            timestamp=timestamp,
            ofi=ofi,
            microprice_drift=mu,
            liquidity_vacuum=vacuum,
            vpin=vpin,
            d_ofi_dt=d_ofi_dt,
            d_mu_dt=d_mu_dt
        )
    
    def get_feature_dataframe(self) -> Optional[pl.DataFrame]:
        """Get feature history as Polars DataFrame."""
        if len(self._ofi_history) == 0:
            return None
            
        df = pl.DataFrame({
            'timestamp': list(range(len(self._ofi_history))),
            'OFI': self._ofi_history,
            'mu': self._mu_history
        })
        
        return df


def create_feature_pipeline(window_size: int = 100) -> FeatureEngine:
    """Factory function to create feature engine with specified window."""
    return FeatureEngine(window_size=window_size)


if __name__ == "__main__":
    # Example usage
    engine = FeatureEngine(window_size=50)
    
    # Simulate some orderbook data
    bids = [(99.9, 1.5), (99.8, 2.0), (99.7, 3.0)]
    asks = [(100.1, 1.2), (100.2, 2.5), (100.3, 1.8)]
    trades = [
        (1000.0, 100.0, 0.5, False),
        (1000.1, 100.05, 0.3, True),
        (1000.2, 100.02, 0.7, False)
    ]
    
    features = engine.compute_features(
        timestamp=1000.5,
        bids=bids,
        asks=asks,
        trades=trades
    )
    
    if features:
        print(f"Features: {features.to_dict()}")
        print(f"Array: {features.to_array()}")
