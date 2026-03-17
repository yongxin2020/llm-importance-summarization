"""
Shared utility functions for probe training scripts

This module contains common helper functions used across all probe training scripts:
- Metric calculation functions (NDCG, KL divergence, Rényi divergence)
- GPU time tracking utilities
"""

import torch
import numpy as np
import time


# --- Metric Helper Functions ---

def ndcg_at_k(pred_scores: np.ndarray, ref_gain: np.ndarray, k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K (NDCG@k)
    
    Args:
        pred_scores: Predicted importance scores (1D array)
        ref_gain: Reference (ground truth) importance scores as gains (1D array)
        k: Cutoff position for evaluation
        
    Returns:
        NDCG@k score in range [0, 1], where 1 is perfect ranking
    """
    if len(pred_scores) == 0:
        return 0.0
    k = min(k, len(pred_scores))
    if k == 0:
        return 0.0
        
    # Get top-k predictions by predicted scores
    order = np.argsort(-pred_scores)[:k]
    gains = ref_gain[order]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** gains - 1) / discounts)
    
    # Compute ideal DCG
    ideal = np.sort(ref_gain)[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / discounts)
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def renyi_divergence(p: np.ndarray, q: np.ndarray, alpha: float = 2.0) -> float:
    """
    Compute Rényi divergence of order alpha: D_α(P||Q)
    
    Rényi divergence is a generalization of KL divergence.
    When alpha → 1, it approaches KL divergence.
    When alpha = 2, it's called collision divergence.
    
    Args:
        p: Probability distribution P (will be normalized)
        q: Probability distribution Q (will be normalized)
        alpha: Order of Rényi divergence (default: 2.0)
        
    Returns:
        Rényi divergence value (≥ 0)
    """
    EPS = 1e-12
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to probability distributions
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    
    # Add epsilon to q to avoid division by zero
    q = np.where(q == 0, EPS, q)
    
    # Compute: D_α(P||Q) = 1/(α-1) * log(sum(p^α * q^(1-α)))
    inner = np.sum((p ** alpha) * (q ** (1.0 - alpha)))
    if inner <= 0:
        return float('inf')
    return float((1.0 / (alpha - 1.0)) * np.log(inner))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Kullback-Leibler divergence: KL(P||Q)
    
    KL divergence measures how one probability distribution diverges from
    a second, expected probability distribution.
    
    Args:
        p: Probability distribution P (will be normalized)
        q: Probability distribution Q (will be normalized)
        
    Returns:
        KL divergence value (≥ 0)
        
    Note:
        KL(P||Q) ≠ KL(Q||P) (not symmetric)
        KL(P||Q) = 0 iff P = Q
    """
    EPS = 1e-12
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to probability distributions
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    
    # KL(P||Q) = sum(P * log(P/Q))
    q = np.where(q == 0, EPS, q)
    return float(np.sum(p * np.log((p + EPS) / q)))


# --- GPU Time Tracking ---

class GPUTimeTracker:
    """
    Track GPU usage time during training sessions
    
    Features:
    - Tracks cumulative GPU time across multiple sessions
    - Handles both GPU and CPU execution
    - Provides formatted time reporting
    
    Usage:
        tracker = GPUTimeTracker()
        tracker.start_session("training")
        # ... training code ...
        tracker.end_session()
        print(f"Total time: {tracker.format_total_time()}")
    """
    
    def __init__(self):
        """Initialize GPU time tracker"""
        self.start_time = None
        self.end_time = None
        self.is_gpu_available = torch.cuda.is_available()
        self.device_name = torch.cuda.get_device_name(0) if self.is_gpu_available else "CPU"
        self.total_gpu_time = 0.0
        self.session_times = []
        
    def start_session(self, session_name: str = "training"):
        """
        Start a new timing session
        
        Args:
            session_name: Descriptive name for this session
        """
        if self.is_gpu_available:
            torch.cuda.synchronize()  # Ensure all GPU operations are complete
        self.start_time = time.time()
        self.session_name = session_name
        print(f"🔥 Starting GPU session: {session_name} on {self.device_name}")
        
    def end_session(self):
        """End the current timing session and accumulate time"""
        if self.start_time is None:
            print("Warning: No active session to end")
            return
            
        if self.is_gpu_available:
            torch.cuda.synchronize()
        self.end_time = time.time()
        
        session_duration = self.end_time - self.start_time
        self.total_gpu_time += session_duration
        self.session_times.append({
            'name': self.session_name,
            'duration': session_duration
        })
        
        print(f"✅ Session '{self.session_name}' completed: {self.format_time(session_duration)}")
        self.start_time = None
        
    def format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable time string
        
        Args:
            seconds: Time duration in seconds
            
        Returns:
            Formatted string (e.g., "1h 23m 45s")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
            
    def format_total_time(self) -> str:
        """Get formatted total GPU time across all sessions"""
        return self.format_time(self.total_gpu_time)
        
    def get_summary(self) -> dict:
        """
        Get summary of all timing sessions
        
        Returns:
            Dictionary with total time and per-session breakdown
        """
        return {
            'total_time': self.total_gpu_time,
            'total_time_formatted': self.format_total_time(),
            'device': self.device_name,
            'sessions': self.session_times
        }
