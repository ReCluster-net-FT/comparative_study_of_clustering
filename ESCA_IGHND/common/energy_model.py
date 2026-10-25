import numpy as np
from . import config as cfg


def e_tx(m_bits, d, out=None):
    """Energy to transmit m_bits over distance d (vectorized over d)."""
    d = np.asarray(d, dtype=float)
    near = d < cfg.D0
    cost = np.where(
        near,
        m_bits * cfg.E_ELEC + m_bits * cfg.E_FS * d**2,
        m_bits * cfg.E_ELEC + m_bits * cfg.E_MP * d**4,
    )
    return cost


def e_rx(m_bits):
    return m_bits * cfg.E_ELEC


def e_da(m_bits, num_signals=1):

    return m_bits * cfg.E_DA * num_signals
