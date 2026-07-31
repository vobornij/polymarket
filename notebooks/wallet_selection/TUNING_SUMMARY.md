# Parameter Tuning Summary

## VWAP_BUCKET_MIN (default: 5min)

| Bucket | Val IC | Train IC | Test IC | Consistent? |
|--------|--------|----------|---------|-------------|
| 1min | 0.0083 | 0.0016 | 0.0020 | ✓ |
| **2min** | **0.0148** | **-0.0014** | **-0.0091** | **✗ flips** |
| 3min | 0.0017 | 0.0000 | -0.0034 | ✓ |
| **5min** | **0.0103** | **0.0058** | **0.0095** | **✓ best** |
| 7min | 0.0051 | 0.0019 | -0.0068 | ✓ |
| 10min | 0.0077 | -0.0049 | -0.0006 | ✗ flips |
| 15min | 0.0118 | -0.0046 | 0.0081 | ✗ flips |
| 20min | -0.0007 | -0.0079 | 0.0046 | ✓ but weak |
| 30min | -0.0015 | -0.0006 | -0.0027 | ✓ but weak |

**Conclusion**: 5min is the only bucket with |Val IC| >= 0.01 AND consistent sign across Train/Val/Test. Default confirmed optimal.

## SIGNAL_WINDOW_MIN (default: 15min)

| Window | Sig | Val IC | Train IC | Test IC |
|--------|-----|--------|----------|---------|
| 10min | mkt_net_vol | -0.0111 | 0.0057 | 0.0046 |
| 10min | mkt_wallet_side_entropy | 0.0088 | 0.0019 | 0.0034 |
| 15min | mkt_net_vol | 0.0127 | 0.0008 | -0.0013 |
| 15min | mkt_wallet_buy_share | 0.0082 | 0.0007 | -0.0014 |
| 30min | mkt_wallet_side_entropy | -0.0152 | 0.0026 | -0.0046 |
| *all* | sig_vwap_copybuy_signed | 0.0103 | 0.0058 | 0.0095 |

VWAP signal is invariant to SIGNAL_WINDOW_MIN (uses separate VWAP_BUCKET_MIN). Market signals vary: 10min best for mkt_net_vol, 30min best for mkt_wallet_side_entropy (|0.0152|). Default 15min remains reasonable.

## QW_TOP_PCT (default: 20%)

| Top % | QW wallets | sig_qw_proximity | qw_wallet_count | cs_qw_trade_count |
|-------|-----------|-----------------|-----------------|-------------------|
| 10% | 358 | 0.0077 ✓ | 0.0028 | 0.0003 ✓ |
| 15% | 537 | -0.0032 ✗ | 0.0065 | 0.0023 ✗ |
| **20%** | **716** | **0.0055** ✓ | **0.0073** | **-0.0011** ✗ |

20% gives strongest qw_wallet_count (0.0073). 10% gives best proximity signal (0.0077). Default 20% acceptable.

## Top Signals (81 tested)

| Signal | Val IC | Train IC | Test IC | Consistent | Coverage |
|--------|--------|----------|---------|------------|----------|
| **qw_net_wallet_imbalance** | **-0.0157** | **-0.0119** | **-0.0030** | **✓** | 22K (37%) |
| cs_mkt_price_momentum | -0.0128 | -0.0035 | 0.0010 | ✓ | 59K (100%) |
| **sig_vwap_copybuy_signed** | **0.0103** | **0.0058** | **0.0095** | **✓** | 25K (42%) |
| cs_qw_trade_count | 0.0101 | 0.0027 | -0.0086 | ✓ | 59K (100%) |
| sig_vwap_pssell_signed | -0.0100 | -0.0205 | -0.0042 | ✓ | 3K (5%) |

**Forward selection**: qw_net_wallet_imbalance → +sig_vwap_pssell_signed (cond IC=0.0037)

**Copy application** (threshold -0.50): Test ROI=0.0207 vs baseline 0.0188




TODO: 
- position discrepancy: market makers vs good predictors