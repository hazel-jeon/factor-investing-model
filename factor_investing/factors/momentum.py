"""
factors/momentum.py
-------------------
Momentum factor — multi-lookback ensemble.

기본 스펙 (Jegadeesh & Titman 1993):
    단일 12-1M 룩백 → Score = normalise( return[t-12M : t-1M] )

앙상블 스펙 (기본값):
    세 룩백 기간의 z-score를 가중 평균합니다.

    ┌──────────┬────────┬─────────────────────────────────────────┐
    │ 구간     │ 가중치 │ 포착하는 것                              │
    ├──────────┼────────┼─────────────────────────────────────────┤
    │ 12-1M    │  0.50  │ 중장기 추세 (Fama-French 클래식)        │
    │  6-1M    │  0.30  │ 중기 모멘텀 (반응 속도↑)                │
    │  3-1M    │  0.20  │ 단기 모멘텀 (최근 강도 반영)            │
    └──────────┴────────┴─────────────────────────────────────────┘

    단일 12-1M 신호 대비 두 가지 이점:
    1. 신호 다각화 — 각 구간이 서로 다른 주기의 가격 정보를 담아
       합산 시 노이즈가 상쇄됩니다.
    2. 반응 속도 — 6-1M / 3-1M 구간이 추세 전환을 더 빠르게 반영해
       극단적 낙폭 구간의 손실을 일부 줄여줍니다.

사용법:
    # 클래식 단일 12-1M (하위 호환)
    MomentumFactor(ensemble=False)

    # 앙상블 (기본)
    MomentumFactor()

    # 가중치 직접 지정
    MomentumFactor(
        lookback_configs=[(12, 1, 0.5), (6, 1, 0.3), (3, 1, 0.2)]
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseFactor

logger = logging.getLogger(__name__)

_DAYS_PER_MONTH = 21

# 기본 앙상블 구성: (lookback_months, skip_months, weight)
_DEFAULT_ENSEMBLE: list[tuple[int, int, float]] = [
    (12, 1, 0.50),
    ( 6, 1, 0.30),
    ( 3, 1, 0.20),
]


class MomentumFactor(BaseFactor):
    """
    Multi-lookback momentum factor (앙상블 기본).

    Parameters
    ----------
    ensemble : bool
        True(기본) = 앙상블, False = 단일 12-1M (클래식).
    lookback_configs : list of (lookback_months, skip_months, weight)
        앙상블 구성 직접 지정 시 사용.  ensemble=True일 때만 적용.
    lookback_months : int
        ensemble=False일 때 단일 룩백 기간 (기본 12).
    skip_months : int
        ensemble=False일 때 단기 반전 회피 기간 (기본 1).
    """

    name = "momentum"

    def __init__(
        self,
        ensemble: bool = True,
        lookback_configs: Optional[list[tuple[int, int, float]]] = None,
        # 단일 모드 파라미터 (하위 호환)
        lookback_months: int = 12,
        skip_months: int = 1,
    ) -> None:
        self.ensemble = ensemble

        if ensemble:
            configs = lookback_configs or _DEFAULT_ENSEMBLE
            # 가중치 정규화
            total_w = sum(w for _, _, w in configs)
            self.configs = [(lb, sk, w / total_w) for lb, sk, w in configs]
            logger.debug(
                "MomentumFactor ensemble: %s",
                [(lb, sk, f"{w:.2f}") for lb, sk, w in self.configs],
            )
        else:
            # 단일 룩백 — 기존 동작 유지
            self.configs = [(lookback_months, skip_months, 1.0)]

    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        as_of: Optional[pd.Timestamp] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted-close price panel (rows = dates, columns = tickers).
        as_of : pd.Timestamp, optional
            기준일.  기본값 = prices 마지막 행.

        Returns
        -------
        pd.Series
            앙상블 모멘텀 점수 (ticker 인덱스).  높을수록 강한 모멘텀.
        """
        if as_of is None:
            as_of = prices.index[-1]

        prices_so_far = prices.loc[:as_of]

        component_scores: list[pd.Series] = []
        weights: list[float] = []

        for lookback_months, skip_months, weight in self.configs:
            score = self._single_momentum(
                prices_so_far,
                lookback_months=lookback_months,
                skip_months=skip_months,
            )
            if score is not None and not score.empty:
                component_scores.append(score)
                weights.append(weight)

        if not component_scores:
            logger.warning("MomentumFactor: no valid component scores as of %s", as_of)
            return pd.Series(dtype=float, name=self.name)

        # 공통 종목만 앙상블
        common_idx = component_scores[0].index
        for s in component_scores[1:]:
            common_idx = common_idx.intersection(s.index)

        if common_idx.empty:
            return pd.Series(dtype=float, name=self.name)

        # 가중합 (재정규화)
        total_w = sum(weights)
        composite = sum(
            s.loc[common_idx] * (w / total_w)
            for s, w in zip(component_scores, weights)
        )

        # 최종 전체 z-score
        result = self.cross_sectional_zscore(composite)
        return result.rename(self.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_momentum(
        self,
        prices: pd.DataFrame,
        lookback_months: int,
        skip_months: int,
    ) -> Optional[pd.Series]:
        """
        단일 룩백 구간의 정규화된 수익률 시리즈를 반환.
        히스토리가 부족하면 None 반환.
        """
        lookback_days = lookback_months * _DAYS_PER_MONTH
        skip_days     = skip_months     * _DAYS_PER_MONTH

        n = len(prices)
        if n < lookback_days + skip_days:
            logger.debug(
                "MomentumFactor(%dM-%dM): only %d rows, need %d",
                lookback_months, skip_months, n, lookback_days + skip_days,
            )
            return None

        # t-skip 시점 가격
        idx_t    = -(skip_days)
        idx_base = -(lookback_days + skip_days)

        price_t    = prices.iloc[idx_t]
        price_base = prices.iloc[idx_base]

        ret = (price_t / price_base - 1).replace([np.inf, -np.inf], np.nan).dropna()

        if len(ret) < 5:
            return None

        return self.normalise(ret)