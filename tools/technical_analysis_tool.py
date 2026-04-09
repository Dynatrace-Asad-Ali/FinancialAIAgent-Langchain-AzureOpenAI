"""Technical analysis tool using yfinance price history."""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from core.exceptions import ToolExecutionError
from core.logging_config import setup_logging

logger = setup_logging()


class TechnicalAnalysisInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")


class TechnicalAnalysisTool(BaseTool):
    """Tool that performs technical analysis using yfinance price history."""

    name: str = "technical_analysis"
    description: str = (
        "Perform technical analysis for a publicly traded stock. "
        "Input should be a ticker symbol such as AAPL, MSFT, GOOGL, TSLA. "
        "Returns moving averages, RSI, MACD, support/resistance levels, "
        "volume trends, and a technical summary."
    )
    args_schema: Type[BaseModel] = TechnicalAnalysisInput

    def _run(
        self,
        symbol: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No price history found for '{symbol}'. Please verify the symbol."

            sections = [
                _price_summary(symbol.upper(), hist),
                _moving_averages(hist),
                _rsi_analysis(hist),
                _macd_analysis(hist),
                _support_resistance(hist),
                _volume_analysis(hist),
                _sector_comparison(ticker),
            ]

            report = f"TECHNICAL ANALYSIS: {symbol.upper()}\n{'=' * 50}\n\n"
            report += "\n\n".join(s for s in sections if s)
            return report

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}")
            raise ToolExecutionError(f"Technical analysis failed for {symbol}: {e}")

    async def _arun(self, symbol: str, run_manager=None) -> str:
        return self._run(symbol)


def _price_summary(symbol: str, hist: pd.DataFrame) -> str:
    current = hist["Close"].iloc[-1]
    week_ago = hist["Close"].iloc[-5] if len(hist) >= 5 else hist["Close"].iloc[0]
    month_ago = hist["Close"].iloc[-21] if len(hist) >= 21 else hist["Close"].iloc[0]
    year_ago = hist["Close"].iloc[0]

    w_chg = (current - week_ago) / week_ago * 100
    m_chg = (current - month_ago) / month_ago * 100
    y_chg = (current - year_ago) / year_ago * 100

    high_52w = hist["High"].max()
    low_52w = hist["Low"].min()
    pct_from_high = (current - high_52w) / high_52w * 100

    return (
        f"PRICE SUMMARY\n"
        f"  Current Price:      ${current:.2f}\n"
        f"  52-Week High:       ${high_52w:.2f} ({pct_from_high:+.1f}% from high)\n"
        f"  52-Week Low:        ${low_52w:.2f}\n"
        f"  1-Week Change:      {w_chg:+.2f}%\n"
        f"  1-Month Change:     {m_chg:+.2f}%\n"
        f"  1-Year Change:      {y_chg:+.2f}%"
    )


def _moving_averages(hist: pd.DataFrame) -> str:
    close = hist["Close"]
    current = close.iloc[-1]

    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None

    lines = ["MOVING AVERAGES"]

    def _ma_line(label, ma_val):
        if ma_val is None:
            return f"  {label}: N/A (insufficient data)"
        diff = (current - ma_val) / ma_val * 100
        signal = "above" if current > ma_val else "below"
        return f"  {label}: ${ma_val:.2f}  ({diff:+.1f}% {signal})"

    lines.append(_ma_line("20-Day MA ", ma20))
    lines.append(_ma_line("50-Day MA ", ma50))
    lines.append(_ma_line("200-Day MA", ma200))

    # Golden/death cross signal
    if ma50 is not None and ma200 is not None:
        if ma50 > ma200:
            lines.append("  Signal: GOLDEN CROSS — 50-day MA above 200-day MA (bullish)")
        else:
            lines.append("  Signal: DEATH CROSS — 50-day MA below 200-day MA (bearish)")

    return "\n".join(lines)


def _rsi_analysis(hist: pd.DataFrame, period: int = 14) -> str:
    close = hist["Close"]
    if len(close) < period + 1:
        return "RSI\n  Insufficient data to compute RSI."

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    if rsi >= 70:
        signal = "OVERBOUGHT — potential reversal or pullback"
    elif rsi <= 30:
        signal = "OVERSOLD — potential bounce or recovery"
    elif rsi >= 55:
        signal = "BULLISH momentum"
    elif rsi <= 45:
        signal = "BEARISH momentum"
    else:
        signal = "NEUTRAL"

    return f"RSI (14-Day)\n  RSI Value: {rsi:.1f}\n  Signal:    {signal}"


def _macd_analysis(hist: pd.DataFrame) -> str:
    close = hist["Close"]
    if len(close) < 35:
        return "MACD\n  Insufficient data to compute MACD."

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_val = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]
    hist_val = histogram.iloc[-1]
    prev_hist = histogram.iloc[-2]

    crossover = ""
    if macd_val > signal_val and macd_line.iloc[-2] <= signal_line.iloc[-2]:
        crossover = "  Crossover: BULLISH crossover just occurred"
    elif macd_val < signal_val and macd_line.iloc[-2] >= signal_line.iloc[-2]:
        crossover = "  Crossover: BEARISH crossover just occurred"

    momentum = "strengthening" if abs(hist_val) > abs(prev_hist) else "weakening"
    direction = "bullish" if hist_val > 0 else "bearish"

    lines = [
        "MACD (12/26/9)",
        f"  MACD Line:     {macd_val:.3f}",
        f"  Signal Line:   {signal_val:.3f}",
        f"  Histogram:     {hist_val:.3f} ({direction}, {momentum})",
    ]
    if crossover:
        lines.append(crossover)

    return "\n".join(lines)


def _support_resistance(hist: pd.DataFrame) -> str:
    close = hist["Close"]
    current = close.iloc[-1]

    # Use recent 6 months for cleaner levels
    recent = hist.tail(126)
    highs = recent["High"]
    lows = recent["Low"]

    # Find local maxima/minima as resistance/support
    resistance_levels = sorted(
        [highs.nlargest(5).iloc[i] for i in range(min(3, len(highs)))], reverse=True
    )
    support_levels = sorted(
        [lows.nsmallest(5).iloc[i] for i in range(min(3, len(lows)))]
    )

    lines = ["SUPPORT & RESISTANCE (6-Month)"]
    lines.append("  Resistance levels:")
    for r in resistance_levels:
        pct = (r - current) / current * 100
        lines.append(f"    ${r:.2f}  ({pct:+.1f}% from current)")

    lines.append("  Support levels:")
    for s in support_levels:
        pct = (s - current) / current * 100
        lines.append(f"    ${s:.2f}  ({pct:+.1f}% from current)")

    return "\n".join(lines)


def _volume_analysis(hist: pd.DataFrame) -> str:
    vol = hist["Volume"]
    avg_vol_3m = vol.tail(63).mean()
    avg_vol_10d = vol.tail(10).mean()
    latest_vol = vol.iloc[-1]

    vol_ratio = latest_vol / avg_vol_3m if avg_vol_3m > 0 else 0
    trend = "above" if avg_vol_10d > avg_vol_3m else "below"
    trend_pct = abs(avg_vol_10d - avg_vol_3m) / avg_vol_3m * 100 if avg_vol_3m > 0 else 0

    return (
        f"VOLUME ANALYSIS\n"
        f"  Latest Volume:          {latest_vol:,.0f}\n"
        f"  10-Day Avg Volume:      {avg_vol_10d:,.0f}\n"
        f"  3-Month Avg Volume:     {avg_vol_3m:,.0f}\n"
        f"  Latest vs 3M Avg:       {vol_ratio:.2f}x\n"
        f"  10-Day Trend:           {trend} 3-month avg by {trend_pct:.1f}%"
    )


def _sector_comparison(ticker: yf.Ticker) -> str:
    try:
        info = ticker.info
        beta = info.get("beta")
        sector = info.get("sector", "N/A")
        industry_pe = info.get("industryPE")
        trailing_pe = info.get("trailingPE")

        lines = [f"SECTOR CONTEXT\n  Sector: {sector}"]

        if beta is not None:
            if beta > 1.2:
                beta_note = "high volatility vs market"
            elif beta < 0.8:
                beta_note = "low volatility vs market"
            else:
                beta_note = "moderate volatility vs market"
            lines.append(f"  Beta: {beta:.2f} ({beta_note})")

        if trailing_pe and industry_pe:
            premium = (trailing_pe - industry_pe) / industry_pe * 100
            lines.append(f"  P/E vs Industry: {trailing_pe:.1f}x vs {industry_pe:.1f}x ({premium:+.1f}%)")

        return "\n".join(lines)
    except Exception:
        return ""
