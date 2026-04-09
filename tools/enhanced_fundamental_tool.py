"""Enhanced fundamental analysis tool using yfinance."""
import yfinance as yf
import pandas as pd
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from core.exceptions import ToolExecutionError
from core.logging_config import setup_logging

logger = setup_logging()


class FundamentalAnalysisInput(BaseModel):
    symbol: str = Field(description="Stock ticker symbol, e.g. AAPL, MSFT, TSLA")


class EnhancedFundamentalTool(BaseTool):
    """Tool that performs comprehensive fundamental analysis using yfinance."""

    name: str = "fundamental_analysis"
    description: str = (
        "Perform comprehensive fundamental analysis for a publicly traded stock. "
        "Input should be a ticker symbol such as AAPL, MSFT, GOOGL, TSLA. "
        "Returns valuation ratios, profitability metrics, balance sheet health, "
        "revenue growth, and an investment summary."
    )
    args_schema: Type[BaseModel] = FundamentalAnalysisInput

    def _run(
        self,
        symbol: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            if not info or info.get("symbol") is None:
                return f"Could not retrieve data for ticker '{symbol}'. Please verify the symbol."

            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow

            sections = [
                _company_overview(info),
                _valuation_metrics(info),
                _profitability_metrics(info),
                _growth_metrics(financials),
                _financial_health(info, balance_sheet, cash_flow),
                _analyst_estimates(info),
            ]

            report = f"FUNDAMENTAL ANALYSIS: {symbol.upper()}\n{'=' * 50}\n\n"
            report += "\n\n".join(s for s in sections if s)
            return report

        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            raise ToolExecutionError(f"Fundamental analysis failed for {symbol}: {e}")

    async def _arun(self, symbol: str, run_manager=None) -> str:
        return self._run(symbol)


def _fmt(value, prefix="", suffix="", decimals=2):
    """Format a numeric value, returning N/A if missing."""
    if value is None or value == 0:
        return "N/A"
    if isinstance(value, float):
        return f"{prefix}{value:,.{decimals}f}{suffix}"
    return f"{prefix}{value:,}{suffix}"


def _fmt_large(value, prefix="$"):
    """Format large numbers in billions/millions."""
    if value is None or value == 0:
        return "N/A"
    if abs(value) >= 1e12:
        return f"{prefix}{value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{prefix}{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{prefix}{value / 1e6:.2f}M"
    return f"{prefix}{value:,.0f}"


def _company_overview(info: dict) -> str:
    name = info.get("longName", info.get("shortName", "N/A"))
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")
    market_cap = _fmt_large(info.get("marketCap"))
    price = _fmt(info.get("currentPrice"), prefix="$")
    employees = info.get("fullTimeEmployees")
    employees_str = f"{employees:,}" if employees else "N/A"

    return (
        f"COMPANY OVERVIEW\n"
        f"  Name:        {name}\n"
        f"  Sector:      {sector}\n"
        f"  Industry:    {industry}\n"
        f"  Market Cap:  {market_cap}\n"
        f"  Price:       {price}\n"
        f"  Employees:   {employees_str}"
    )


def _valuation_metrics(info: dict) -> str:
    pe = _fmt(info.get("trailingPE"), suffix="x")
    fwd_pe = _fmt(info.get("forwardPE"), suffix="x")
    pb = _fmt(info.get("priceToBook"), suffix="x")
    ps = _fmt(info.get("priceToSalesTrailing12Months"), suffix="x")
    peg = _fmt(info.get("pegRatio"), suffix="x")
    ev_ebitda = _fmt(info.get("enterpriseToEbitda"), suffix="x")
    ev_rev = _fmt(info.get("enterpriseToRevenue"), suffix="x")

    return (
        f"VALUATION METRICS\n"
        f"  Trailing P/E:    {pe}\n"
        f"  Forward P/E:     {fwd_pe}\n"
        f"  Price/Book:      {pb}\n"
        f"  Price/Sales:     {ps}\n"
        f"  PEG Ratio:       {peg}\n"
        f"  EV/EBITDA:       {ev_ebitda}\n"
        f"  EV/Revenue:      {ev_rev}"
    )


def _profitability_metrics(info: dict) -> str:
    gross = _fmt(info.get("grossMargins", 0) * 100 if info.get("grossMargins") else None, suffix="%")
    op = _fmt(info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None, suffix="%")
    net = _fmt(info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None, suffix="%")
    roe = _fmt(info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None, suffix="%")
    roa = _fmt(info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None, suffix="%")
    revenue = _fmt_large(info.get("totalRevenue"))
    ebitda = _fmt_large(info.get("ebitda"))

    return (
        f"PROFITABILITY\n"
        f"  Revenue (TTM):      {revenue}\n"
        f"  EBITDA:             {ebitda}\n"
        f"  Gross Margin:       {gross}\n"
        f"  Operating Margin:   {op}\n"
        f"  Net Margin:         {net}\n"
        f"  Return on Equity:   {roe}\n"
        f"  Return on Assets:   {roa}"
    )


def _growth_metrics(financials: pd.DataFrame) -> str:
    if financials is None or financials.empty or len(financials.columns) < 2:
        return "GROWTH METRICS\n  Insufficient historical data available."

    lines = ["GROWTH METRICS"]
    for row_name, label in [
        ("Total Revenue", "Revenue YoY"),
        ("Gross Profit", "Gross Profit YoY"),
        ("Net Income", "Net Income YoY"),
    ]:
        try:
            if row_name in financials.index:
                current = financials.loc[row_name].iloc[0]
                previous = financials.loc[row_name].iloc[1]
                if previous and previous != 0:
                    growth = (current - previous) / abs(previous) * 100
                    lines.append(f"  {label}: {growth:+.1f}%")
        except Exception:
            pass

    return "\n".join(lines) if len(lines) > 1 else "GROWTH METRICS\n  Could not compute growth metrics."


def _financial_health(info: dict, balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> str:
    current_ratio = _fmt(info.get("currentRatio"), suffix="x")
    quick_ratio = _fmt(info.get("quickRatio"), suffix="x")
    debt_equity = _fmt(info.get("debtToEquity"), suffix="%")
    total_debt = _fmt_large(info.get("totalDebt"))
    total_cash = _fmt_large(info.get("totalCash"))
    free_cashflow = _fmt_large(info.get("freeCashflow"))
    op_cashflow = _fmt_large(info.get("operatingCashflow"))

    return (
        f"FINANCIAL HEALTH\n"
        f"  Current Ratio:      {current_ratio}\n"
        f"  Quick Ratio:        {quick_ratio}\n"
        f"  Debt/Equity:        {debt_equity}\n"
        f"  Total Debt:         {total_debt}\n"
        f"  Total Cash:         {total_cash}\n"
        f"  Operating Cash Flow:{op_cashflow}\n"
        f"  Free Cash Flow:     {free_cashflow}"
    )


def _analyst_estimates(info: dict) -> str:
    target_mean = _fmt(info.get("targetMeanPrice"), prefix="$")
    target_low = _fmt(info.get("targetLowPrice"), prefix="$")
    target_high = _fmt(info.get("targetHighPrice"), prefix="$")
    recommendation = info.get("recommendationKey", "N/A").upper()
    num_analysts = info.get("numberOfAnalystOpinions", "N/A")

    return (
        f"ANALYST ESTIMATES\n"
        f"  Recommendation:     {recommendation}\n"
        f"  # of Analysts:      {num_analysts}\n"
        f"  Price Target (avg): {target_mean}\n"
        f"  Price Target (low): {target_low}\n"
        f"  Price Target (high):{target_high}"
    )
