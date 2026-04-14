from __future__ import annotations

import os
import json
import requests
from typing import Type

import yfinance as yf
from pydantic import BaseModel, Field
from sec_api import QueryApi, ExtractorApi
from crewai.tools import BaseTool


class StockPriceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")


class SecFilingsInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. NVDA")
    form_type: str = Field(default="10-K", description="SEC form type like 10-K or 10-Q")
    limit: int = Field(default=3, description="Number of filings to return")


class SecSectionInput(BaseModel):
    filing_url: str = Field(..., description="SEC filing details URL or filing HTML URL")
    section: str = Field(default="1A", description="Section identifier like 1, 1A, 7, 7A")


class StockSnapshotTool(BaseTool):
    name: str = "stock_snapshot"
    description: str = (
        "Get a concise market snapshot for a public stock using yfinance. "
        "Returns price, market cap, valuation, growth, profitability, and balance sheet fields when available."
    )
    args_schema: Type[BaseModel] = StockPriceInput

    def _run(self, ticker: str) -> str:
        t = yf.Ticker(ticker)
        info = t.info or {}
        fast = getattr(t, "fast_info", {}) or {}
        payload = {
            "ticker": ticker.upper(),
            "current_price": fast.get("lastPrice") or info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
            "forward_pe": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "price_to_book": info.get("priceToBook"),
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "free_cashflow": info.get("freeCashflow"),
            "operating_cashflow": info.get("operatingCashflow"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "debt_to_equity": info.get("debtToEquity"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "analyst_target_mean": info.get("targetMeanPrice"),
            "business_summary": info.get("longBusinessSummary"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
        }
        return json.dumps(payload, indent=2, default=str)


class SecFilingsTool(BaseTool):
    name: str = "sec_filings_search"
    description: str = (
        "Search recent SEC filings for a ticker using SEC-API. "
        "Useful for fetching the latest 10-K, 10-Q, or 8-K metadata and filing URLs."
    )
    args_schema: Type[BaseModel] = SecFilingsInput

    def _run(self, ticker: str, form_type: str = "10-K", limit: int = 3) -> str:
        query_api = QueryApi(api_key=self._get_sec_api_key())
        query = {
            "query": {"query_string": {"query": f"ticker:{ticker.upper()} AND formType:{form_type}"}},
            "from": "0",
            "size": str(limit),
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        res = query_api.get_filings(query)
        filings = []
        for item in res.get("filings", []):
            if item.get("formType") != form_type:
                continue
            filings.append({
                "ticker": item.get("ticker"),
                "companyName": item.get("companyName"),
                "formType": item.get("formType"),
                "filedAt": item.get("filedAt"),
                "linkToFilingDetails": item.get("linkToFilingDetails"),
                "linkToTxt": item.get("linkToTxt"),
                "linkToHtml": item.get("linkToHtml"),
            })
        return json.dumps(filings, indent=2)

    @staticmethod
    def _get_sec_api_key() -> str:
        import os
        key = os.getenv("SEC_API_KEY")
        if not key:
            raise ValueError("SEC_API_KEY environment variable is required")
        return key


class SecSectionTool(BaseTool):
    name: str = "sec_filing_section"
    description: str = (
        "Extract a section from an SEC filing using SEC-API Extractor. "
        "Useful for risk factors (1A), business overview (1), and MD&A (7)."
    )
    args_schema: Type[BaseModel] = SecSectionInput

    def _run(self, filing_url: str, section: str = "1A") -> str:
        extractor = ExtractorApi(self._get_sec_api_key())
        text = extractor.get_section(filing_url, section, "text")
        if not text:
            return "No section text returned."
        return text[:12000]

    @staticmethod
    def _get_sec_api_key() -> str:
        import os
        key = os.getenv("SEC_API_KEY")
        if not key:
            raise ValueError("SEC_API_KEY environment variable is required")
        return key


class SerpApiSearchInput(BaseModel):
    query: str = Field(..., description="Google search query")
    news: bool = Field(default=False, description="Whether to search Google News")


class SerpApiSearchTool(BaseTool):
    name: str = "serpapi_search"
    description: str = "Search Google or Google News using SerpApi."

    args_schema: Type[BaseModel] = SerpApiSearchInput

    def _run(self, query: str, news: bool = False) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is required")

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": 5
        }

        if news:
            params["tbm"] = "nws"

        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        return json.dumps({
            "organic_results": data.get("organic_results", [])[:5],
            "news_results": data.get("news_results", [])[:5],
            "search_information": data.get("search_information", {})
        }, indent=2)
