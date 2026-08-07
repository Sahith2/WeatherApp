"""
Client for the National Weather Service (NWS) API.

Provides helper methods for retrieving weather alerts and forecasts
from api.weather.gov.
"""

import os
from typing import Any

import requests


_BASE_URL = os.environ.get(
    
    "WEATHER_API_BASE_URL", 
    "https://api.weather.gov"
)

_DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Thin wrapper around the National Weather Service (NWS) API with auth + retry-friendly session."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                 "User-Agent": "TicketHubWeatherApp",
                 "Accept": "application/geo+json"
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
    
    def get_points(self, latitude: float, longitude: float) -> dict:
        """
        Resolve a latitude/longitude to the NWS grid point.
        Returns gridId, gridX, and gridY needed for forecasts.
        """
        return self.get(f"/points/{latitude},{longitude}")
    
    
    def get_alerts(self, area: str) -> dict:
        """
        Get active weather alerts for a U.S. state or area.
        Example: "IL", "TX", "GA"
        """
        return self.get("/alerts/active", params={"area": area})
    
    def get_forecast(self, grid_id: str, grid_x: int, grid_y: int) -> dict:
        """
        Get the weather forecast for a grid point.
        """
        return self.get(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast")


   