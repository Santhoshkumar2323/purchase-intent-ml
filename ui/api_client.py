import os
import requests

BASE_URL = os.environ.get("PURCHASE_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 90  


class ApiError(Exception):
    """Something went wrong talking to the API; the message is safe to show a user."""


def _detail(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # 422 validation errors
        parts = [f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail]
        detail = "; ".join(parts)
    return f"API error {resp.status_code}: {detail or resp.text[:200]}"


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    try:
        resp = requests.request(method, f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.ConnectionError as err:
        raise ApiError(f"Cannot reach the API at {BASE_URL}. Is it running?") from err
    except requests.Timeout as err:
        raise ApiError("The API took too long to respond. Try again in a moment.") from err
    except requests.RequestException as err:
        raise ApiError(f"Request failed: {err}") from err
    if resp.status_code != 200:
        raise ApiError(_detail(resp))
    return resp.json()


def health() -> dict:
    return _call("GET", "/health")


def predict(payload: dict) -> dict:
    return _call("POST", "/predict", payload)


def explain(payload: dict) -> dict:
    return _call("POST", "/explain", payload)