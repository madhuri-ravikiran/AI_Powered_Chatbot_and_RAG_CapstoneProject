"""Centralize frontend-to-FastAPI HTTP calls and friendly transport errors."""

from typing import Any

import httpx

UNAVAILABLE = "Backend service is currently unavailable. Please start the FastAPI server."


class ApiError(Exception):
    """A user-safe backend or network failure suitable for display in Streamlit."""


class ApiClient:
    """Reusable API adapter with connection pooling and bounded timeouts."""

    def __init__(self, base_url: str, client: httpx.Client | None = None):
        """Use an injected HTTP transport in tests or a normal pooled client."""
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(300, connect=5),
            trust_env=False,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        """Normalize connection, HTTP, and JSON failures without showing stack traces."""
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.ConnectError:
            raise ApiError(UNAVAILABLE) from None
        except httpx.TimeoutException:
            raise ApiError(
                "The request timed out. Check document status before retrying processing."
            ) from None
        except httpx.HTTPError:
            raise ApiError("The backend connection failed. Please try again.") from None
        try:
            payload = response.json()
        except ValueError:
            raise ApiError("The backend returned an unreadable response.") from None
        if response.is_error:
            # Only display the stable error contract, never arbitrary reverse-proxy HTML.
            message = payload.get("message") if isinstance(payload, dict) else None
            raise ApiError(message if isinstance(message, str) else "The backend request failed.")
        if not isinstance(payload, dict):
            raise ApiError("The backend returned an unexpected response.")
        return payload

    def health_check(self) -> dict:
        """Read liveness with a short timeout so an offline backend cannot freeze the UI."""
        return self._request("GET", "/health", timeout=5)

    def get_config(self) -> dict:
        """Fetch non-secret UI settings once per frontend session."""
        return self._request("GET", "/api/config")

    def send_chat_message(
        self, session_id: str, message: str, mode: str, top_k: int, temperature: float
    ) -> dict:
        """Submit one validated conversation turn to the backend."""
        return self._request(
            "POST",
            "/api/chat",
            json={
                "session_id": session_id,
                "message": message,
                "mode": mode,
                "top_k": top_k,
                "temperature": temperature,
            },
        )

    def upload_documents(self, session_id: str, files: list[tuple[str, bytes]]) -> dict:
        """Send multiple files as one multipart transaction."""
        return self._request(
            "POST",
            "/api/documents/upload",
            data={"session_id": session_id},
            files=[("files", (name, data)) for name, data in files],
        )

    def process_documents(self, session_id: str) -> dict:
        """Ask the backend to embed only documents still marked uploaded."""
        return self._request("POST", "/api/documents/process", json={"session_id": session_id})

    def get_documents(self, session_id: str) -> dict:
        """List the current session's knowledge base."""
        return self._request("GET", "/api/documents", params={"session_id": session_id})

    def clear_documents(self, session_id: str) -> dict:
        """Delete the current session's knowledge base."""
        return self._request("DELETE", "/api/documents", params={"session_id": session_id})

    def clear_chat(self, session_id: str) -> dict:
        """Clear memory without deleting uploaded or indexed documents."""
        return self._request("POST", "/api/chat/clear", json={"session_id": session_id})

    def close(self) -> None:
        """Release the HTTP connection pool when the client is no longer needed."""
        self.client.close()
