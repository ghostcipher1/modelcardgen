"""
Security validators for CLEAR project.

Provides utilities for:
- YAML file size validation to prevent DoS attacks
- URL validation with optional allowlist support
- Input sanitization and validation
"""

from typing import Optional
from urllib.parse import urlparse

__all__ = [
    "validate_yaml_file_size",
    "validate_url",
    "DEFAULT_MAX_YAML_SIZE",
]

DEFAULT_MAX_YAML_SIZE = 10 * 1024 * 1024


def validate_yaml_file_size(
    file_path: str, max_size: int = DEFAULT_MAX_YAML_SIZE
) -> None:
    """
    Validate that a YAML file doesn't exceed maximum size to prevent DoS attacks.

    Args:
        file_path: Path to the YAML file
        max_size: Maximum allowed file size in bytes (default: 10 MB)

    Raises:
        ValueError: If file exceeds maximum size
        FileNotFoundError: If file doesn't exist
    """
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValueError(
            f"YAML file exceeds maximum size limit: {file_size} bytes > {max_size} bytes. "
            f"This limit prevents denial-of-service attacks such as billion laughs attacks."
        )


def validate_url(
    url: str,
    allowed_schemes: Optional[list[str]] = None,
    blocked_hosts: Optional[list[str]] = None,
) -> bool:
    """
    Validate a URL for security concerns.

    Args:
        url: URL string to validate
        allowed_schemes: List of allowed schemes (default: ['http', 'https']).
                        If None, http and https are allowed.
        blocked_hosts: List of blocked hostnames (e.g., ['localhost', '127.0.0.1']).
                      If None, uses default private address ranges.

    Returns:
        True if URL passes validation

    Raises:
        ValueError: If URL fails validation
    """
    if not url:
        raise ValueError("URL cannot be empty")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")

    if allowed_schemes is None:
        allowed_schemes = ["http", "https"]

    if parsed.scheme not in allowed_schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Allowed schemes: {', '.join(allowed_schemes)}"
        )

    if not parsed.netloc:
        raise ValueError(f"URL missing hostname/netloc: {url}")

    if blocked_hosts is None:
        blocked_hosts = [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
        ]

    hostname = parsed.hostname or ""
    if hostname.lower() in [h.lower() for h in blocked_hosts]:
        raise ValueError(
            f"URL hostname '{hostname}' is not allowed. "
            f"URLs to local services are blocked for security reasons."
        )

    if (
        hostname.startswith("192.168.")
        or hostname.startswith("10.")
        or hostname.startswith("172.")
    ):
        raise ValueError(
            f"URL hostname '{hostname}' appears to be a private IP address. "
            f"Private IP ranges are not allowed for security reasons."
        )

    return True
