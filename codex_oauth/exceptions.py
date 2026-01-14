class CodexOAuthError(Exception):
    """Base error for this project."""


class NotAuthenticatedError(CodexOAuthError):
    """Raised when no OAuth credentials are available."""


class OAuthFlowError(CodexOAuthError):
    """Raised when the OAuth flow fails."""


class TokenRefreshError(CodexOAuthError):
    """Raised when refresh token exchange fails."""


class CodexAPIError(CodexOAuthError):
    """Raised when the Codex backend returns an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
