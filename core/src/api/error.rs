/// Unified API error type.
///
/// Every handler returns `Result<T, ApiError>` instead of inline `(StatusCode, Json(...))` tuples.
/// `ApiError` implements `IntoResponse` so Axum converts it automatically.
///
/// Benefits:
///   - One place to change error format for all endpoints
///   - `?` operator works on `anyhow::Error` and plugin errors
///   - Consistent JSON shape: `{ "error": "...", "code": "..." }`

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

// ── error variants ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum ApiError {
    /// 400 — malformed request, missing fields, invalid values
    BadRequest(String),
    /// 422 — request is well-formed but semantically invalid
    UnprocessableEntity(String),
    /// 503 — plugin server not reachable
    ServiceUnavailable(String),
    /// 502 — plugin call failed (embed / extract / generate)
    BadGateway(String),
    /// 404 — resource not found
    NotFound(String),
    /// 500 — unexpected internal error
    Internal(String),
}

// ── IntoResponse ─────────────────────────────────────────────────────────────

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            ApiError::BadRequest(msg)           => (StatusCode::BAD_REQUEST,            "bad_request",            msg),
            ApiError::UnprocessableEntity(msg)  => (StatusCode::UNPROCESSABLE_ENTITY,   "unprocessable_entity",   msg),
            ApiError::ServiceUnavailable(msg)   => (StatusCode::SERVICE_UNAVAILABLE,    "plugin_unavailable",     msg),
            ApiError::BadGateway(msg)           => (StatusCode::BAD_GATEWAY,            "bad_gateway",            msg),
            ApiError::NotFound(msg)             => (StatusCode::NOT_FOUND,              "not_found",              msg),
            ApiError::Internal(msg)             => (StatusCode::INTERNAL_SERVER_ERROR,  "internal_error",         msg),
        };

        tracing::error!(status = status.as_u16(), code, message);

        (status, Json(json!({ "error": message, "code": code }))).into_response()
    }
}

// ── conversions ───────────────────────────────────────────────────────────────

/// `anyhow::Error` → 500 Internal.
/// Handlers can use `?` directly on any `anyhow::Result`.
impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::Internal(e.to_string())
    }
}

/// `validator::ValidationErrors` → 400 Bad Request with field-level detail.
impl From<validator::ValidationErrors> for ApiError {
    fn from(e: validator::ValidationErrors) -> Self {
        ApiError::BadRequest(e.to_string())
    }
}

/// `axum_valid::HasValidate` rejection → surfaces validation errors as 400.
impl From<axum_valid::ValidRejection<axum::extract::rejection::JsonRejection>> for ApiError {
    fn from(e: axum_valid::ValidRejection<axum::extract::rejection::JsonRejection>) -> Self {
        ApiError::BadRequest(e.to_string())
    }
}

// ── constructors (ergonomic helpers) ─────────────────────────────────────────

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError::BadRequest(msg.into())
    }

    /// Use for plugin failures — surface the upstream error clearly.
    pub fn plugin(op: &str, e: anyhow::Error) -> Self {
        ApiError::BadGateway(format!("{op} failed: {e}"))
    }

    pub fn unprocessable(msg: impl Into<String>) -> Self {
        ApiError::UnprocessableEntity(msg.into())
    }
}
