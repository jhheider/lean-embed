use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;

// Wire (de)serialization is unit-tested inside each `providers::*` module. These
// integration tests drive the public `Client` API against an offline mock.

// ---- Builder / config ----------------------------------------------------

#[test]
fn base_url_defaults_per_provider() {
    let ollama = Client::builder(Provider::Ollama, "m").build().unwrap();
    assert_eq!(ollama.base_url, "http://localhost:11434");
    let voyage = Client::builder(Provider::Voyage, "m")
        .api_key("k")
        .build()
        .unwrap();
    assert_eq!(voyage.base_url, "https://api.voyageai.com/v1");
}

#[test]
fn base_url_override_is_trimmed() {
    let c = Client::builder(Provider::Ollama, "m")
        .base_url("http://host:1234/")
        .build()
        .unwrap();
    assert_eq!(c.base_url, "http://host:1234");
    // Blank override falls back to the default.
    let c2 = Client::builder(Provider::Ollama, "m")
        .base_url("   ")
        .build()
        .unwrap();
    assert_eq!(c2.base_url, "http://localhost:11434");
}

#[test]
fn ollama_needs_no_key() {
    let c = Client::builder(Provider::Ollama, "m").build().unwrap();
    assert!(c.api_key.is_none());
    assert_eq!(c.provider(), Provider::Ollama);
}

#[test]
fn voyage_key_from_builder_wins() {
    let c = Client::builder(Provider::Voyage, "m")
        .api_key("explicit")
        .build()
        .unwrap();
    assert_eq!(c.api_key.as_deref(), Some("explicit"));
}

#[test]
fn voyage_without_key_errors_when_env_unset() {
    // Only meaningful when the ambient environment has no key; skip otherwise
    // so the suite stays deterministic without mutating process env.
    if std::env::var(VOYAGE_API_KEY_ENV).is_ok() {
        return;
    }
    let err = Client::builder(Provider::Voyage, "m").build().unwrap_err();
    assert!(matches!(err, Error::MissingApiKey { .. }));
}

// ---- HTTP round-trip (offline mock) --------------------------------------

#[tokio::test]
async fn ollama_round_trip_returns_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})),
        )
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "nomic-embed-text")
        .base_url(server.uri())
        .build()
        .unwrap();
    let v = client
        .embed(&["a".into(), "b".into()], EmbedKind::Document)
        .await
        .unwrap();
    assert_eq!(v, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
}

#[tokio::test]
async fn voyage_round_trip_sorts_and_sends_dim_and_type() {
    let server = MockServer::start().await;
    // The mock only matches if the body carried the query input_type and the
    // pinned dimension - so a match proves the wire fields were sent.
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_partial_json(
            json!({"input_type": "query", "output_dimension": 2}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"embedding": [9.0, 9.0], "index": 1},
                {"embedding": [1.0, 1.0], "index": 0}
            ]
        })))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Voyage, "voyage-3.5-lite")
        .api_key("k")
        .base_url(server.uri())
        .output_dimension(2)
        .build()
        .unwrap();
    let v = client
        .embed(&["x".into(), "y".into()], EmbedKind::Query)
        .await
        .unwrap();
    // Sorted back into input order despite the out-of-order response.
    assert_eq!(v, vec![vec![1.0, 1.0], vec![9.0, 9.0]]);
}

#[tokio::test]
async fn non_success_status_becomes_api_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(503).set_body_string("model loading"))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Document)
        .await
        .unwrap_err();
    match err {
        Error::Api {
            provider,
            status,
            body,
        } => {
            assert_eq!(provider, "ollama");
            assert_eq!(status, 503);
            assert!(body.contains("model loading"));
        }
        other => panic!("expected Api error, got {other:?}"),
    }
}

#[tokio::test]
async fn pinned_dimension_mismatch_is_caught() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        // Returns width 2 while the client pinned 1024.
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings": [[0.1, 0.2]]})))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .output_dimension(1024)
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Document)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::DimMismatch {
            got: 2,
            expected: 1024,
            ..
        }
    ));
}

#[tokio::test]
async fn count_mismatch_is_caught() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        // One vector for two inputs.
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings": [[0.1]]})))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into(), "b".into()], EmbedKind::Document)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::CountMismatch {
            got: 1,
            expected: 2,
            ..
        }
    ));
}

#[tokio::test]
async fn empty_input_short_circuits_without_a_request() {
    // No mock mounted: if embed hit the network this would error.
    let server = MockServer::start().await;
    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .build()
        .unwrap();
    let v = client.embed(&[], EmbedKind::Document).await.unwrap();
    assert!(v.is_empty());
}

#[tokio::test]
async fn max_batch_splits_requests_and_preserves_order() {
    let server = MockServer::start().await;
    // Each single-input request returns one vector; expect exactly 3 calls for
    // 3 inputs at max_batch(1).
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"embeddings": [[7.0]]})))
        .expect(3)
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .max_batch(1)
        .build()
        .unwrap();
    let v = client
        .embed(&["a".into(), "b".into(), "c".into()], EmbedKind::Document)
        .await
        .unwrap();
    assert_eq!(v, vec![vec![7.0], vec![7.0], vec![7.0]]);
    // `.expect(3)` is verified on drop of the server.
}

#[tokio::test]
async fn malformed_json_becomes_decode_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_string("definitely not json"))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Ollama, "m")
        .base_url(server.uri())
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Document)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Decode {
            provider: "ollama",
            ..
        }
    ));
}

#[tokio::test]
async fn voyage_malformed_json_becomes_decode_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{ nope"))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Voyage, "m")
        .api_key("k")
        .base_url(server.uri())
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Query)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Decode {
            provider: "voyage",
            ..
        }
    ));
}

#[tokio::test]
async fn unreachable_endpoint_becomes_request_error() {
    // Port 1 refuses immediately; a tight timeout also exercises the builder.
    let client = Client::builder(Provider::Ollama, "m")
        .base_url("http://127.0.0.1:1")
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Document)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Request {
            provider: "ollama",
            ..
        }
    ));
}

#[tokio::test]
async fn voyage_unreachable_endpoint_becomes_request_error() {
    let client = Client::builder(Provider::Voyage, "m")
        .api_key("k")
        .base_url("http://127.0.0.1:1")
        .build()
        .unwrap();
    let err = client
        .embed(&["a".into()], EmbedKind::Query)
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Request {
            provider: "voyage",
            ..
        }
    ));
}

/// The `Display` strings that thiserror derives, exercised for the variants a
/// caller is most likely to surface to a user.
#[test]
fn error_display_strings() {
    let api = Error::Api {
        provider: "voyage",
        status: 429,
        body: "rate limited".into(),
    };
    assert_eq!(api.to_string(), "voyage returned HTTP 429: rate limited");

    let dim = Error::DimMismatch {
        provider: "voyage",
        got: 512,
        expected: 1024,
    };
    assert_eq!(
        dim.to_string(),
        "voyage returned dimension 512 (expected 1024)"
    );

    let missing = Error::MissingApiKey {
        provider: "voyage",
        env: VOYAGE_API_KEY_ENV,
    };
    assert!(missing.to_string().contains("VOYAGE_API_KEY"));
}

// ---- OpenAI + Gemini builder / round-trip --------------------------------

#[test]
fn openai_and_gemini_base_url_defaults() {
    let openai = Client::builder(Provider::OpenAi, "m")
        .api_key("k")
        .build()
        .unwrap();
    assert_eq!(openai.base_url, "https://api.openai.com/v1");
    let gemini = Client::builder(Provider::Gemini, "m")
        .api_key("k")
        .build()
        .unwrap();
    assert_eq!(
        gemini.base_url,
        "https://generativelanguage.googleapis.com/v1beta"
    );
}

#[test]
fn keyed_providers_error_without_a_key_when_env_unset() {
    for (provider, env) in [
        (Provider::OpenAi, OPENAI_API_KEY_ENV),
        (Provider::Gemini, GEMINI_API_KEY_ENV),
    ] {
        if std::env::var(env).is_ok() {
            continue; // keep deterministic without mutating process env
        }
        let err = Client::builder(provider, "m").build().unwrap_err();
        assert!(matches!(err, Error::MissingApiKey { .. }));
    }
}

#[tokio::test]
async fn openai_round_trip_sorts_and_sends_dimensions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(body_partial_json(
            json!({"encoding_format": "float", "dimensions": 2}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"embedding": [9.0, 9.0], "index": 1},
                {"embedding": [1.0, 1.0], "index": 0}
            ]
        })))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::OpenAi, "text-embedding-3-small")
        .api_key("k")
        .base_url(server.uri())
        .output_dimension(2)
        .build()
        .unwrap();
    let v = client
        .embed(&["x".into(), "y".into()], EmbedKind::Document)
        .await
        .unwrap();
    // Sorted back into input order despite the out-of-order response.
    assert_eq!(v, vec![vec![1.0, 1.0], vec![9.0, 9.0]]);
}

#[tokio::test]
async fn gemini_round_trip_uses_header_auth_and_task_type() {
    let server = MockServer::start().await;
    // Model is `models/`-prefixed in the path; auth is the x-goog-api-key header;
    // the body carries the query taskType and pinned dimensionality.
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-004:batchEmbedContents"))
        .and(header("x-goog-api-key", "secret"))
        .and(body_partial_json(json!({
            "requests": [{"taskType": "RETRIEVAL_QUERY", "outputDimensionality": 3}]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [{"values": [1.0, 2.0, 3.0]}]
        })))
        .mount(&server)
        .await;

    let client = Client::builder(Provider::Gemini, "text-embedding-004")
        .api_key("secret")
        .base_url(server.uri())
        .output_dimension(3)
        .build()
        .unwrap();
    let v = client
        .embed(&["a question".into()], EmbedKind::Query)
        .await
        .unwrap();
    assert_eq!(v, vec![vec![1.0, 2.0, 3.0]]);
}

#[tokio::test]
async fn gemini_accepts_already_prefixed_model() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-004:batchEmbedContents"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"embeddings": [{"values": [0.5]}]})),
        )
        .mount(&server)
        .await;
    // Passing the model already `models/`-prefixed must not double-prefix.
    let client = Client::builder(Provider::Gemini, "models/text-embedding-004")
        .api_key("k")
        .base_url(server.uri())
        .build()
        .unwrap();
    let v = client
        .embed(&["x".into()], EmbedKind::Document)
        .await
        .unwrap();
    assert_eq!(v, vec![vec![0.5]]);
}

#[test]
fn api_key_is_redacted_in_debug() {
    let client = Client::builder(Provider::OpenAi, "m")
        .api_key("super-secret-key")
        .build()
        .unwrap();
    let dbg = format!("{client:?}");
    assert!(
        !dbg.contains("super-secret-key"),
        "key leaked in Debug: {dbg}"
    );
    assert!(dbg.contains("<redacted>"));
}
