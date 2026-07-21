//! A lean, provider-agnostic text-embeddings client.
//!
//! One [`Client`] turns batches of text into vectors against any of four
//! [`Provider`]s - **[Voyage AI]**, **[OpenAI]** (or any OpenAI-compatible
//! endpoint), **[Gemini]**, or **[Ollama]** (local, no key, offline) - behind a
//! single [`embed`](Client::embed) call. [`EmbedKind`] selects query- vs
//! document-side vectors where the provider supports it, and an optional
//! [`output_dimension`](ClientBuilder::output_dimension) is requested *and*
//! validated so a model drift can't silently desync a fixed-width column.
//!
//! The wire is [`reqwest`] on **rustls + ring** only - never OpenSSL or aws-lc -
//! so the dependency tree stays small and cross-compiles cleanly (musl,
//! aarch64). That lean stack is the reason this crate exists instead of a full
//! agent/RAG framework: it is *only* the embeddings HTTP client, so a vector
//! store, chunking, and retrieval stay in the caller where they belong.
//!
//! # Example
//!
//! ```no_run
//! use lean_embed::{Client, EmbedKind, Provider};
//!
//! # async fn run() -> Result<(), lean_embed::Error> {
//! // Local Ollama - no key, offline.
//! let client = Client::builder(Provider::Ollama, "nomic-embed-text").build()?;
//! let vectors = client
//!     .embed(&["hello".into(), "world".into()], EmbedKind::Document)
//!     .await?;
//! assert_eq!(vectors.len(), 2);
//!
//! // Hosted Voyage, pinned to 1024 dimensions (key from VOYAGE_API_KEY).
//! let voyage = Client::builder(Provider::Voyage, "voyage-3.5-lite")
//!     .output_dimension(1024)
//!     .max_batch(96)
//!     .build()?;
//! let q = voyage.embed(&["a question".into()], EmbedKind::Query).await?;
//! # let _ = (vectors, q);
//! # Ok(())
//! # }
//! ```
//!
//! [Voyage AI]: https://www.voyageai.com/
//! [OpenAI]: https://platform.openai.com/docs/guides/embeddings
//! [Gemini]: https://ai.google.dev/gemini-api/docs/embeddings
//! [Ollama]: https://ollama.com/

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

use std::sync::Once;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default request timeout: generous, because a cold Ollama model load can take
/// several seconds before the first byte.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";
const VOYAGE_DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const GEMINI_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// The environment variable [`Provider::Voyage`] reads when no key is passed.
pub const VOYAGE_API_KEY_ENV: &str = "VOYAGE_API_KEY";
/// The environment variable [`Provider::OpenAi`] reads when no key is passed.
pub const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
/// The environment variable [`Provider::Gemini`] reads when no key is passed.
pub const GEMINI_API_KEY_ENV: &str = "GEMINI_API_KEY";

/// Which embeddings backend a [`Client`] talks to.
///
/// `#[non_exhaustive]`: more providers can be added in a minor release, so match
/// with a `_ =>` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Provider {
    /// A local Ollama server (`{base_url}/api/embed`, default
    /// `http://localhost:11434`). No API key, works offline; ignores
    /// [`EmbedKind`].
    Ollama,
    /// Voyage AI (`{base_url}/embeddings`, default `https://api.voyageai.com/v1`).
    /// Needs [`VOYAGE_API_KEY_ENV`]; honours `input_type` ([`EmbedKind`]) and
    /// `output_dimension`.
    Voyage,
    /// OpenAI, or any OpenAI-compatible `/v1/embeddings` endpoint (together.ai,
    /// vLLM, LocalAI, ...) via a `base_url` override. Default
    /// `https://api.openai.com/v1`. Needs [`OPENAI_API_KEY_ENV`]; honours
    /// `dimensions` ([`ClientBuilder::output_dimension`]). Symmetric - ignores
    /// [`EmbedKind`].
    OpenAi,
    /// Google Gemini (Generative Language API,
    /// `{base_url}/models/{model}:batchEmbedContents`, default
    /// `https://generativelanguage.googleapis.com/v1beta`). Needs
    /// [`GEMINI_API_KEY_ENV`]; maps [`EmbedKind`] to `taskType` and honours
    /// `outputDimensionality`.
    Gemini,
}

impl Provider {
    fn label(self) -> &'static str {
        match self {
            Provider::Ollama => "ollama",
            Provider::Voyage => "voyage",
            Provider::OpenAi => "openai",
            Provider::Gemini => "gemini",
        }
    }

    fn default_base_url(self) -> &'static str {
        match self {
            Provider::Ollama => OLLAMA_DEFAULT_BASE_URL,
            Provider::Voyage => VOYAGE_DEFAULT_BASE_URL,
            Provider::OpenAi => OPENAI_DEFAULT_BASE_URL,
            Provider::Gemini => GEMINI_DEFAULT_BASE_URL,
        }
    }

    /// The environment variable a missing key falls back to, or `None` for a
    /// keyless provider (Ollama).
    fn api_key_env(self) -> Option<&'static str> {
        match self {
            Provider::Ollama => None,
            Provider::Voyage => Some(VOYAGE_API_KEY_ENV),
            Provider::OpenAi => Some(OPENAI_API_KEY_ENV),
            Provider::Gemini => Some(GEMINI_API_KEY_ENV),
        }
    }
}

/// Whether a batch is stored **documents** or a search **query**. Voyage uses
/// this for asymmetric retrieval (query- and document-side vectors differ, which
/// retrieves better); Ollama ignores it.
///
/// `#[non_exhaustive]` because the set of input types is a provider-defined
/// vocabulary that may grow; match with a `_ =>` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmbedKind {
    /// A stored document (Voyage `input_type = "document"`).
    Document,
    /// A search query (Voyage `input_type = "query"`).
    Query,
}

impl EmbedKind {
    /// Voyage / OpenAI-style `input_type`.
    fn as_str(self) -> &'static str {
        match self {
            EmbedKind::Document => "document",
            EmbedKind::Query => "query",
        }
    }

    /// Gemini `taskType`.
    fn gemini_task_type(self) -> &'static str {
        match self {
            EmbedKind::Document => "RETRIEVAL_DOCUMENT",
            EmbedKind::Query => "RETRIEVAL_QUERY",
        }
    }
}

/// A transport/decoding error, kept opaque on purpose. The concrete HTTP
/// backend ([`reqwest`]) is an implementation detail, so it is boxed rather than
/// exposed in the public API - a reqwest major bump must not force a breaking
/// release of this crate. The message and [`std::error::Error::source`] chain
/// are preserved.
pub type TransportError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Everything that can go wrong producing embeddings. Each variant carries the
/// provider that raised it so a caller can log or match without string-scraping.
///
/// The struct variants are `#[non_exhaustive]` so fields can be added (e.g. a
/// `retry_after` on [`Error::Api`]) without a breaking release; match them with
/// a trailing `..`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Building the underlying HTTP client failed (bad TLS config, etc.).
    #[error("failed to build HTTP client: {0}")]
    ClientBuild(#[source] TransportError),

    /// Voyage was selected but no API key was supplied and the environment
    /// variable is unset.
    #[error("{provider}: no API key (pass .api_key(..) or set {env})")]
    #[non_exhaustive]
    MissingApiKey {
        /// The provider that needed a key (`"voyage"`).
        provider: &'static str,
        /// The environment variable that was consulted ([`VOYAGE_API_KEY_ENV`]).
        env: &'static str,
    },

    /// The HTTP request never completed (connection refused, timeout, DNS, ...).
    /// For Ollama this usually means the server is not running.
    #[error("{provider} request failed: {source}")]
    #[non_exhaustive]
    Request {
        /// The provider the request targeted.
        provider: &'static str,
        /// The underlying transport error.
        #[source]
        source: TransportError,
    },

    /// The provider answered with a non-success status; `body` is its message.
    #[error("{provider} returned HTTP {status}: {body}")]
    #[non_exhaustive]
    Api {
        /// The provider that returned the error.
        provider: &'static str,
        /// The HTTP status code.
        status: u16,
        /// The response body (the provider's error message).
        body: String,
    },

    /// The success response could not be decoded into the expected shape.
    #[error("{provider} failed to decode response: {source}")]
    #[non_exhaustive]
    Decode {
        /// The provider whose response failed to decode.
        provider: &'static str,
        /// The underlying transport error.
        #[source]
        source: TransportError,
    },

    /// The provider returned a different number of vectors than inputs given.
    #[error("{provider} returned {got} embeddings for {expected} inputs")]
    #[non_exhaustive]
    CountMismatch {
        /// The provider that returned the wrong count.
        provider: &'static str,
        /// How many vectors came back.
        got: usize,
        /// How many were expected (one per input).
        expected: usize,
    },

    /// An `output_dimension` was pinned but a returned vector had a different
    /// width - a schema-desync guard for callers that store into a fixed-width
    /// column (e.g. pgvector `vector(1024)`).
    #[error("{provider} returned dimension {got} (expected {expected})")]
    #[non_exhaustive]
    DimMismatch {
        /// The provider that returned the wrong width.
        provider: &'static str,
        /// The width actually returned.
        got: usize,
        /// The pinned [`ClientBuilder::output_dimension`].
        expected: usize,
    },
}

/// Install the ring crypto provider exactly once, process-wide. Ignoring the
/// `Err` is intentional: it only means a provider is already installed.
fn install_ring() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// Build a [`Client`]. Start with [`Client::builder`]. `Debug` redacts the API
/// key.
#[derive(Clone)]
pub struct ClientBuilder {
    provider: Provider,
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    output_dimension: Option<usize>,
    timeout: Duration,
    max_batch: Option<usize>,
}

/// Render `Option<String>` API keys as presence-only, never the secret itself.
fn redacted(key: &Option<String>) -> Option<&'static str> {
    key.as_ref().map(|_| "<redacted>")
}

impl std::fmt::Debug for ClientBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientBuilder")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &redacted(&self.api_key))
            .field("output_dimension", &self.output_dimension)
            .field("timeout", &self.timeout)
            .field("max_batch", &self.max_batch)
            .finish()
    }
}

impl ClientBuilder {
    /// Override the provider base URL. Blank/whitespace falls back to the
    /// provider default; a trailing `/` is trimmed.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API key explicitly instead of reading it from the environment.
    /// Consulted by every keyed provider (Voyage, OpenAI, Gemini); ignored by
    /// keyless Ollama.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Request a specific embedding width: Voyage `output_dimension`, OpenAI
    /// `dimensions`, Gemini `outputDimensionality` (Ollama is model-fixed and
    /// ignores it). Whatever the provider, every returned vector is validated
    /// against it, so a drift becomes an [`Error::DimMismatch`] rather than a
    /// silent schema desync.
    pub fn output_dimension(mut self, dim: usize) -> Self {
        self.output_dimension = Some(dim);
        self
    }

    /// Override the request timeout (default 120s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Cap inputs per HTTP request; larger batches are split into sequential
    /// requests and their results concatenated in order. Unset sends every input
    /// in a single request. `0` is treated as `1`.
    pub fn max_batch(mut self, max_batch: usize) -> Self {
        self.max_batch = Some(max_batch.max(1));
        self
    }

    /// Finish building. Installs the ring TLS provider, constructs the HTTP
    /// client, and - for a keyed provider - resolves the API key (erroring with
    /// [`Error::MissingApiKey`] if it is neither passed nor in the provider's
    /// environment variable).
    pub fn build(self) -> Result<Client, Error> {
        install_ring();

        let api_key = match self.provider.api_key_env() {
            // Keyless (Ollama).
            None => None,
            Some(env) => {
                let key = self
                    .api_key
                    .or_else(|| std::env::var(env).ok())
                    .filter(|k| !k.trim().is_empty());
                match key {
                    Some(k) => Some(k),
                    None => {
                        return Err(Error::MissingApiKey {
                            provider: self.provider.label(),
                            env,
                        });
                    }
                }
            }
        };

        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| Error::ClientBuild(Box::new(e)))?;

        let base_url = self
            .base_url
            .map(|b| b.trim().trim_end_matches('/').to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| self.provider.default_base_url().to_string());

        Ok(Client {
            http,
            provider: self.provider,
            model: self.model,
            base_url,
            api_key,
            output_dimension: self.output_dimension,
            max_batch: self.max_batch,
        })
    }
}

/// A configured embeddings client. Cheap to clone (`reqwest::Client` is an
/// `Arc` internally); build it once and reuse it. `Debug` redacts the API key.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    provider: Provider,
    model: String,
    base_url: String,
    api_key: Option<String>,
    output_dimension: Option<usize>,
    max_batch: Option<usize>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &redacted(&self.api_key))
            .field("output_dimension", &self.output_dimension)
            .field("max_batch", &self.max_batch)
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Start configuring a client for `provider` using `model`.
    pub fn builder(provider: Provider, model: impl Into<String>) -> ClientBuilder {
        ClientBuilder {
            provider,
            model: model.into(),
            base_url: None,
            api_key: None,
            output_dimension: None,
            timeout: DEFAULT_TIMEOUT,
            max_batch: None,
        }
    }

    /// The provider this client talks to.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Embed a batch of texts, preserving input order. `kind` distinguishes
    /// stored documents from search queries (see [`EmbedKind`]; symmetric
    /// providers ignore it). Batches larger than `max_batch` are split into
    /// sequential requests and their results concatenated.
    ///
    /// **No retry or backoff.** A transient failure (network blip, a `429`
    /// rate-limit) returns `Err` immediately; if it happens partway through a
    /// split batch, the vectors already fetched in this call are discarded.
    /// Resilience is deliberately the caller's job - inspect [`Error::Api`]'s
    /// `status` and re-invoke `embed` if you need it.
    pub async fn embed(&self, texts: &[String], kind: EmbedKind) -> Result<Vec<Vec<f32>>, Error> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let batch = self.max_batch.unwrap_or(texts.len()).max(1);
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch) {
            let vectors = match self.provider {
                Provider::Ollama => self.embed_ollama(chunk).await?,
                Provider::Voyage => self.embed_voyage(chunk, kind).await?,
                Provider::OpenAi => self.embed_openai(chunk).await?,
                Provider::Gemini => self.embed_gemini(chunk, kind).await?,
            };
            out.extend(vectors);
        }
        self.validate(out, texts.len())
    }

    /// The resolved API key for a keyed provider. `build()` guarantees it is
    /// present for every provider except keyless Ollama, which never calls this.
    fn require_key(&self) -> &str {
        self.api_key
            .as_deref()
            .expect("invariant: build() resolves an api_key for keyed providers")
    }

    async fn embed_ollama(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let url = format!("{}/api/embed", self.base_url);
        let body = OllamaRequest {
            model: &self.model,
            input: texts,
        };
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(request_err(Provider::Ollama))?;
        let resp = error_for_status(Provider::Ollama, resp).await?;
        let parsed: OllamaResponse = resp.json().await.map_err(decode_err(Provider::Ollama))?;
        Ok(parsed.embeddings)
    }

    async fn embed_voyage(
        &self,
        texts: &[String],
        kind: EmbedKind,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let url = format!("{}/embeddings", self.base_url);
        let body = VoyageRequest {
            input: texts,
            model: &self.model,
            input_type: kind.as_str(),
            output_dimension: self.output_dimension,
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.require_key())
            .json(&body)
            .send()
            .await
            .map_err(request_err(Provider::Voyage))?;
        let resp = error_for_status(Provider::Voyage, resp).await?;
        let parsed: DataResponse = resp.json().await.map_err(decode_err(Provider::Voyage))?;
        Ok(sorted_by_index(parsed.data))
    }

    async fn embed_openai(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let url = format!("{}/embeddings", self.base_url);
        let body = OpenAiRequest {
            input: texts,
            model: &self.model,
            encoding_format: "float",
            dimensions: self.output_dimension,
        };
        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.require_key())
            .json(&body)
            .send()
            .await
            .map_err(request_err(Provider::OpenAi))?;
        let resp = error_for_status(Provider::OpenAi, resp).await?;
        let parsed: DataResponse = resp.json().await.map_err(decode_err(Provider::OpenAi))?;
        Ok(sorted_by_index(parsed.data))
    }

    async fn embed_gemini(
        &self,
        texts: &[String],
        kind: EmbedKind,
    ) -> Result<Vec<Vec<f32>>, Error> {
        // Gemini names the model in the URL path and each sub-request, always
        // `models/`-prefixed.
        let model = if self.model.starts_with("models/") {
            self.model.clone()
        } else {
            format!("models/{}", self.model)
        };
        let url = format!("{}/{}:batchEmbedContents", self.base_url, model);
        let requests = texts
            .iter()
            .map(|t| GeminiEmbedRequest {
                model: &model,
                content: GeminiContent {
                    parts: [GeminiPart { text: t }],
                },
                task_type: kind.gemini_task_type(),
                output_dimensionality: self.output_dimension,
            })
            .collect();
        let body = GeminiRequest { requests };
        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", self.require_key())
            .json(&body)
            .send()
            .await
            .map_err(request_err(Provider::Gemini))?;
        let resp = error_for_status(Provider::Gemini, resp).await?;
        let parsed: GeminiResponse = resp.json().await.map_err(decode_err(Provider::Gemini))?;
        // Gemini returns embeddings in request order (no index field).
        Ok(parsed.embeddings.into_iter().map(|e| e.values).collect())
    }

    /// Enforce one vector per input and, if a dimension was pinned, that every
    /// vector matches it.
    fn validate(&self, vectors: Vec<Vec<f32>>, expected: usize) -> Result<Vec<Vec<f32>>, Error> {
        let provider = self.provider.label();
        if vectors.len() != expected {
            return Err(Error::CountMismatch {
                provider,
                got: vectors.len(),
                expected,
            });
        }
        if let Some(dim) = self.output_dimension {
            for v in &vectors {
                if v.len() != dim {
                    return Err(Error::DimMismatch {
                        provider,
                        got: v.len(),
                        expected: dim,
                    });
                }
            }
        }
        Ok(vectors)
    }
}

/// Turn a non-success HTTP response into an [`Error::Api`] carrying the body.
async fn error_for_status(
    provider: Provider,
    resp: reqwest::Response,
) -> Result<reqwest::Response, Error> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    Err(Error::Api {
        provider: provider.label(),
        status,
        body,
    })
}

/// Map a transport error to [`Error::Request`], boxing it opaque.
fn request_err(provider: Provider) -> impl FnOnce(reqwest::Error) -> Error {
    move |e| Error::Request {
        provider: provider.label(),
        source: Box::new(e),
    }
}

/// Map a decode error to [`Error::Decode`], boxing it opaque.
fn decode_err(provider: Provider) -> impl FnOnce(reqwest::Error) -> Error {
    move |e| Error::Decode {
        provider: provider.label(),
        source: Box::new(e),
    }
}

/// Reorder an OpenAI-style `data` array by its `index` and drop to bare vectors.
/// The APIs return data in input order, but sorting is cheap insurance.
fn sorted_by_index(mut data: Vec<EmbeddingDatum>) -> Vec<Vec<f32>> {
    data.sort_by_key(|d| d.index);
    data.into_iter().map(|d| d.embedding).collect()
}

// ---- Wire types ----------------------------------------------------------

// Ollama: `{base}/api/embed`.
#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

// Voyage: `{base}/embeddings`.
#[derive(Serialize)]
struct VoyageRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
}

// OpenAI (and OpenAI-compatible): `{base}/embeddings`. Symmetric, so no
// input_type; `dimensions` is OpenAI's name for the requested width.
#[derive(Serialize)]
struct OpenAiRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

/// The response shape shared by Voyage and OpenAI: `{"data":[{embedding,index}]}`.
#[derive(Deserialize)]
struct DataResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

// Gemini: `{base}/models/{model}:batchEmbedContents`.
#[derive(Serialize)]
struct GeminiRequest<'a> {
    requests: Vec<GeminiEmbedRequest<'a>>,
}

#[derive(Serialize)]
struct GeminiEmbedRequest<'a> {
    model: &'a str,
    content: GeminiContent<'a>,
    #[serde(rename = "taskType")]
    task_type: &'a str,
    #[serde(
        rename = "outputDimensionality",
        skip_serializing_if = "Option::is_none"
    )]
    output_dimensionality: Option<usize>,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    parts: [GeminiPart<'a>; 1],
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct GeminiResponse {
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests;
