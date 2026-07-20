//! A lean, provider-agnostic text-embeddings client.
//!
//! One [`Client`] turns batches of text into vectors against either
//! **[Voyage AI]** (hosted, with the asymmetric `input_type` and an optional
//! `output_dimension`) or **[Ollama]** (local, no API key, fully offline). The
//! wire is [`reqwest`] on **rustls + ring** only - never OpenSSL or aws-lc - so
//! the dependency tree stays small and cross-compiles cleanly (musl, aarch64).
//! That lean stack is the reason this crate exists instead of a full agent/RAG
//! framework: it is *only* the embeddings HTTP client, so a vector store,
//! chunking, and retrieval stay in the caller where they belong.
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
//! [Ollama]: https://ollama.com/

use std::sync::Once;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Default request timeout: generous, because a cold Ollama model load can take
/// several seconds before the first byte.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

const VOYAGE_DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";
const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434";

/// The environment variable the Voyage provider reads when no key is passed to
/// the builder.
pub const VOYAGE_API_KEY_ENV: &str = "VOYAGE_API_KEY";

/// Which embeddings backend a [`Client`] talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Provider {
    /// A local Ollama server (`{base_url}/api/embed`, default
    /// `http://localhost:11434`). No API key, works offline.
    Ollama,
    /// Voyage AI (`{base_url}/embeddings`, default `https://api.voyageai.com/v1`).
    /// Needs an API key and honours `input_type` and `output_dimension`.
    Voyage,
}

impl Provider {
    fn label(self) -> &'static str {
        match self {
            Provider::Ollama => "ollama",
            Provider::Voyage => "voyage",
        }
    }

    fn default_base_url(self) -> &'static str {
        match self {
            Provider::Ollama => OLLAMA_DEFAULT_BASE_URL,
            Provider::Voyage => VOYAGE_DEFAULT_BASE_URL,
        }
    }
}

/// Whether a batch is stored **documents** or a search **query**. Voyage uses
/// this for asymmetric retrieval (query- and document-side vectors differ, which
/// retrieves better); Ollama ignores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedKind {
    Document,
    Query,
}

impl EmbedKind {
    fn as_str(self) -> &'static str {
        match self {
            EmbedKind::Document => "document",
            EmbedKind::Query => "query",
        }
    }
}

/// Everything that can go wrong producing embeddings. Each variant carries the
/// provider that raised it so a caller can log or match without string-scraping.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Building the underlying [`reqwest::Client`] failed (bad TLS config, etc.).
    #[error("failed to build HTTP client: {0}")]
    ClientBuild(#[source] reqwest::Error),

    /// Voyage was selected but no API key was supplied and the environment
    /// variable is unset.
    #[error("{provider}: no API key (pass .api_key(..) or set {env})")]
    MissingApiKey {
        provider: &'static str,
        env: &'static str,
    },

    /// The HTTP request never completed (connection refused, timeout, DNS, ...).
    /// For Ollama this usually means the server is not running.
    #[error("{provider} request failed: {source}")]
    Request {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },

    /// The provider answered with a non-success status; `body` is its message.
    #[error("{provider} returned HTTP {status}: {body}")]
    Api {
        provider: &'static str,
        status: u16,
        body: String,
    },

    /// The success response could not be decoded into the expected shape.
    #[error("{provider} failed to decode response: {source}")]
    Decode {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },

    /// The provider returned a different number of vectors than inputs given.
    #[error("{provider} returned {got} embeddings for {expected} inputs")]
    CountMismatch {
        provider: &'static str,
        got: usize,
        expected: usize,
    },

    /// An `output_dimension` was pinned but a returned vector had a different
    /// width - a schema-desync guard for callers that store into a fixed-width
    /// column (e.g. pgvector `vector(1024)`).
    #[error("{provider} returned dimension {got} (expected {expected})")]
    DimMismatch {
        provider: &'static str,
        got: usize,
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

/// Build a [`Client`]. Start with [`Client::builder`].
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    provider: Provider,
    model: String,
    base_url: Option<String>,
    api_key: Option<String>,
    output_dimension: Option<usize>,
    timeout: Duration,
    max_batch: Option<usize>,
}

impl ClientBuilder {
    /// Override the provider base URL. Blank/whitespace falls back to the
    /// provider default; a trailing `/` is trimmed.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the API key explicitly instead of reading it from the environment.
    /// Only consulted by [`Provider::Voyage`].
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Request a specific embedding width. Sent to Voyage as `output_dimension`
    /// (Ollama is model-fixed and ignores it), and - for either provider -
    /// validated against every returned vector so a drift becomes a
    /// [`Error::DimMismatch`] rather than a silent schema desync.
    pub fn output_dimension(mut self, dim: usize) -> Self {
        self.output_dimension = Some(dim);
        self
    }

    /// Override the request timeout (default 120s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Cap inputs per HTTP request; larger batches are split and their results
    /// concatenated in order. Unset sends every input in a single request.
    pub fn max_batch(mut self, max_batch: usize) -> Self {
        self.max_batch = Some(max_batch.max(1));
        self
    }

    /// Finish building. Installs the ring TLS provider, constructs the HTTP
    /// client, and - for Voyage - resolves the API key (erroring if it is
    /// neither passed nor in [`VOYAGE_API_KEY_ENV`]).
    pub fn build(self) -> Result<Client, Error> {
        install_ring();

        let api_key = match self.provider {
            Provider::Voyage => {
                let key = self
                    .api_key
                    .or_else(|| std::env::var(VOYAGE_API_KEY_ENV).ok())
                    .filter(|k| !k.trim().is_empty());
                match key {
                    Some(k) => Some(k),
                    None => {
                        return Err(Error::MissingApiKey {
                            provider: self.provider.label(),
                            env: VOYAGE_API_KEY_ENV,
                        });
                    }
                }
            }
            Provider::Ollama => None,
        };

        let http = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(Error::ClientBuild)?;

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
/// `Arc` internally); build it once and reuse it.
#[derive(Debug, Clone)]
pub struct Client {
    http: reqwest::Client,
    provider: Provider,
    model: String,
    base_url: String,
    api_key: Option<String>,
    output_dimension: Option<usize>,
    max_batch: Option<usize>,
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
    /// stored documents from search queries (see [`EmbedKind`]). Batches larger
    /// than `max_batch` are split transparently and their results concatenated.
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
            };
            out.extend(vectors);
        }
        self.validate(out, texts.len())
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
            .map_err(|source| Error::Request {
                provider: Provider::Ollama.label(),
                source,
            })?;
        let resp = error_for_status(Provider::Ollama, resp).await?;
        let parsed: OllamaResponse = resp.json().await.map_err(|source| Error::Decode {
            provider: Provider::Ollama.label(),
            source,
        })?;
        Ok(parsed.embeddings)
    }

    async fn embed_voyage(
        &self,
        texts: &[String],
        kind: EmbedKind,
    ) -> Result<Vec<Vec<f32>>, Error> {
        // Present because build() enforces it for Voyage.
        let key = self.api_key.as_deref().unwrap_or_default();
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
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|source| Error::Request {
                provider: Provider::Voyage.label(),
                source,
            })?;
        let resp = error_for_status(Provider::Voyage, resp).await?;
        let parsed: VoyageResponse = resp.json().await.map_err(|source| Error::Decode {
            provider: Provider::Voyage.label(),
            source,
        })?;
        // Voyage returns data in input order, but sort by index to be safe.
        let mut data = parsed.data;
        data.sort_by_key(|d| d.index);
        Ok(data.into_iter().map(|d| d.embedding).collect())
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

// ---- Wire types ----------------------------------------------------------

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct VoyageRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageDatum>,
}

#[derive(Deserialize)]
struct VoyageDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[cfg(test)]
mod tests;
