# lean-embed

A **lean, provider-agnostic text-embeddings client** for Rust. One small client
turns batches of text into vectors against either **[Voyage AI]** (hosted) or
**[Ollama]** (local, offline, no API key) - and nothing more. No vector store,
no chunking, no agent loop: just the embeddings HTTP call, so your retrieval
stack stays yours.

The wire is [`reqwest`] on **rustls + [ring]** only - **never OpenSSL, never
aws-lc**. That is the whole reason this crate exists instead of reaching for a
full agent/RAG framework: the dependency tree stays small and cross-compiles
cleanly to musl and aarch64. `cargo tree -i aws-lc-sys` and `-i openssl-sys` are
empty, and staying that way is a standing commitment.

## Why not `rig` / `async-openai` / a RAG framework?

- **`rig`** is the only crate with first-class Voyage + Ollama, but its TLS is
  hardwired to reqwest's rustls default (aws-lc-rs) or native-tls (OpenSSL) with
  no ring path - and it is a full agent/RAG framework for one `embed()` call.
- **`async-openai`** ships clean deps (even `rustls-no-provider`) but has no
  Voyage, and Voyage is not drop-in OpenAI-compatible, so a `base_url` override
  loses its `output_dimension` and `input_type` knobs.
- A **RAG framework's** value is its store + pipeline; but each real consumer
  already owns a *different* store (pgvector-in-Postgres here, an offline
  brute-force index there), so the framework part is exactly what can't be
  shared. This crate lifts only the portable part: the client.

## Install

```toml
[dependencies]
lean-embed = "0.1"
```

## Usage

```rust
use lean_embed::{Client, EmbedKind, Provider};

# async fn run() -> Result<(), lean_embed::Error> {
// Local Ollama - no key, offline.
let ollama = Client::builder(Provider::Ollama, "nomic-embed-text").build()?;
let vectors = ollama
    .embed(&["hello".into(), "world".into()], EmbedKind::Document)
    .await?;

// Hosted Voyage, pinned to 1024 dimensions (key from VOYAGE_API_KEY,
// or pass .api_key(..)). max_batch splits large batches transparently.
let voyage = Client::builder(Provider::Voyage, "voyage-3.5-lite")
    .output_dimension(1024)
    .max_batch(96)
    .build()?;
let query = voyage.embed(&["how do I center clay?".into()], EmbedKind::Query).await?;
# let _ = (vectors, query);
# Ok(())
# }
```

### Notes

- **`EmbedKind`** picks Voyage's asymmetric `input_type` (`Document` vs
  `Query`), which retrieves better. Ollama ignores it.
- **`output_dimension`** is sent to Voyage and, for *either* provider, validated
  against every returned vector - so a model drifting off your stored width
  becomes an `Error::DimMismatch` instead of a silent schema desync. Ollama
  models are fixed-width and ignore the request but are still validated.
- **`max_batch`** caps inputs per HTTP request; larger batches are split and
  their results concatenated in input order.
- A `Client` owns its `reqwest::Client` (cheap to clone, `Arc` inside) - build
  it once and reuse it; do not rebuild per call.

## Providers

| Provider | Endpoint | Key | Offline | Honours dim / kind |
|----------|----------|-----|---------|--------------------|
| Ollama   | `{base_url}/api/embed` (default `http://localhost:11434`) | none | yes | dim validated only |
| Voyage   | `{base_url}/embeddings` (default `https://api.voyageai.com/v1`) | `VOYAGE_API_KEY` | no | yes |

An OpenAI-compatible (`/v1/embeddings`) provider is a natural future addition.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.

[Voyage AI]: https://www.voyageai.com/
[Ollama]: https://ollama.com/
[ring]: https://github.com/briansmith/ring
[`reqwest`]: https://github.com/seanmonstar/reqwest
