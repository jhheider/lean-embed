# lean-embed

A **lean, provider-agnostic text-embeddings client** for Rust. One small client
turns batches of text into vectors against **[OpenAI]** (or any OpenAI-compatible
endpoint), **[Gemini]**, **[Voyage AI]**, or **[Ollama]** (local, offline, no
API key), and nothing more. No vector store, no chunking, no agent loop: just
the embeddings HTTP call, so your retrieval stack stays yours.

The wire is [`reqwest`] on **rustls + [ring]** only: **never OpenSSL, never
aws-lc**. That is the whole reason this crate exists instead of reaching for a
full agent/RAG framework: the dependency tree stays small and cross-compiles
cleanly to musl and aarch64. `cargo tree -i aws-lc-sys` and `-i openssl-sys` are
empty, and staying that way is a standing commitment.

## Why not `rig` / `async-openai` / a RAG framework?

- **`rig`** is the only crate with first-class Voyage + Ollama, but its TLS is
  hardwired to reqwest's rustls default (aws-lc-rs) or native-tls (OpenSSL) with
  no ring path, and it is a full agent/RAG framework for one `embed()` call.
- **`async-openai`** ships clean deps (even `rustls-no-provider`) but is
  OpenAI-only: no Voyage, Gemini, or Ollama, each of which has its own wire
  shape and asymmetry knob.
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

async fn embed_examples() -> Result<(), lean_embed::Error> {
    // Local Ollama, no key, offline.
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
    let query = voyage
        .embed(&["how do I center clay?".into()], EmbedKind::Query)
        .await?;

    // OpenAI (or any OpenAI-compatible endpoint via .base_url(..)); Gemini too.
    let openai = Client::builder(Provider::OpenAi, "text-embedding-3-small").build()?;
    let gemini = Client::builder(Provider::Gemini, "text-embedding-004").build()?;
    let _ = (openai, gemini);

    Ok(())
}
```

### Notes

- **`EmbedKind`** selects each provider's asymmetric retrieval knob - Voyage's
  `input_type` and Gemini's `taskType` (`Document` vs `Query`), which retrieves
  better. OpenAI is symmetric and Ollama is local; both ignore it.
- **`output_dimension`** is requested where the provider supports it (Voyage
  `output_dimension`, OpenAI `dimensions`, Gemini `outputDimensionality`) and,
  for *every* provider, validated against every returned vector, so a model
  drifting off your stored width becomes an `Error::DimMismatch` instead of a
  silent schema desync.
- **`max_batch`** caps inputs per HTTP request; larger batches are split into
  sequential requests and concatenated in input order.
- **No retry/backoff** - a transient or `429` failure returns `Err` (and a
  mid-batch failure discards that call's already-fetched vectors). Wrap `embed`
  yourself if you need resilience; `Error::Api { status, .. }` exposes the code.
- A `Client` owns its HTTP client (cheap to clone, `Arc` inside) - build it once
  and reuse it. `Debug` redacts the API key.

## Providers

| Provider | Endpoint | Key env | Offline | `EmbedKind` |
|----------|----------|---------|---------|-------------|
| Ollama   | `{base}/api/embed` (default `http://localhost:11434`) | none | yes | ignored |
| Voyage   | `{base}/embeddings` (default `https://api.voyageai.com/v1`) | `VOYAGE_API_KEY` | no | `input_type` |
| OpenAI   | `{base}/embeddings` (default `https://api.openai.com/v1`) | `OPENAI_API_KEY` | no | symmetric (ignored) |
| Gemini   | `{base}/models/{model}:batchEmbedContents` (default `…/v1beta`) | `GEMINI_API_KEY` | no | `taskType` |

OpenAI's entry doubles as the client for any OpenAI-compatible `/v1/embeddings`
server (together.ai, vLLM, LocalAI, …) via a `base_url` override.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.

[Voyage AI]: https://www.voyageai.com/
[OpenAI]: https://platform.openai.com/docs/guides/embeddings
[Gemini]: https://ai.google.dev/gemini-api/docs/embeddings
[Ollama]: https://ollama.com/
[ring]: https://github.com/briansmith/ring
[`reqwest`]: https://github.com/seanmonstar/reqwest
