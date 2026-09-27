# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-27 (yanked)

> **Yanked** (replaced by 0.1.2): prefixes were on by default, which
> would corrupt mixed-version indexes. Yank removes it from resolution;
> anyone who pinned exactly 0.1.1 should move to 0.1.2.

### Changed

- **Ollama provider honours `EmbedKind` for `nomic-embed-text*` models**: the
  request now carries nomic's trained task prefixes (`search_query: ` on
  queries, `search_document: ` on stored documents). Ollama's nomic modelfile
  is a bare `{{ .Prompt }}` that accepts `input_type` without applying it
  (verified against Ollama 0.32.15), so without client-side prefixes every
  embedding was effectively an unprompted document - measurable retrieval
  loss for asymmetric queries. Other models are untouched; extend the keyed
  list in `providers/ollama.rs` when another model needs a scheme.

## [0.1.2] - 2026-09-27

### Added

- `ClientBuilder::task_prefixes(bool)` - opt-in application of model task
  prefixes where the provider has no native asymmetric-retrieval knob
  (today: nomic-style `search_query: ` / `search_document: ` for Ollama
  `nomic-embed-text*`). **Default off**: 0.1.1 shipped it on and was
  **yanked** the same day, because a `version = "0.1"` consumer with an
  existing vector index (gloaming, hearth-app) would have silently mixed
  prefixed query vectors with unprefixed stored ones. Enable it when
  building a fresh index and keep it on for that index's life; changing
  the scheme for an existing index is a reindex.

## [Unreleased]

### Security

- Lockfile: bump h2 to 0.4.19 (RUSTSEC-2026-0258) and rustls to 0.23.45
  (RUSTSEC-2026-0285). Downstream crates resolve their own lockfiles; run
  `cargo update -p h2 -p rustls` to pick up the fixes.

## [0.1.0] - 2026-07-20

### Added

- Initial release: a provider-agnostic embeddings client on rustls + ring (no
  OpenSSL, no aws-lc), with four providers: **Voyage**, **OpenAI** (and any
  OpenAI-compatible endpoint), **Gemini**, and **Ollama** (local, offline).
- `Client` / `ClientBuilder` with `base_url`, `api_key`, `output_dimension`,
  `timeout`, and `max_batch` (transparent request splitting).
- `EmbedKind` mapping to each provider's asymmetric retrieval knob
  (`input_type` / `taskType`), and an optional `output_dimension` requested and
  validated against every returned vector.
- Typed, `#[non_exhaustive]` `Error` carrying the offending provider; the
  transport error is kept opaque so a reqwest bump is not a breaking change.
- `Debug` on `Client`/`ClientBuilder` redacts the API key.

[0.1.0]: https://github.com/jhheider/lean-embed/releases/tag/v0.1.0
[Unreleased]: https://github.com/jhheider/lean-embed/compare/v0.1.0...main
