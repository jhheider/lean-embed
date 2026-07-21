# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-20

### Added

- Initial release: a provider-agnostic embeddings client on rustls + ring (no
  OpenSSL, no aws-lc), with four providers - **Voyage**, **OpenAI** (and any
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
