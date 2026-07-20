# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release: a provider-agnostic embeddings client with **Voyage** and
  **Ollama** providers, on rustls + ring (no OpenSSL, no aws-lc).
- `Client` / `ClientBuilder` with `base_url`, `api_key`, `output_dimension`,
  `timeout`, and `max_batch` (transparent request splitting).
- `EmbedKind` (Voyage asymmetric `input_type`) and optional `output_dimension`
  pinning validated against every returned vector.
- Typed `Error` carrying the offending provider.

[Unreleased]: https://github.com/jhheider/lean-embed/commits/main
