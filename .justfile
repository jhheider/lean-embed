import '../justfile'

[private]
a:
    @just -l

# Run the test suite.
[group('checks')]
test:
    cargo test

# fmt + clippy the way CI does (warnings are errors).
[group('checks')]
lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings

# The full local gate: lint + test + a real build.
[group('checks')]
check: lint test
    cargo build

# Confirm the lean-deps commitment: no aws-lc/openssl in the tree.
[group('checks')]
lean:
    @! cargo tree -i aws-lc-sys 2>/dev/null | grep -q . || (echo "aws-lc-sys present!" && exit 1)
    @! cargo tree -i openssl-sys 2>/dev/null | grep -q . || (echo "openssl-sys present!" && exit 1)
    @echo "lean: no aws-lc-sys / openssl-sys"

# Generate coverage/lcov.info (needs cargo-llvm-cov).
[group('coverage')]
coverage:
    mkdir -p coverage
    cargo llvm-cov --lcov --output-path coverage/lcov.info

[group('coverage')]
coverage-summary:
    cargo llvm-cov --summary-only
