//! End-to-end tests against a live local Ollama (skipped by default, like a
//! network test). Run with:
//!
//! ```sh
//! ollama pull nomic-embed-text
//! cargo test --test ollama_live -- --ignored --nocapture
//! ```
//!
//! These prove the real round-trip: correct dimensions, order-preserving batch
//! splitting, dim-pin validation, and that the vectors are semantically sane
//! (a query lands nearer the relevant document than an unrelated one).

use lean_embed::{Client, EmbedKind, Provider};

const MODEL: &str = "nomic-embed-text";

fn client() -> Client {
    Client::builder(Provider::Ollama, MODEL)
        .build()
        .expect("build ollama client")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::test]
#[ignore = "needs a running Ollama with nomic-embed-text"]
async fn documents_embed_with_consistent_dimensions() {
    let inputs: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let vectors = client()
        .embed(&inputs, EmbedKind::Document)
        .await
        .expect("embed documents");

    assert_eq!(vectors.len(), inputs.len(), "one vector per input");
    let dim = vectors[0].len();
    assert!(dim > 0, "non-empty embedding");
    for v in &vectors {
        assert_eq!(v.len(), dim, "all vectors share a width");
        assert!(v.iter().all(|x| x.is_finite()), "no NaN/inf components");
        assert!(v.iter().any(|&x| x != 0.0), "not an all-zero vector");
    }
    eprintln!("nomic-embed-text dimension: {dim}");
}

#[tokio::test]
#[ignore = "needs a running Ollama with nomic-embed-text"]
async fn query_and_document_share_a_dimension() {
    let c = client();
    let doc = c
        .embed(&["a stored document".into()], EmbedKind::Document)
        .await
        .expect("embed doc");
    let q = c
        .embed(&["a search query".into()], EmbedKind::Query)
        .await
        .expect("embed query");
    assert_eq!(doc[0].len(), q[0].len());
}

#[tokio::test]
#[ignore = "needs a running Ollama with nomic-embed-text"]
async fn max_batch_splitting_preserves_order_live() {
    // With max_batch(1) each input is its own request; the concatenation must
    // still line up with the single-request result element-for-element.
    let inputs: Vec<String> = ["one", "two", "three", "four", "five"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let single = client()
        .embed(&inputs, EmbedKind::Document)
        .await
        .expect("single request");

    let split_client = Client::builder(Provider::Ollama, MODEL)
        .max_batch(1)
        .build()
        .unwrap();
    let split = split_client
        .embed(&inputs, EmbedKind::Document)
        .await
        .expect("split requests");

    assert_eq!(single.len(), split.len());
    for (i, (a, b)) in single.iter().zip(&split).enumerate() {
        // Same model, same text -> identical vectors regardless of batching.
        assert_eq!(a, b, "vector {i} differs between batched and split");
    }
}

#[tokio::test]
#[ignore = "needs a running Ollama with nomic-embed-text"]
async fn pinned_dimension_validates_against_the_live_model() {
    // Discover the real width, pin it, and confirm the client accepts it...
    let dim = client()
        .embed(&["probe".into()], EmbedKind::Document)
        .await
        .expect("probe")[0]
        .len();

    let ok = Client::builder(Provider::Ollama, MODEL)
        .output_dimension(dim)
        .build()
        .unwrap();
    assert!(
        ok.embed(&["fits".into()], EmbedKind::Document)
            .await
            .is_ok(),
        "correct pin should pass"
    );

    // ...then a wrong pin must be caught as a DimMismatch, not silently stored.
    let bad = Client::builder(Provider::Ollama, MODEL)
        .output_dimension(dim + 1)
        .build()
        .unwrap();
    let err = bad
        .embed(&["mismatch".into()], EmbedKind::Document)
        .await
        .expect_err("wrong pin should fail");
    assert!(
        matches!(err, lean_embed::Error::DimMismatch { .. }),
        "expected DimMismatch, got {err:?}"
    );
}

#[tokio::test]
#[ignore = "needs a running Ollama with nomic-embed-text"]
async fn embeddings_are_semantically_sane() {
    let c = client();
    let docs: Vec<String> = [
        "A kiln fires pottery and ceramics at high temperature.",
        "Interest rates influence bond prices in financial markets.",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let doc_vecs = c.embed(&docs, EmbedKind::Document).await.expect("docs");

    let q = c
        .embed(
            &["How hot does a pottery oven get?".into()],
            EmbedKind::Query,
        )
        .await
        .expect("query");

    let sim_relevant = cosine(&q[0], &doc_vecs[0]);
    let sim_unrelated = cosine(&q[0], &doc_vecs[1]);
    eprintln!("relevant={sim_relevant:.4} unrelated={sim_unrelated:.4}");
    assert!(
        sim_relevant > sim_unrelated,
        "the pottery query should sit nearer the kiln doc ({sim_relevant:.4}) \
         than the finance doc ({sim_unrelated:.4})"
    );
}
