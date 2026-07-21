//! End-to-end tests against the live Voyage AI API (skipped by default). Run
//! with a real key:
//!
//! ```sh
//! VOYAGE_API_KEY=... cargo test --test voyage_live -- --ignored --nocapture
//! ```
//!
//! These exercise the knobs Ollama can't: an arbitrary `output_dimension`, and
//! the asymmetric `input_type` (query- vs document-side vectors actually
//! differ). Model/base URL come from the environment so it matches a consumer's
//! config (hearth uses `voyage-3.5-lite` at 1024 dims).

use lean_embed::{Client, EmbedKind, Provider};

fn model() -> String {
    std::env::var("VOYAGE_MODEL").unwrap_or_else(|_| "voyage-3.5-lite".into())
}

fn client(output_dimension: Option<usize>) -> Client {
    let mut b = Client::builder(Provider::Voyage, model());
    if let Ok(base) = std::env::var("VOYAGE_BASE_URL") {
        b = b.base_url(base);
    }
    if let Some(d) = output_dimension {
        b = b.output_dimension(d);
    }
    b.build()
        .expect("build voyage client (is VOYAGE_API_KEY set?)")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::test]
#[ignore = "needs VOYAGE_API_KEY and network"]
async fn documents_embed_at_the_pinned_dimension() {
    // hearth pins 1024 to match its pgvector column; prove the request is
    // honoured and validated end-to-end.
    let c = client(Some(1024));
    let inputs: Vec<String> = ["alpha", "beta", "gamma"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let vectors = c
        .embed(&inputs, EmbedKind::Document)
        .await
        .expect("embed docs");
    assert_eq!(vectors.len(), inputs.len());
    for v in &vectors {
        assert_eq!(v.len(), 1024, "output_dimension must be honoured");
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[tokio::test]
#[ignore = "needs VOYAGE_API_KEY and network"]
async fn output_dimension_is_arbitrary() {
    // A different pin -> a different width (Voyage supports several).
    let c = client(Some(512));
    let v = c
        .embed(&["probe".into()], EmbedKind::Document)
        .await
        .expect("embed");
    assert_eq!(v[0].len(), 512);
}

#[tokio::test]
#[ignore = "needs VOYAGE_API_KEY and network"]
async fn query_and_document_vectors_are_asymmetric() {
    // The same text embedded as a query vs a document should differ, which is
    // the whole point of input_type, and it's what EmbedKind selects.
    let c = client(Some(1024));
    let text = "how do I center clay on the wheel";
    let as_query = c
        .embed(&[text.into()], EmbedKind::Query)
        .await
        .expect("query");
    let as_doc = c
        .embed(&[text.into()], EmbedKind::Document)
        .await
        .expect("doc");
    assert_ne!(
        as_query[0], as_doc[0],
        "query- and document-side vectors should differ (asymmetric retrieval)"
    );
}

#[tokio::test]
#[ignore = "needs VOYAGE_API_KEY and network"]
async fn embeddings_are_semantically_sane() {
    let c = client(Some(1024));
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
    eprintln!("voyage relevant={sim_relevant:.4} unrelated={sim_unrelated:.4}");
    assert!(sim_relevant > sim_unrelated);
}
