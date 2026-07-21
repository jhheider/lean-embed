//! End-to-end tests against the live Gemini embeddings API (skipped by default).
//! Run with a key:
//!
//! ```sh
//! GEMINI_API_KEY=... cargo test --test gemini_live -- --ignored --nocapture
//! ```
//!
//! Exercises the `taskType` asymmetry and `outputDimensionality`. Model comes
//! from the environment (`GEMINI_MODEL`, default `text-embedding-004`); the
//! client `models/`-prefixes it.

use lean_embed::{Client, EmbedKind, Provider};

fn model() -> String {
    std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "text-embedding-004".into())
}

fn client(output_dimension: Option<usize>) -> Client {
    let mut b = Client::builder(Provider::Gemini, model());
    if let Ok(base) = std::env::var("GEMINI_BASE_URL") {
        b = b.base_url(base);
    }
    if let Some(d) = output_dimension {
        b = b.output_dimension(d);
    }
    b.build()
        .expect("build gemini client (is GEMINI_API_KEY set?)")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::test]
#[ignore = "needs GEMINI_API_KEY and network"]
async fn documents_embed_at_the_pinned_dimension() {
    let c = client(Some(256));
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
        assert_eq!(v.len(), 256, "outputDimensionality must be honoured");
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[tokio::test]
#[ignore = "needs GEMINI_API_KEY and network"]
async fn query_and_document_vectors_are_asymmetric() {
    // Gemini's taskType gives distinct query- and document-side vectors.
    let c = client(None);
    let text = "how do I center clay on the wheel";
    let as_query = c
        .embed(&[text.into()], EmbedKind::Query)
        .await
        .expect("query");
    let as_doc = c
        .embed(&[text.into()], EmbedKind::Document)
        .await
        .expect("doc");
    assert_ne!(as_query[0], as_doc[0], "taskType should make them differ");
}

#[tokio::test]
#[ignore = "needs GEMINI_API_KEY and network"]
async fn embeddings_are_semantically_sane() {
    let c = client(None);
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
    eprintln!("gemini relevant={sim_relevant:.4} unrelated={sim_unrelated:.4}");
    assert!(sim_relevant > sim_unrelated);
}
