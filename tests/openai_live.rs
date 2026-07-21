//! End-to-end tests against the live OpenAI embeddings API (skipped by default).
//! Run with a key (and optionally point at an OpenAI-compatible endpoint):
//!
//! ```sh
//! OPENAI_API_KEY=... cargo test --test openai_live -- --ignored --nocapture
//! ```
//!
//! Model/base URL come from the environment so the same test covers OpenAI and
//! any OpenAI-compatible server (together.ai, vLLM, LocalAI, ...).

use lean_embed::{Client, EmbedKind, Provider};

fn model() -> String {
    std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "text-embedding-3-small".into())
}

fn client(output_dimension: Option<usize>) -> Client {
    let mut b = Client::builder(Provider::OpenAi, model());
    if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
        b = b.base_url(base);
    }
    if let Some(d) = output_dimension {
        b = b.output_dimension(d);
    }
    b.build()
        .expect("build openai client (is OPENAI_API_KEY set?)")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY and network"]
async fn documents_embed_at_the_pinned_dimension() {
    // text-embedding-3-* honour `dimensions`; prove it round-trips + validates.
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
        assert_eq!(v.len(), 256, "dimensions must be honoured");
        assert!(v.iter().all(|x| x.is_finite()));
    }
}

#[tokio::test]
#[ignore = "needs OPENAI_API_KEY and network"]
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
    // OpenAI embeddings are symmetric; EmbedKind is ignored (Document is fine).
    let q = c
        .embed(
            &["How hot does a pottery oven get?".into()],
            EmbedKind::Query,
        )
        .await
        .expect("query");
    let sim_relevant = cosine(&q[0], &doc_vecs[0]);
    let sim_unrelated = cosine(&q[0], &doc_vecs[1]);
    eprintln!("openai relevant={sim_relevant:.4} unrelated={sim_unrelated:.4}");
    assert!(sim_relevant > sim_unrelated);
}
