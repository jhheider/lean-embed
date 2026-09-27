//! Ollama: a local server at `{base}/api/embed`. No key, offline, fixed-width.

use serde::{Deserialize, Serialize};

use crate::EmbedKind;

use super::{decode_err, error_for_status, request_err};
use crate::{Client, Error, Provider};

pub(crate) async fn embed(
    client: &Client,
    texts: &[String],
    kind: EmbedKind,
) -> Result<Vec<Vec<f32>>, Error> {
    let url = format!("{}/api/embed", client.base_url);
    let input = prepare_input(&client.model, texts, kind, client.task_prefixes);
    let body = OllamaRequest {
        model: &client.model,
        input: &input,
    };
    let resp = client
        .http
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(request_err(Provider::Ollama))?;
    let resp = error_for_status(Provider::Ollama, resp).await?;
    let parsed: OllamaResponse = resp.json().await.map_err(decode_err(Provider::Ollama))?;
    Ok(parsed.embeddings)
}

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct OllamaResponse {
    embeddings: Vec<Vec<f32>>,
}

/// Nomic-style task instructions, applied only when the builder opted
/// in (`ClientBuilder::task_prefixes`): the published nomic-embed-text
/// model is trained with `search_query: ` / `search_document: `
/// prefixes, and Ollama's nomic modelfile template is a bare
/// `{{ .Prompt }}` - it accepts `input_type` but does not apply it
/// (verified against Ollama 0.32.15: raw == input_type, raw !=
/// prefixed). Other models get no munging; extend this keyed list when
/// a model needs its own scheme.
fn task_prefix(model: &str, kind: EmbedKind) -> Option<&'static str> {
    if model.starts_with("nomic-embed-text") {
        Some(match kind {
            EmbedKind::Query => "search_query: ",
            EmbedKind::Document => "search_document: ",
        })
    } else {
        None
    }
}

/// Apply the model's task prefix, if it has one.
fn prepare_input(model: &str, texts: &[String], kind: EmbedKind, enabled: bool) -> Vec<String> {
    if !enabled {
        return texts.to_vec();
    }
    match task_prefix(model, kind) {
        Some(prefix) => texts.iter().map(|t| format!("{prefix}{t}")).collect(),
        None => texts.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nomic_models_get_task_prefixes_by_kind() {
        assert_eq!(
            task_prefix("nomic-embed-text", EmbedKind::Query),
            Some("search_query: ")
        );
        assert_eq!(
            task_prefix("nomic-embed-text:latest", EmbedKind::Document),
            Some("search_document: ")
        );
        assert_eq!(task_prefix("bge-m3", EmbedKind::Query), None);
    }

    #[test]
    fn prepare_input_is_opt_in() {
        let texts = vec!["grappling".to_string()];
        assert_eq!(
            prepare_input("nomic-embed-text", &texts, EmbedKind::Query, true),
            ["search_query: grappling"]
        );
        assert_eq!(
            prepare_input("nomic-embed-text", &texts, EmbedKind::Document, true),
            ["search_document: grappling"]
        );
        assert_eq!(
            prepare_input("bge-m3", &texts, EmbedKind::Query, true),
            ["grappling"]
        );
        // Off (the default): byte-identical input to 0.1.0.
        assert_eq!(
            prepare_input("nomic-embed-text", &texts, EmbedKind::Query, false),
            ["grappling"]
        );
    }

    #[test]
    fn request_serializes_to_model_and_input() {
        let body = OllamaRequest {
            model: "nomic-embed-text",
            input: &["a".to_string(), "b".to_string()],
        };
        let j = serde_json::to_value(&body).unwrap();
        assert_eq!(j["model"], "nomic-embed-text");
        assert_eq!(j["input"][1], "b");
    }

    #[test]
    fn response_parses_embeddings() {
        let r: OllamaResponse =
            serde_json::from_str(r#"{"embeddings":[[0.1,0.2],[0.3,0.4]]}"#).unwrap();
        assert_eq!(r.embeddings.len(), 2);
        assert_eq!(r.embeddings[0], vec![0.1, 0.2]);
    }
}
