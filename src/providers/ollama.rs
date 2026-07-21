//! Ollama: a local server at `{base}/api/embed`. No key, offline, fixed-width.

use serde::{Deserialize, Serialize};

use super::{decode_err, error_for_status, request_err};
use crate::{Client, Error, Provider};

pub(crate) async fn embed(client: &Client, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
    let url = format!("{}/api/embed", client.base_url);
    let body = OllamaRequest {
        model: &client.model,
        input: texts,
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

#[cfg(test)]
mod tests {
    use super::*;

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
