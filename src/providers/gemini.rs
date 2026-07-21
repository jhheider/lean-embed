//! Google Gemini: `{base}/models/{model}:batchEmbedContents`. `x-goog-api-key`
//! header auth, `taskType` from [`EmbedKind`], optional `outputDimensionality`.
//! The model is `models/`-prefixed in both the URL path and each sub-request.

use serde::{Deserialize, Serialize};

use super::{decode_err, error_for_status, request_err};
use crate::{Client, EmbedKind, Error, Provider};

pub(crate) async fn embed(
    client: &Client,
    texts: &[String],
    kind: EmbedKind,
) -> Result<Vec<Vec<f32>>, Error> {
    let model = if client.model.starts_with("models/") {
        client.model.clone()
    } else {
        format!("models/{}", client.model)
    };
    let url = format!("{}/{}:batchEmbedContents", client.base_url, model);
    let requests = texts
        .iter()
        .map(|t| GeminiEmbedRequest {
            model: &model,
            content: GeminiContent {
                parts: [GeminiPart { text: t }],
            },
            task_type: kind.gemini_task_type(),
            output_dimensionality: client.output_dimension,
        })
        .collect();
    let body = GeminiRequest { requests };
    let resp = client
        .http
        .post(&url)
        .header("x-goog-api-key", client.require_key())
        .json(&body)
        .send()
        .await
        .map_err(request_err(Provider::Gemini))?;
    let resp = error_for_status(Provider::Gemini, resp).await?;
    let parsed: GeminiResponse = resp.json().await.map_err(decode_err(Provider::Gemini))?;
    // Gemini returns embeddings in request order (no index field).
    Ok(parsed.embeddings.into_iter().map(|e| e.values).collect())
}

#[derive(Serialize)]
struct GeminiRequest<'a> {
    requests: Vec<GeminiEmbedRequest<'a>>,
}

#[derive(Serialize)]
struct GeminiEmbedRequest<'a> {
    model: &'a str,
    content: GeminiContent<'a>,
    #[serde(rename = "taskType")]
    task_type: &'a str,
    #[serde(
        rename = "outputDimensionality",
        skip_serializing_if = "Option::is_none"
    )]
    output_dimensionality: Option<usize>,
}

#[derive(Serialize)]
struct GeminiContent<'a> {
    parts: [GeminiPart<'a>; 1],
}

#[derive(Serialize)]
struct GeminiPart<'a> {
    text: &'a str,
}

#[derive(Deserialize)]
struct GeminiResponse {
    embeddings: Vec<GeminiEmbedding>,
}

#[derive(Deserialize)]
struct GeminiEmbedding {
    values: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_nests_content_and_maps_task_type() {
        let body = GeminiRequest {
            requests: vec![GeminiEmbedRequest {
                model: "models/text-embedding-004",
                content: GeminiContent {
                    parts: [GeminiPart { text: "hello" }],
                },
                task_type: EmbedKind::Query.gemini_task_type(),
                output_dimensionality: Some(768),
            }],
        };
        let j = serde_json::to_value(&body).unwrap();
        assert_eq!(j["requests"][0]["model"], "models/text-embedding-004");
        assert_eq!(j["requests"][0]["content"]["parts"][0]["text"], "hello");
        assert_eq!(j["requests"][0]["taskType"], "RETRIEVAL_QUERY");
        assert_eq!(j["requests"][0]["outputDimensionality"], 768);
    }

    #[test]
    fn response_parses_values() {
        let r: GeminiResponse =
            serde_json::from_str(r#"{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]}]}"#)
                .unwrap();
        assert_eq!(r.embeddings.len(), 2);
        assert_eq!(r.embeddings[1].values, vec![0.3, 0.4]);
    }
}
