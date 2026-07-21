//! OpenAI, or any OpenAI-compatible `/v1/embeddings` endpoint. Bearer auth,
//! symmetric (no `input_type`), `dimensions` for the requested width. Shares the
//! `data`/`index` response with Voyage.

use serde::Serialize;

use super::{DataResponse, decode_err, error_for_status, request_err, sorted_by_index};
use crate::{Client, Error, Provider};

pub(crate) async fn embed(client: &Client, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
    let url = format!("{}/embeddings", client.base_url);
    let body = OpenAiRequest {
        input: texts,
        model: &client.model,
        encoding_format: "float",
        dimensions: client.output_dimension,
    };
    let resp = client
        .http
        .post(&url)
        .bearer_auth(client.require_key())
        .json(&body)
        .send()
        .await
        .map_err(request_err(Provider::OpenAi))?;
    let resp = error_for_status(Provider::OpenAi, resp).await?;
    let parsed: DataResponse = resp.json().await.map_err(decode_err(Provider::OpenAi))?;
    Ok(sorted_by_index(parsed.data))
}

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    input: &'a [String],
    model: &'a str,
    encoding_format: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_carries_encoding_format_and_dimensions() {
        let body = OpenAiRequest {
            input: &["q".to_string()],
            model: "text-embedding-3-small",
            encoding_format: "float",
            dimensions: Some(512),
        };
        let j = serde_json::to_value(&body).unwrap();
        assert_eq!(j["encoding_format"], "float");
        assert_eq!(j["dimensions"], 512);
        assert_eq!(j["model"], "text-embedding-3-small");
    }

    #[test]
    fn request_omits_dimensions_when_unset() {
        let body = OpenAiRequest {
            input: &["q".to_string()],
            model: "m",
            encoding_format: "float",
            dimensions: None,
        };
        let j = serde_json::to_value(&body).unwrap();
        assert!(j.get("dimensions").is_none());
    }
}
