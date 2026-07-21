//! Voyage AI: `{base}/embeddings`. Bearer auth, asymmetric `input_type`, and an
//! optional `output_dimension`. Shares the `data`/`index` response with OpenAI.

use serde::Serialize;

use super::{DataResponse, decode_err, error_for_status, request_err, sorted_by_index};
use crate::{Client, EmbedKind, Error, Provider};

pub(crate) async fn embed(
    client: &Client,
    texts: &[String],
    kind: EmbedKind,
) -> Result<Vec<Vec<f32>>, Error> {
    let url = format!("{}/embeddings", client.base_url);
    let body = VoyageRequest {
        input: texts,
        model: &client.model,
        input_type: kind.as_str(),
        output_dimension: client.output_dimension,
    };
    let resp = client
        .http
        .post(&url)
        .bearer_auth(client.require_key())
        .json(&body)
        .send()
        .await
        .map_err(request_err(Provider::Voyage))?;
    let resp = error_for_status(Provider::Voyage, resp).await?;
    let parsed: DataResponse = resp.json().await.map_err(decode_err(Provider::Voyage))?;
    Ok(sorted_by_index(parsed.data))
}

#[derive(Serialize)]
struct VoyageRequest<'a> {
    input: &'a [String],
    model: &'a str,
    input_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_dimension: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_carries_input_type_and_dim() {
        let body = VoyageRequest {
            input: &["q".to_string()],
            model: "voyage-3.5-lite",
            input_type: EmbedKind::Query.as_str(),
            output_dimension: Some(1024),
        };
        let j = serde_json::to_value(&body).unwrap();
        assert_eq!(j["input_type"], "query");
        assert_eq!(j["model"], "voyage-3.5-lite");
        assert_eq!(j["output_dimension"], 1024);
    }

    #[test]
    fn request_omits_dim_when_unset() {
        let body = VoyageRequest {
            input: &["q".to_string()],
            model: "m",
            input_type: EmbedKind::Document.as_str(),
            output_dimension: None,
        };
        let j = serde_json::to_value(&body).unwrap();
        assert!(j.get("output_dimension").is_none());
    }
}
