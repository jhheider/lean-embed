//! Provider wire implementations. Each submodule owns exactly one provider's
//! request/response types and its `embed` function; [`Client::embed`] dispatches
//! to them. Adding a provider is a new file here plus a [`Provider`] variant and
//! a match arm - no other module changes.
//!
//! [`Client::embed`]: crate::Client::embed

use serde::Deserialize;

use crate::{Error, Provider};

pub(crate) mod gemini;
pub(crate) mod ollama;
pub(crate) mod openai;
pub(crate) mod voyage;

/// Turn a non-success HTTP response into an [`Error::Api`] carrying the body.
pub(crate) async fn error_for_status(
    provider: Provider,
    resp: reqwest::Response,
) -> Result<reqwest::Response, Error> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    Err(Error::Api {
        provider: provider.label(),
        status,
        body,
    })
}

/// Map a transport error to [`Error::Request`], boxing it opaque.
pub(crate) fn request_err(provider: Provider) -> impl FnOnce(reqwest::Error) -> Error {
    move |e| Error::Request {
        provider: provider.label(),
        source: Box::new(e),
    }
}

/// Map a decode error to [`Error::Decode`], boxing it opaque.
pub(crate) fn decode_err(provider: Provider) -> impl FnOnce(reqwest::Error) -> Error {
    move |e| Error::Decode {
        provider: provider.label(),
        source: Box::new(e),
    }
}

/// The response shape shared by Voyage and OpenAI: `{"data":[{embedding,index}]}`.
#[derive(Deserialize)]
pub(crate) struct DataResponse {
    pub(crate) data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
pub(crate) struct EmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

/// Reorder an OpenAI-style `data` array by its `index` and drop to bare vectors.
/// The APIs return data in input order, but sorting is cheap insurance.
pub(crate) fn sorted_by_index(mut data: Vec<EmbeddingDatum>) -> Vec<Vec<f32>> {
    data.sort_by_key(|d| d.index);
    data.into_iter().map(|d| d.embedding).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_response_parses_and_sorts_by_index() {
        // The shape shared by Voyage and OpenAI.
        let r: DataResponse = serde_json::from_str(
            r#"{"data":[{"embedding":[2.0],"index":1},{"embedding":[1.0],"index":0}]}"#,
        )
        .unwrap();
        assert_eq!(sorted_by_index(r.data), vec![vec![1.0], vec![2.0]]);
    }
}
