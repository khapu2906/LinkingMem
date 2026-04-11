/// Ingest CLI — one-time data preparation.
/// Usage: cargo run --bin ingest -- --input data.json --data-dir ../data

use std::path::PathBuf;
use anyhow::Result;
use ai_graph_engine::{
    config::PluginsConfig,
    graph::builder::{from_json_payload, save as save_graph},
    plugin::PluginClient,
    storage::LocalStorage,
    vector::{hnsw::normalise, store::VectorStore},
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

    let args: Vec<String> = std::env::args().collect();
    let input_path = get_arg(&args, "--input").unwrap_or_else(|| "input.json".into());
    let data_dir   = PathBuf::from(get_arg(&args, "--data-dir").unwrap_or_else(|| "../data".into()));
    let plugin_url = get_arg(&args, "--plugin-url").unwrap_or_else(|| "http://localhost:8001".into());

    tracing::info!("reading {}", input_path);
    let raw     = std::fs::read_to_string(&input_path)?;
    let payload: serde_json::Value = serde_json::from_str(&raw)?;

    // ── build graph ─────────────────────────────────────────────────────
    let graph = from_json_payload(&payload)?;
    tracing::info!("graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());

    // ── embed entity names via Python ────────────────────────────────────
    let plugins_cfg = PluginsConfig::from_single_url(&plugin_url);
    let plugin = PluginClient::new(&plugins_cfg)?;
    let names: Vec<String> = graph.nodes.iter().map(|n| n.name.clone()).collect();

    tracing::info!("embedding {} nodes in batches of 256...", names.len());
    let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(names.len());
    for chunk in names.chunks(256) {
        let mut vecs = plugin.embed(chunk.to_vec()).await?;
        for v in &mut vecs { normalise(v); }
        all_vecs.extend(vecs);
    }

    let dim = all_vecs.first().map(|v| v.len()).unwrap_or(0);
    tracing::info!("embeddings: {} × dim={}", all_vecs.len(), dim);

    // ── save all artefacts ───────────────────────────────────────────────
    std::fs::create_dir_all(&data_dir)?;
    let storage = LocalStorage::new(data_dir.clone());
    save_graph(&graph, &storage)?;
    VectorStore::write(&data_dir.join("vectors.bin"), dim, &all_vecs)?;

    tracing::info!("ingest complete → {}", data_dir.display());
    Ok(())
}

fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
