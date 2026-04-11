//! Criterion benchmarks — CSR graph traversal
//! cargo bench --bench graph_traversal
//!
//! Measures:
//!   - neighbors() lookup (O(1) slice)
//!   - BFS depth=1,2,3 at 1k / 10k / 100k node graphs
//!   - CSR build time from edge list
//!   - CSR vs simulated adjacency-list (baseline comparison)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;
use ai_graph_engine::graph::csr::{CsrGraph, EdgeInfo, NodeInfo};

// ── graph factories ───────────────────────────────────────────────────────────

fn make_node(id: u32) -> NodeInfo {
    NodeInfo { id, name: format!("n{id}"), node_type: "E".into(), weight: 0.0, props: serde_json::Value::Null, full_context: String::new(), embed_context: None }
}

/// Random-ish dense graph: each node has ~avg_degree neighbors
fn make_graph(num_nodes: usize, avg_degree: usize) -> CsrGraph {
    let nodes: Vec<NodeInfo> = (0..num_nodes as u32).map(make_node).collect();
    let mut edges: Vec<EdgeInfo> = Vec::with_capacity(num_nodes * avg_degree);
    for i in 0..num_nodes {
        for d in 1..=avg_degree {
            let to = (i + d) % num_nodes;
            edges.push(EdgeInfo { from: i as u32, to: to as u32, edge_type: String::new(), weight: 1.0, full_context: String::new(), embed_context: None });
        }
    }
    CsrGraph::build(nodes, &edges)
}

/// Simulated adjacency list baseline (Vec<Vec<u32>>) for comparison
struct AdjList(Vec<Vec<u32>>);

impl AdjList {
    fn build(num_nodes: usize, avg_degree: usize) -> Self {
        let mut adj = vec![Vec::new(); num_nodes];
        for i in 0..num_nodes {
            for d in 1..=avg_degree {
                adj[i].push(((i + d) % num_nodes) as u32);
            }
        }
        AdjList(adj)
    }
    fn neighbors(&self, node: u32) -> &[u32] { &self.0[node as usize] }
}

// ── benchmarks ────────────────────────────────────────────────────────────────

fn bench_neighbors_lookup(c: &mut Criterion) {
    let g = make_graph(100_000, 8);
    let mut group = c.benchmark_group("neighbors_lookup");
    group.throughput(Throughput::Elements(1));

    group.bench_function("CSR_neighbors_100k", |b| {
        b.iter(|| {
            // access a spread of nodes to avoid cache warming the same line
            for i in (0..1000u32).step_by(97) {
                black_box(g.neighbors(i % 100_000));
            }
        })
    });

    let adj = AdjList::build(100_000, 8);
    group.bench_function("AdjList_neighbors_100k", |b| {
        b.iter(|| {
            for i in (0..1000u32).step_by(97) {
                black_box(adj.neighbors(i % 100_000));
            }
        })
    });

    group.finish();
}

fn bench_bfs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_traversal");

    for &(size, label) in &[(1_000usize, "1k"), (10_000, "10k"), (100_000, "100k")] {
        let g = make_graph(size, 6);
        let seeds = vec![0u32, size as u32 / 4, size as u32 / 2];

        for &depth in &[1u8, 2, 3] {
            group.bench_with_input(
                BenchmarkId::new(format!("depth{depth}"), label),
                &(&g, &seeds),
                |b, (g, seeds)| {
                    b.iter(|| {
                        black_box(g.bfs_expand(seeds, depth, 10_000, false))
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_build");

    for &(size, label) in &[(1_000usize, "1k"), (10_000, "10k"), (100_000, "100k")] {
        let nodes: Vec<NodeInfo> = (0..size as u32).map(make_node).collect();
        let edges: Vec<EdgeInfo> = (0..size)
            .flat_map(|i| (1..=6usize).map(move |d| EdgeInfo {
                from: i as u32,
                to: ((i + d) % size) as u32,
                edge_type: String::new(),
                weight: 1.0,
                full_context: String::new(),
                embed_context: None,
            }))
            .collect();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("build", label),
            &(nodes.clone(), edges.clone()),
            |b, (nodes, edges)| {
                b.iter(|| {
                    black_box(CsrGraph::build(nodes.clone(), edges))
                })
            },
        );
    }
    group.finish();
}

/// Sequential traversal of 1000 consecutive nodes.
/// This reveals CSR's true memory-locality advantage over AdjList.
fn bench_sequential_traversal(c: &mut Criterion) {
    let g   = make_graph(100_000, 8);
    let adj = AdjList::build(100_000, 8);

    let mut group = c.benchmark_group("sequential_traversal");
    group.throughput(Throughput::Elements(1_000));

    group.bench_function("CSR_1000_nodes", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for i in 0..1_000u32 {
                total += g.neighbors(i).len();
            }
            black_box(total)
        })
    });

    group.bench_function("AdjList_1000_nodes", |b| {
        b.iter(|| {
            let mut total = 0usize;
            for i in 0..1_000u32 {
                total += adj.neighbors(i).len();
            }
            black_box(total)
        })
    });

    group.finish();
}

/// Parallel BFS — 8 independent queries on the same graph via Rayon.
fn bench_parallel_bfs(c: &mut Criterion) {
    use rayon::prelude::*;

    let g = make_graph(100_000, 8);
    // 8 disjoint seed sets spread across the graph
    let seed_sets: Vec<Vec<u32>> = (0..8u32)
        .map(|i| vec![i * 12_500])
        .collect();

    let mut group = c.benchmark_group("parallel_bfs");
    group.throughput(Throughput::Elements(8));

    group.bench_function("8_bfs_sequential", |b| {
        b.iter(|| {
            seed_sets.iter()
                .map(|s| g.bfs_expand(s, 2, 500, false))
                .collect::<Vec<_>>()
        })
    });

    group.bench_function("8_bfs_rayon_par", |b| {
        b.iter(|| {
            seed_sets.par_iter()
                .map(|s| g.bfs_expand(s, 2, 500, false))
                .collect::<Vec<_>>()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_neighbors_lookup, bench_bfs, bench_build, bench_sequential_traversal, bench_parallel_bfs);
criterion_main!(benches);
