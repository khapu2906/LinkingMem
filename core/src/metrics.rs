/// Metrics — in-process Prometheus-style counters and histograms.
///
/// Exposed on GET /metrics as plain text (Prometheus format).
/// No external dependency — pure std atomics.
///
/// Tracked:
///   queries_total          counter
///   queries_failed_total   counter
///   query_latency_ms       histogram (buckets: 50,100,250,500,1000,2500,∞)
///   embed_latency_ms       histogram
///   llm_latency_ms         histogram
///   cache_hits_total       counter
///   cache_misses_total     counter
///   delta_size             gauge
///   graph_nodes            gauge
///   graph_edges            gauge

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ── histogram ────────────────────────────────────────────────────────────────

const BUCKETS: &[u64] = &[50, 100, 250, 500, 1000, 2500, u64::MAX];

pub struct Histogram {
    buckets: Vec<AtomicU64>, // len = BUCKETS.len()
    sum: AtomicU64,
    count: AtomicU64,
    name: &'static str,
}

impl Histogram {
    pub fn new(name: &'static str) -> Self {
        Self {
            buckets: (0..BUCKETS.len()).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            name,
        }
    }

    pub fn observe(&self, value_ms: u64) {
        self.sum.fetch_add(value_ms, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Prometheus histograms are cumulative: every bucket whose upper bound
        // is >= value gets incremented. Store in the first matching bucket only
        // and accumulate at render time.
        for (i, &upper) in BUCKETS.iter().enumerate() {
            if value_ms <= upper {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // value exceeds all finite buckets — falls into +Inf (last bucket)
        self.buckets[BUCKETS.len() - 1].fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        // Buckets must be rendered as cumulative counts (Prometheus spec).
        let mut cumulative = 0u64;
        for (i, &upper) in BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            let le = if upper == u64::MAX { "+Inf".to_string() } else { upper.to_string() };
            out.push_str(&format!("{}_bucket{{le=\"{}\"}} {}\n", self.name, le, cumulative));
        }
        out.push_str(&format!("{}_sum {}\n",   self.name, self.sum.load(Ordering::Relaxed)));
        out.push_str(&format!("{}_count {}\n", self.name, self.count.load(Ordering::Relaxed)));
        out
    }

    pub fn p50_ms(&self) -> u64 { self.percentile(0.50) }
    pub fn p95_ms(&self) -> u64 { self.percentile(0.95) }
    pub fn p99_ms(&self) -> u64 { self.percentile(0.99) }

    fn percentile(&self, p: f64) -> u64 {
        let total = self.count.load(Ordering::Relaxed);
        if total == 0 { return 0; }
        let target = (total as f64 * p).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &upper) in BUCKETS.iter().enumerate() {
            cumulative += self.buckets[i].load(Ordering::Relaxed);
            if cumulative >= target {
                return if upper == u64::MAX { 9999 } else { upper };
            }
        }
        9999
    }
}

// ── counter / gauge ──────────────────────────────────────────────────────────

pub struct Counter {
    value: AtomicU64,
    name: &'static str,
}

impl Counter {
    pub fn new(name: &'static str) -> Self { Self { value: AtomicU64::new(0), name } }
    pub fn inc(&self)             { self.value.fetch_add(1, Ordering::Relaxed); }
    pub fn add(&self, n: u64)     { self.value.fetch_add(n, Ordering::Relaxed); }
    pub fn get(&self) -> u64      { self.value.load(Ordering::Relaxed) }
    pub fn render(&self) -> String { format!("{} {}\n", self.name, self.get()) }
}

pub struct Gauge {
    value: AtomicU64,
    name: &'static str,
}

impl Gauge {
    pub fn new(name: &'static str) -> Self { Self { value: AtomicU64::new(0), name } }
    pub fn set(&self, v: u64) { self.value.store(v, Ordering::Relaxed); }
    pub fn get(&self) -> u64  { self.value.load(Ordering::Relaxed) }
    pub fn render(&self) -> String { format!("{} {}\n", self.name, self.get()) }
}

// ── metrics registry ─────────────────────────────────────────────────────────

pub struct Metrics {
    pub queries_total:        Counter,
    pub queries_failed_total: Counter,
    pub query_latency:        Histogram,
    pub embed_latency:        Histogram,
    pub llm_latency:          Histogram,
    pub cache_hits:           Counter,
    pub cache_misses:         Counter,
    pub ingest_total:         Counter,
    pub ingest_failed_total:  Counter,
    pub nodes_ingested_total: Counter,
    pub ingest_latency:       Histogram,
    pub delta_size:           Gauge,
    pub graph_nodes:          Gauge,
    pub graph_edges:          Gauge,
}

impl Metrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queries_total:        Counter::new("queries_total"),
            queries_failed_total: Counter::new("queries_failed_total"),
            query_latency:        Histogram::new("query_latency_ms"),
            embed_latency:        Histogram::new("embed_latency_ms"),
            llm_latency:          Histogram::new("llm_latency_ms"),
            cache_hits:           Counter::new("cache_hits_total"),
            cache_misses:         Counter::new("cache_misses_total"),
            ingest_total:         Counter::new("ingest_total"),
            ingest_failed_total:  Counter::new("ingest_failed_total"),
            nodes_ingested_total: Counter::new("nodes_ingested_total"),
            ingest_latency:       Histogram::new("ingest_latency_ms"),
            delta_size:           Gauge::new("delta_size"),
            graph_nodes:          Gauge::new("graph_nodes"),
            graph_edges:          Gauge::new("graph_edges"),
        })
    }

    /// Render all metrics in Prometheus text format
    pub fn render_prometheus(&self) -> String {
        let mut out = String::new();
        out.push_str("# HELP queries_total Total query requests\n# TYPE queries_total counter\n");
        out.push_str(&self.queries_total.render());
        out.push_str("# HELP queries_failed_total Failed queries\n# TYPE queries_failed_total counter\n");
        out.push_str(&self.queries_failed_total.render());
        out.push_str("# HELP query_latency_ms Query end-to-end latency\n# TYPE query_latency_ms histogram\n");
        out.push_str(&self.query_latency.render());
        out.push_str("# HELP embed_latency_ms Embedding latency\n# TYPE embed_latency_ms histogram\n");
        out.push_str(&self.embed_latency.render());
        out.push_str("# HELP llm_latency_ms LLM generation latency\n# TYPE llm_latency_ms histogram\n");
        out.push_str(&self.llm_latency.render());
        out.push_str("# HELP cache_hits_total LRU cache hits\n# TYPE cache_hits_total counter\n");
        out.push_str(&self.cache_hits.render());
        out.push_str("# HELP cache_misses_total LRU cache misses\n# TYPE cache_misses_total counter\n");
        out.push_str(&self.cache_misses.render());
        out.push_str("# HELP ingest_total Total ingest requests\n# TYPE ingest_total counter\n");
        out.push_str(&self.ingest_total.render());
        out.push_str("# HELP ingest_failed_total Failed ingest requests\n# TYPE ingest_failed_total counter\n");
        out.push_str(&self.ingest_failed_total.render());
        out.push_str("# HELP nodes_ingested_total Total new nodes committed to delta\n# TYPE nodes_ingested_total counter\n");
        out.push_str(&self.nodes_ingested_total.render());
        out.push_str("# HELP ingest_latency_ms Ingest end-to-end latency\n# TYPE ingest_latency_ms histogram\n");
        out.push_str(&self.ingest_latency.render());
        out.push_str("# HELP delta_size Current delta buffer size\n# TYPE delta_size gauge\n");
        out.push_str(&self.delta_size.render());
        out.push_str("# HELP graph_nodes Total nodes in main graph\n# TYPE graph_nodes gauge\n");
        out.push_str(&self.graph_nodes.render());
        out.push_str("# HELP graph_edges Total edges in main graph\n# TYPE graph_edges gauge\n");
        out.push_str(&self.graph_edges.render());
        out
    }

    /// Compact JSON summary for /health endpoint
    pub fn summary_json(&self) -> serde_json::Value {
        let cache_total = self.cache_hits.get() + self.cache_misses.get();
        let hit_rate = if cache_total > 0 {
            self.cache_hits.get() as f64 / cache_total as f64
        } else {
            0.0
        };

        serde_json::json!({
            "queries": {
                "total":          self.queries_total.get(),
                "failed":         self.queries_failed_total.get(),
                "latency_p50_ms": self.query_latency.p50_ms(),
                "latency_p95_ms": self.query_latency.p95_ms(),
                "latency_p99_ms": self.query_latency.p99_ms(),
            },
            "cache": {
                "hits":     self.cache_hits.get(),
                "misses":   self.cache_misses.get(),
                "hit_rate": format!("{:.1}%", hit_rate * 100.0),
            },
            "ingest": {
                "total":          self.ingest_total.get(),
                "failed":         self.ingest_failed_total.get(),
                "nodes_ingested": self.nodes_ingested_total.get(),
                "latency_p95_ms": self.ingest_latency.p95_ms(),
            },
            "graph": {
                "nodes":      self.graph_nodes.get(),
                "edges":      self.graph_edges.get(),
                "delta_size": self.delta_size.get(),
            }
        })
    }
}

// ── timer helper ─────────────────────────────────────────────────────────────

/// RAII timer: records elapsed ms to histogram on drop.
pub struct Timer<'a> {
    start: Instant,
    histogram: &'a Histogram,
}

impl<'a> Timer<'a> {
    pub fn new(h: &'a Histogram) -> Self {
        Self { start: Instant::now(), histogram: h }
    }
}

impl<'a> Drop for Timer<'a> {
    fn drop(&mut self) {
        self.histogram.observe(self.start.elapsed().as_millis() as u64);
    }
}
