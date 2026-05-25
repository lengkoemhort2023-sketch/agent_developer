# AMK Agent Observability Stack

This document describes the production-grade observability system implemented for the AMK Agent platform.

## Overview

The observability stack consists of:

1. **Prometheus** - Metrics collection and storage
2. **OpenTelemetry** - Distributed tracing instrumentation
3. **Grafana Tempo** - Trace storage and visualization
4. **Grafana Loki** - Log aggregation and analysis
5. **Grafana** - Unified dashboard platform
6. **Langfuse** - LLM observability (optional)

## Architecture

```
┌─────────────┐         ┌──────────────┐
│  Django App │────────▶│  Prometheus  │
└─────────────┘         └──────────────┘
       │                       │
       │                       ▼
       ├─────────────────────▶Grafana
       │                       ▲
       │ Traces                │
       ▼                       │
    Tempo ◀──────────────────┘
       │
       │ Logs
       ▼
    Loki
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- The main AMK Agent codebase
- 16GB+ RAM recommended
- Adequate disk space for metrics, logs, and traces

### Starting the Observability Stack

1. **Load observability variables** (optional):
   ```bash
   cat .env.observability >> .env.dev
   ```

2. **Start with compose**:
   ```bash
   # Start everything (backend + observability)
   docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d
   
   # Or just observability stack
   docker-compose -f docker-compose.observability.yml up -d
   ```

3. **Wait for services to initialize** (30-60 seconds):
   ```bash
   docker-compose logs -f grafana
   ```

### Accessing the Stack

| Service | URL | Default Credentials |
|---------|-----|------------------|
| **Grafana** | http://localhost:3000 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |
| **Tempo** | http://localhost:3200 | - |
| **Loki** | http://localhost:3100 | - |

## Metrics

### Prometheus Metrics Endpoint

Django exposes metrics at: `http://localhost:8000/api/metrics/`

Prometheus is configured to scrape this endpoint every 10 seconds.

### Available Metrics

#### API Metrics
- `api_requests_total` - Total API requests (by method, endpoint, status)
- `api_request_duration_seconds` - Request latency histogram (p95, p99)
- `api_errors_total` - Error rate (by method, endpoint, error type)

#### RAG Pipeline Metrics
- `rag_retrieval_duration_seconds` - Document retrieval latency
- `rag_llm_duration_seconds` - LLM response generation time
- `rag_total_duration_seconds` - Full pipeline latency
- `rag_retrievals_total` - Retrieval count (by status)

#### Database Metrics
- `db_query_duration_seconds` - Query execution time
- `db_connection_pool_size` - Connection pool utilization

#### Celery Task Metrics
- `celery_task_duration_seconds` - Task execution time
- `celery_tasks_total` - Task count (by name, status)

## Distributed Tracing

### OpenTelemetry Instrumentation

Automatic instrumentation is enabled for:
- Django HTTP requests
- Database queries (SQLAlchemy)
- External HTTP requests
- Celery tasks

### Custom Spans

Instrument RAG operations using context managers:

```python
from app.core.rag_metrics import track_rag_pipeline, track_rag_retrieval

# Full pipeline tracking
with track_rag_pipeline("generative") as ctx:
    # Retrieval
    with track_rag_retrieval("vector") as ret_metrics:
        documents = retriever.search(query)
        ret_metrics["success"] = True
    
    # LLM call
    with track_rag_llm_call("mistral") as llm_metrics:
        response = llm.generate(context=documents)
        llm_metrics["tokens_used"] = 150
    
    ctx["doc_count"] = len(documents)
```

### Viewing Traces

1. Open Grafana: http://localhost:3000
2. Navigate to "Explore"
3. Select "Tempo" as data source
4. Search by service name, trace ID, or span attributes

## Structured Logging

### Log Format

All logs are output in JSON format for easy parsing by Loki:

```json
{
  "timestamp": "2026-05-25T15:10:01.483000Z",
  "level": "INFO",
  "logger": "chat.views",
  "message": "API Request: POST /api/chats/message",
  "module": "views",
  "function": "chat_message",
  "line": 42,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "duration_ms": 145.23
}
```

### Accessing Logs via Loki

1. Open Grafana: http://localhost:3000
2. Navigate to "Explore"
3. Select "Loki" as data source
4. Use LogQL to query:

```logql
{job="django"} | json | level="ERROR"
{job="django"} | json | duration_ms > 1000
```

## Dashboards

### Pre-configured Dashboards

1. **System Metrics** (`system-metrics`)
   - Django memory and CPU usage
   - API request rates and latency (p95/p99)
   - System resource utilization

2. **RAG Pipeline** (`rag-pipeline`)
   - Retrieval latency distribution
   - LLM generation time
   - Pipeline throughput and error rates
   - Service map of trace dependencies

### Creating Custom Dashboards

1. Open Grafana: http://localhost:3000
2. Click "+" → "Dashboard"
3. Add panels by selecting data sources:
   - **Prometheus** for metrics
   - **Loki** for logs
   - **Tempo** for traces

## Performance Tuning

### Retention Policies

Edit the configurations to adjust retention:

**Prometheus** (`monitoring/prometheus.yml`):
```yaml
command:
  - '--storage.tsdb.retention.time=15d'
```

**Loki** (`monitoring/loki.yml`):
```yaml
limits_config:
  retention_period: 1440h  # 60 days
```

**Tempo** (`monitoring/tempo.yml`):
```yaml
storage:
  trace:
    wal:
      path: /var/tempo/wal
```

### Resource Limits

Adjust Docker resource limits in `docker-compose.observability.yml`:

```yaml
services:
  prometheus:
    deploy:
      resources:
        limits:
          memory: 4G  # Increase for high-volume metrics
```

## Integrating Langfuse (Optional)

Langfuse provides detailed LLM observability:

1. **Set environment variables**:
   ```bash
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=<your_key>
   LANGFUSE_SECRET_KEY=<your_key>
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

2. **Decorate LLM functions**:
   ```python
   from app.core.observability import observability
   
   @observability.langfuse.observe()
   def generate_response(query, context):
       return llm.generate(...)
   ```

3. **Access Langfuse dashboard**: https://cloud.langfuse.com

## Troubleshooting

### Metrics not appearing

1. Check Django is running: `curl http://localhost:8000/api/metrics/`
2. Verify Prometheus can scrape: http://localhost:9090/targets
3. Check middleware is enabled in settings.py

### High memory usage

- Reduce `PROMETHEUS_METRICS_EXPORT_PORT` scrape interval
- Lower Tempo trace sampling rate
- Increase retention time if storage is full

### Traces missing

1. Verify `OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317`
2. Check logs: `docker-compose logs tempo`
3. Ensure services can communicate via Docker network

### No logs in Loki

1. Check Loki is running: `docker-compose ps loki`
2. Verify logging format is JSON
3. Confirm `LOKI_ENABLED=true`

## Security Considerations

### Production Deployment

1. **Change Grafana password**: Admin > Preferences
2. **Use authentication** for Prometheus/Loki (reverse proxy with OAuth)
3. **Enable TLS** for OTLP endpoints
4. **Limit metrics exposure**: Don't expose `/metrics` publicly
5. **Rotate API keys**: For Langfuse, use environment-specific keys

### Data Privacy

- Remove sensitive data from traces (PII, credentials)
- Use trace sampling to reduce sensitive data volume
- Implement data retention policies compliant with regulations

## API Reference

### Metrics Endpoint

```bash
# Get all metrics
curl http://localhost:8000/api/metrics/

# Prometheus format (OpenMetrics)
# Used by Prometheus scraper
# Content-Type: application/openmetrics-text; version=1.0.0; charset=utf-8
```

### Middleware Hooks

**APIMetricsMiddleware** - Automatic API metrics:
```python
# Tracks automatically:
# - api_requests_total
# - api_request_duration_seconds
# - api_errors_total

# Customize paths to exclude:
EXCLUDED_PATHS = {"/health/", "/metrics/", "/admin/"}
```

**RequestIDMiddleware** - Request tracing:
```python
# Generates unique request_id for tracing
# Add to response headers: X-Request-ID
# Use in logs for request correlation
```

## Maintenance

### Regular Checks

- Monitor disk usage: `df -h /var/lib/docker/volumes/`
- Review error rates: Check Grafana dashboards
- Test alert rules: Verify alerting works
- Update images: `docker-compose pull && docker-compose up -d`

### Backup

```bash
# Backup Prometheus data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup \
  busybox tar czf /backup/prometheus-backup.tar.gz -C /data .

# Backup Grafana dashboards
docker exec grafana grafana-cli admin export-dashboard > dashboard-backup.json
```

## Further Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Tempo Documentation](https://grafana.com/docs/tempo/latest/)
- [Loki Documentation](https://grafana.com/docs/loki/latest/)

## Support

For issues or questions:
1. Check logs: `docker-compose logs <service>`
2. Review configuration files in `monitoring/`
3. Verify all services are healthy: `docker-compose ps`
