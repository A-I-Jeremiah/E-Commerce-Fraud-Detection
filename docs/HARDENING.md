# Production Hardening Checklist

## API Security
- [ ] Restrict CORS `allow_origins` to known frontend domains
- [ ] Add rate limiting (e.g. `slowapi` or API gateway)
- [ ] Enforce maximum batch size (already capped at 500)
- [ ] Run behind a reverse proxy (nginx / Cloud Load Balancer) with TLS
- [ ] Do not expose `/docs` in production (or protect it)

## Secrets & Configuration
- [ ] Move any future secrets to environment variables or a secret manager
- [ ] Never commit real credentials or production data
- [ ] Use read-only volume mounts for model artifacts in Docker

## Model & Data
- [ ] Validate input ranges more strictly in Pydantic if needed
- [ ] Monitor prediction volume and latency
- [ ] Keep training data and prediction logs access-controlled
- [ ] Review SHAP / feature importance periodically for concept drift

## Operational
- [ ] Set resource limits (CPU / memory) on the container
- [ ] Configure liveness & readiness probes (already have `/health`)
- [ ] Centralise logs (JSONL → ELK / Cloud Logging / Datadog)
- [ ] Alert on:
  - High drift (PSI > 0.25 on multiple features)
  - Performance degradation
  - API error rate / latency spikes
  - Retrain triggers

## Deployment
- [ ] Prefer blue-green or canary deployment for new model versions
- [ ] Keep previous model artifacts for fast rollback
- [ ] Document the exact artifact set that constitutes a release