# FrontierHarness Eval: Maki 0.5.5 (run 2026-09-19-maki)

Third-party evaluation of the maki coding harness (v0.5.5, repo commit
`3a0c8de56008edcf4255df1c00f10ab6a2fe1980`) against the published 30-task
FrontierHarness set. Model: Kimi K3 served by Neuralwatt (`neuralwatt/kimi-k3`,
provider `custom`, host `api.neuralwatt.com`).

## Result

**24/30 passed (80.0%)** · cost per pass $1.75 · median cost/task $0.30 ·
median successful-task cost $0.21 · median task wall time 1m 41s ·
median cache hit rate 65.9% · mean 28.1 turns.
Full metrics and task-by-task evidence: `runs/2026-09-19-maki/report/REPORT.md`
(+ `index.html`, `chart.svg`). The `runs/` evidence (trajectories, verifier
logs, model patches) is retained locally by default (`runs/` is gitignored);
attach it on request or publish a gist for the report-only view.

Two exclusive solves: `terminal-bench/kv-store-grpc` and
`terminal-bench/largest-eigenval` passed with 0/12 published baselines passing.

## Harness under test

- Binary: static musl release `v0.5.5`, pinned GitHub release tarball
- Registered in Harbor 0.22.0 and datacurve-pier 0.3.1 by name `maki`
  (`AgentName` enum + factory patch; see `maki-agent/` and `install-maki.sh`,
  embedded verbatim in `install-maki-standalone.sh`)
- Defaults evaluated: `always_workflow = true`, `always_thinking = "max"`,
  `always_yolo = true` (seeded as `~/.config/maki/init.lua` in each trial
  container), provider pinned via `~/.config/maki/providers.toml`
- Invocation: `maki -p --yolo --trust --verbose --output-format stream-json -m neuralwatt/kimi-k3 "<instruction>"`

## Deviations from the published pipeline (all visible as diffs on this branch)

- `skills/frontierharness-eval/scripts/providers.sh` — trial egress allowlist
  extended: `docker.io`, `*.docker.com` (cloudfront blob host), `ghcr.io`.
  Recorded in each run's `run.json`; new-run-id rule respected per change.
- `skills/frontierharness-eval/scripts/run-trials.sh` — `prepare_image` waits
  for dockerd readiness and retries pulls with 20s spacing (frozen dockerd
  resumes transients right after restore).
- `skills/frontierharness-eval/scripts/usage_details.py` — registry entry
  mapping maki's stream-json `maki.txt` to the Claude-Code-compatible usage
  parser (maki emits CC envelope shapes). Fixes turns/cold-cost/cache fields
  for this harness only; no scoring-rule changes.
- maki install in task containers downloads the release tarball straight from
  GitHub (`github.com` was already allowlisted; `maki.sh` was not).
- Task-image builds inject the Runta egress CA (public cert, validity
  1975-01-01–4096-01-01) for the MITM'd egress proxy; `SSL_CERT_FILE` /
  `CURL_CA_BUNDLE` point at the augmented system bundle.

## Provenance

- Golden checkpoint: `fh-golden-maki-v055` (manifest:
  `manifest-fh-golden-maki-v055.json`; full manifest lives at
  `/work/manifest.json` inside restored runtimes)
- DeepSWE corpus pinned: `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`
- Runtime per trial: 4 vCPU / 8 GiB / 50 GiB disk, fresh restore per task
- Credential handling: `NEURALWATT_API_KEY` stored as a Runta tenant secret;
  runtimes see only the stub `runta-secret-stub`; the egress proxy injects
  the real key for `api.neuralwatt.com`. No key material appears in this
  repository (audited).
- Costs use the frozen benchmark price table (kimi-k3-2026-08-20:
  $3/$15 per 1M in/out, $0.30/1M cache read), not provider billing.

## Comparability status

Unranked (`comparable: false`). New-provider serving of Kimi K3 is not proven
equivalent to the published baselines' route, the trial egress policy differs
from the (unrecorded) published policy, and a matched control with identical
policy was not run. Treat positions in the chart/report as observed results
under stated conditions.

## Re-running

The golden checkpoint lives in the evaluating tenant; rebuild via
`skills/frontierharness-eval/scripts/provision-golden-checkpoint.sh` with
`--install-script install-maki-standalone.sh`, then
`run-trials.sh --checkpoint <name> --harness maki --provider custom
--model neuralwatt/kimi-k3 --secret-name NEURALWATT_API_KEY
--secret-host api.neuralwatt.com --run-id <id> --out runs`.
