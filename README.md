# GTA VI trophy-rarity forecast — gta6-trophy-rarity-forecast-01

Forecast, for every Grand Theft Auto VI PlayStation 5 trophy, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at the case target.
Entry for make.fun bounty `gta6-trophy-rarity-forecast-01`, account
@gentle-beacon-34. One file, Node 22 standard library only, offline,
deterministic.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory, writes `<id>.out` for each:
`{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`.

## The model (see the comments in `main.ts` for the full reasoning)

Exophase shows rates as of its **last refresh**, not live play. On launch
night the page sat unchanged for hours (638 players tracked) and then
refreshed once: 3,868 players tracked, trophies earned 943 → 4,276, every
rate but the 100% one lower. That refresh is pure dilution arithmetic: the
3,230 new players all hold the one 100% trophy, so the other 54 trophies
gained only 103 earners between them — a new tracked player earns a given
trophy at roughly 7% of the launch cohort's rate. A rate-only trend model
misses this completely.

So the program works in **counts, not rates**: earners `E(t)` and players
tracked `N(t)` are fitted separately against `u = ln(1 + hours since
release)`, and the rate is `E/N`. A refresh that adds barely-played
players moves `N` a lot and `E` a little, which is exactly what the split
captures.

Two regimes, chosen from the case itself, never from the case id:

1. **Flat history** (every read identical, checkpoint within a day of
   release, short horizon): no trend exists to fit; the page is stale and
   a refresh is overdue. The forecast assumes the launch-night refresh
   pattern — `N` × ~6, new-cohort earn fraction ~0.067 — so a non-100%
   rate is multiplied by ~0.221, while a trophy at 100% stays at 100%.
2. **Trend history** (the final's two days of distinct reads): log `E`
   and log `N` against `u` by recency-weighted least squares, leaning on
   the recent half of the window because earn curves saturate; each
   trophy's slope is shrunk toward its tier's pooled slope and the global
   slope. A second estimator (logit rate against `u`, anchored on the
   last read) cross-checks the counts estimator; the median is their mean
   in logit space, with saturation damping for trophies already at high
   rates. The spread adds fit uncertainty, estimator disagreement, count
   noise and refresh-lag uncertainty (floored, capped, never tuned to a
   single cut).

## Validation

The kit publishes no practice answers (the server grades them), so the
practice loss below is a simulation, not a server score: answers were
generated to match the kit's own published refresh aggregates exactly
(3,868 players tracked, 4,276 trophies earned, the 100% trophy staying
at 100%), with the 103 new earners distributed across trophies several
ways (proportional to launch rate, and Dirichlet-concentrated).

| scenario | this entry | house baseline (~1.51 stated) |
|---|---|---|
| practice, simulated answers | **0.14–0.21** | 1.41–1.69 |
| synthetic final, saturating growth | **0.17** | 0.28 |
| synthetic final, power-law growth | **0.16–0.18** | 0.34–0.46 |

The synthetic finals are self-built stand-ins for the real final case
(their history was generated with refresh-quantised reads, growing
player counts and per-trophy progression curves); they test the trend
regime's code path and its margin over persistence, not the real final
score, which nobody can know before 26 November.

Runtime: ~0.1 s per case; two runs are byte-identical.

## Files

- `main.ts` — the whole entry (Node 22, stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — the account's wallet, as the kit requires
