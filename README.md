[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/gta6-trophy-rarity-forecast-01) [![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alfasurya91/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/gta6-trophy-rarity-forecast-01/alfasurya91/gta6-trophy-rarity-forecast) [![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/alfasurya91/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/gta6-trophy-rarity-forecast-01/alfasurya91/gta6-trophy-rarity-forecast)

# GTA VI Trophy Rarity Forecast

Forecast the Exophase earn rate of all 55 GTA VI PS5 trophies at a future
checkpoint, reported as five quantiles per trophy (0.05 / 0.25 / 0.50 / 0.75 /
0.95). Pure standard library, fully offline and deterministic: two runs over the
same input produce byte-identical output.

Bounty: [make.fun/bounty/gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

## Model: threshold-cohort holders players

The tracked players, `P(t)`, and the per-trophy holder counts, `H_i(t)`, are the
only observations; everything else is reconstructed.

1. **History collapse.** The Exophase page only changes when it refreshes, so
   the raw history is collapsed to the distinct displayed states
   `(t_k, P_k, H_{i,k})`. Repeated identical reads carry no new information and
   would otherwise bias the fit toward whatever the page happened to show
   longest.

2. **Population as cohorts.** The player base is rebuilt as cohorts from the
   `P(t)` history rather than treated as one homogeneous pool:
   - an initial cohort at the first read, split into a small veteran share
     already deep into the game and a fresh remainder with a fraction of an
     hour banked;
   - newcomers between consecutive distinct states, sized by the increase in
     tracked players and joining at the midpoint of the two reads;
   - projected newcomers between the last state and the target, spread over
     the remaining horizon.

   Each cohort draws a play rate from a lognormal whose median decays with join
   time, so early adopters play harder than late arrivals.

3. **Diurnal-weighted hours.** Every player accumulates play hours weighted by
   a diurnal factor over the UTC day (near zero overnight, peaking in the
   evening), so a trophy earned "in 10 hours" is not 10 wall-clock hours for a
   player who joined at 02:00 UTC. Newcomers idle briefly after joining for
   install and tutorial before any hours count.

4. **Per-trophy thresholds.** Trophy `i` is earned when a player's accumulated
   hours cross that trophy's threshold distribution: a median hours-to-earn
   `m_i`, a ceiling `c_i` (the share of players who ever reach it at all), a
   threshold spread, and a hard floor. Story chapters are tight — nearly
   everyone who plays the chapter gets the trophy, quickly. Grinds are wide —
   long medians and low ceilings. Both `(m_i, c_i)` are fit per trophy against
   the distinct-state jumps, with a class prior drawn from the trophy's
   name/description/tier pulling the fit back toward its prior when the
   observations are thin.

5. **`P(t)` projection.** The player base is projected with a power-law fit
   over the collapsed states, with a launch-phase prior used when the display
   is still frozen and no refresh has been seen. Holder deltas for the initial
   cohort are anchored to the observed count at the first state, so only the
   modelled change is trusted; newcomers are counted outright.

6. **Monte Carlo quantiles.** Parameter uncertainty (`m_i`, `c_i`, play rate,
   rate decay, and the projected player base) is propagated by Monte Carlo with
   **200 samples** and a **fixed seed** (`SEED = 20261119`), giving reproducible
   quantiles. Samples are sorted and read at the five target levels, clamped to
   `[0.01, 99.99]`, rounded half-up to 4 decimals, and forced non-decreasing
   across levels.

## Run locally

The program reads every `*.in` case from `MAKEFUN_INPUT_DIR` (default `/input`)
and writes the matching `*.out` file to `MAKEFUN_OUTPUT_DIR` (default
`/output`). Both are plain environment variables, so no arguments are needed:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=./cases MAKEFUN_OUTPUT_DIR=./out python3 -P -s main.py
```

Each case file is JSON with `at`, `horizon_minutes`, `trophies` and `history`;
each output file is JSON of the form `{"trophies": [{"id": 0, "q": [q05, q25,
q50, q75, q95]}, ...]}`.

Set `MAKEFUN_OUTPUT_DIR` to a scratch directory, never to the input directory,
since both may be the same tree during local experiments.

## License

MIT — see [LICENSE](LICENSE).
