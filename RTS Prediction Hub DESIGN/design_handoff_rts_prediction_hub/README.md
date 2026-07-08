# Handoff: RTS Prediction Hub

## Overview
RTS Prediction Hub is an internal, data-dense analytics dashboard. A user picks one of three
prediction apps and runs live predictions against a shared LightGBM endpoint (DGX Spark). The
UI is chart-forward with a modern analytics aesthetic (Linear / Vercel / Observable), dark by
default with a light-mode toggle.

The three apps:
1. **Multifamily — Tenant Move-Out Risk**
2. **Brokerage — Prospect Conversion (Signal Desk)**
3. **Marketing — Cohort & Industry Analytics**

## About the Design Files
The files in this bundle are **design references created in HTML** — a working prototype showing
the intended look, layout, charts, and interactions. They are **not** production code to copy
verbatim. The task is to **recreate these designs in your target codebase** using its established
patterns and libraries. The prototype was authored with React + Recharts + inline styles; the
spec it was built to calls for **React + Tailwind + Recharts** (with a CSS-grid heatmap, since
Recharts has no native heatmap). Use that stack if you have no existing environment; otherwise
adapt to your codebase's component and styling conventions.

## Fidelity
**High-fidelity (hifi).** Final colors, typography, spacing, chart types, and interactions are
all specified below. Recreate pixel-closely, then swap the mock data layer for real API calls.

## Data Layer & API Contract (read this first)
All data is isolated in **one module, `api.js`** (TypeScript target: `api.ts`), exposing async
functions. The prototype returns mock data shaped EXACTLY like the live responses, so wiring the
real endpoint is a one-line swap per function.

- **Live endpoint:** `POST /predict/{contract}` with header `X-API-Key: <key>`.
- Each function currently `await delay(...)` + returns mock JSON; replace the body with
  `return fetchPredict('<contract>', body, apiKey)`.

Functions:
- `predictMultifamily({ buildings, asOf, maxUnits })` → multifamily result
- `listBrokerageCompanies()` → `[{ hMy, display_name, snapshots }]`
- `predictBrokerage({ companyId, query })` → brokerage result
- `aggregateMarketing({ mode, key })` → marketing result  (mode: `'cohort' | 'industry'`)
- Constants exported: `MODEL_VERSION`, `RISK_COLORS`, `MK_COHORTS`, `MK_INDUSTRIES`,
  `SIZE_BAND_ORDER`, `VERTICAL_ORDER`.

### Response shapes
**Multifamily**
```json
{ "model_version":"v0-standin", "model_source":"real", "as_of":"2026-06-23",
  "results":[ {"scode":"t0000021","full_name":"...","unit":"A-101",
    "property":"Country Club Village","days_to_lease_end":45,"has_notice":1,
    "actual_churn":0,"churn_probability":0.48,"risk_tier":"high"} ] }
```
**Brokerage**
```json
{ "company":{"hMy":840,"display_name":"Interiors Revived","market":"DSM","industry":"Retail"},
  "current_rules_score":{"Probability":0.40,"Decile":2,"ScoredAt":"2026-04-21"},
  "history":[ {"snapshot_date":"2024-11-30","conversion_probability":0.03,"risk_tier":"low"} ] }
```
**Marketing**
```json
{ "segment":"healthcare-vertical",
  "by_market":[ {"Market":"DSM","n":55,"avg_score":11.1,"avg_win_prob":0.21} ],
  "decile_distribution":[ {"Decile":1,"n":49} ],
  "cohort_win_rates":[ {"Vertical":"healthcare","SizeBand":"micro","DealCount":42,"WinRate":0.167} ] }
```

### Risk tier rule
`tier = p >= 0.6 ? 'high' : p >= 0.3 ? 'medium' : 'low'`. Probabilities are 0–1; always display
as percentages.

## Global Layout
Two-column app shell.

- **Left sidebar** — fixed `268px`, sticky full-height, `background: --panel`, right border `--border`.
  - Logo mark: 30×30, radius 8, `linear-gradient(135deg,#5b8def,#8b5cf6)`, white "R", glow shadow.
  - Title "RTS Prediction Hub" (15px/700), subtitle "Shared LightGBM endpoint · DGX Spark" (11.5px, --muted).
  - **Status pill** "API connected": green dot (7px, `#16a34a`, pulsing `rtsPulse 2s`) + label `#22c55e`,
    bg `color-mix(in srgb,#16a34a 14%,transparent)`, border `…35%`, radius 999.
  - **App switcher** (3 items): each is a full-width button, 28×28 letter icon tile + title (12.5px/600) +
    sub (10.5px, --muted, truncated). Active item: bg `--panel2`, border `--border`. Inactive: transparent.
    Icon tints — MF `#7aa5f5` on `#5b8def/18%`; BK `#a78bfa` on `#8b5cf6/18%`; MK `#4ade80` on `#22c55e/18%`.
  - **Bottom**: theme toggle button (full width, bordered, --panel2) showing "Dark theme ◒" / "Light theme ☀",
    and caption "model: v0-standin (interim)" (10px, mono, --muted, centered).
- **Main area** — `flex:1`, `max-width:1500px`, padding `32px 36px 64px`.
  - Header: kicker (11px/600, uppercase, letter-spacing .09em, `--accent`), H1 (26px/700, -.02em),
    description (14px, --muted, max-width 680).
  - Controls row (bordered card, --panel, radius 14, padding 18/20, flex-wrap, gap 18, align-end).
  - KPI row (CSS grid; 4 cols MF/BK, 3 cols MK; gap 16).
  - Charts grid (responsive; see each view).
  - Detail table / aggregation block.
  - Result caption (11px, mono, --muted): `model: v0-standin (interim) · source: real · <date>`.
- **Loading**: shimmer skeletons (`.rts-skel`, gradient sweep `rtsShimmer 1.3s`) replacing KPI + chart cards.
- **Responsive**: stacks to 1 column on mobile (grids → single column; sidebar collapses).

## Screens / Views

### View 1 — Multifamily (Tenant Move-Out Risk)
- **Controls:** building multiselect as toggle **chips** (Country Club Village, Sierra Pointe,
  Residences at 62W, Winwood — all selected by default; active chip bg `color-mix(#5b8def 20%)`,
  text `#9bbbf7`); **as-of** `<input type=date>` (mono); **max units** range slider 10–400 step 10
  (default 120, `accent-color:--accent`), label shows live value; **Run prediction** button (primary).
- **KPI cards:** Units Scored · Mean Move-Out Probability (1 decimal %) · High-Risk Count (red `#dc2626`,
  with "% of portfolio" sub) · Tenants With Notice Filed (amber `#f59e0b`).
- **Charts:**
  - **Histogram** of `churn_probability`, 20 bins (width 5%), each bar `<Cell>`-colored by the bin's
    risk tier. X ticks every 4th bin ("0%…80%"), grid dashed `3 3` horizontal only.
  - **Donut** (Recharts Pie, innerRadius 62 / outerRadius 96, paddingAngle 2) of low/medium/high counts,
    with centered total count + "scored" label. Legend row of 3 colored squares below.
  - **Top 15 at-risk** horizontal bar (`layout="vertical"`), X domain 0–1 as %, Y = tenant name (width 130).
    Red gradient by rank: `rgb(220, g, g)` where `g = 165 - 105*(p/maxP)`. Value label at bar end.
  - **Sortable table:** columns Tenant (`scode`, mono muted) · Property · Unit (mono) · Move-out prob
    (right, %) · Tier (badge) · Notice (chip "Filed" amber / "—") · Actual (Yes/No). Click any header to
    sort (toggles desc↔asc; arrow ↑/↓ on active col). Sticky header, max-height 520 scroll, cap 200 rows.

### View 2 — Brokerage (Prospect Conversion / Signal Desk)
- **Controls:** company search `<input type=text>` (by id or name); **sample companies** `<select>`
  ("Name · N snapshots"); **Run prediction** button. Selecting a company clears the search box.
- **Company header strip** (--panel2 card): display_name (17px/700) · `hMy` (mono muted) · market pill ·
  industry pill.
- **KPI cards:** Latest Model Probability (tier-colored) · History Points · Current Rules Probability
  (sub: "scored <ScoredAt>") · Rules Decile (`--accent`, sub "1 = lowest, 10 = highest").
- **Charts:**
  - **Time-series area** (Recharts AreaChart) of `conversion_probability` over snapshot date (markers on,
    `dot r=3`, activeDot 5). Y domain 0–1 as %. Accent gradient fill (`bkGrad`, .45→.02). Subtle shaded
    risk bands via ReferenceArea: 0–.3 green / .3–.6 amber / .6–1 red, all fillOpacity .05. Dashed
    **ReferenceLine** at current rules score (`#94a3b8`, `5 4`) labeled "rules NN%".
  - **Risk-tier strip:** thin flex row, one segment per snapshot, each `flex:1` colored by tier
    (opacity .85), 2px gaps, radius 6, height 26. Start/end dates beneath (mono, 10px).
  - **History table:** Snapshot date (mono) · Conversion probability (right, %) · Risk tier badge.
    Newest first. Sticky header, max-height 420 scroll.

### View 3 — Marketing (Cohort & Industry Analytics)
- **Controls:** segmented toggle **[Cohort | Industry]** (active segment bg `--accent`, white text);
  options `<select>` (cohorts: healthcare-vertical, growth-mode, tech-vertical, lease-expiring-12mo;
  industries: Banking, Retail, Healthcare, Manufacturing, Technology); **Aggregate** button.
  Switching mode resets the select to that mode's first option.
- **KPI cards (3):** Companies in Segment (sum of `by_market.n`) · Average Probability (n-weighted
  `avg_win_prob`, `--accent`) · Markets Covered (count of markets with n>0).
- **Charts:**
  - **Grouped/colored bar** of `avg_win_prob` by market (DSM/OMA/OTHER), bars colored
    `[--accent, #8b5cf6, #22c55e]`, **company count labeled on top** of each bar (`n=NN`). Y as %.
  - **Decile distribution** bar (deciles 1–10, `n` companies), **blue gradient** by decile
    `hsl(217,80%, L%)` with `L = 70 - 22*(i/9)`.
  - **Win-rate heatmap** — **CSS grid**, `grid-template-columns: 140px repeat(5,1fr)`, gap 6.
    Rows = verticals (healthcare, finance-insurance, technology, retail, manufacturing, real-estate);
    columns = size bands (micro, small, mid, large, enterprise). Cell bg
    `color-mix(in srgb, #16a34a <6–94>%, --panel2)` scaled by `winRate/maxWinRate`; cell text white when
    intensity > .55 else `--text`; value shown in-cell as %. Header legend shows 0% → green ramp → max%.
    Tooltip (title attr): "<vertical> · <band>: NN.N% win · NN deals".
  - **Cohort win rates** — top 10 by win rate, each a row: label (vertical · band) + `n=deals` + a
    horizontal bar in a `--panel2` track whose **width = dealCount/maxDeals** (animated) with green
    gradient whose intensity tracks win rate, + win-rate % at the right.

## Interactions & Behavior
- **App switch:** clicking a sidebar item sets the active view; first visit to BK or MK auto-runs its
  prediction. MF auto-runs on initial mount.
- **Run / Aggregate:** sets `loading=true` → await api fn → store data → `loading=false` → start counter
  tween. Button label shows "Running…/Aggregating…" and `cursor:wait`, opacity .6 while loading.
- **Animated KPI counters:** on each new result, `anim` tweens 0→1 over **750ms, easeOutCubic**
  (`1 - (1-p)^3`) via requestAnimationFrame; displayed KPI numbers = target × anim. The deal-count bars
  in MK also scale with `anim`.
- **Theme toggle:** flips `dark`; recolors via CSS custom properties (see tokens). `transition: background .25s`.
  Charts re-render with theme-aware axis/grid/tooltip colors.
- **Table sort (MF):** click header → sort by that key; clicking the active key flips direction.
- **Chart tooltips:** Recharts Tooltip, `contentStyle` = --panel bg, --border, radius 10, 12px; values
  formatted ("NN.N%", "NN tenants", "n=NN", etc.). Cursor highlight `rgba(127,127,140,.08)`.
- **Hover:** chips, nav items, buttons have `transition .15s` color/bg shifts.

## State Management
Single component holds:
- `view` `'mf'|'bk'|'mk'`, `dark` boolean, `loading` boolean, `anim` 0–1 (tween progress).
- MF: `mfData`, `selBuildings[]`, `asOf`, `maxUnits`, `sortKey`, `sortDir`.
- BK: `bkData`, `bkCompanies[]`, `bkCompany`, `bkQuery`.
- MK: `mkData`, `mkMode`, `mkKey`.
- Data fetching: the four `api` functions above. No global store needed; local component state +
  the isolated `api` module is sufficient. In React, lift to a small context/store only if you split
  views into routes.

## Design Tokens
**Colors — dark (default):**
`--bg #0a0a0b` · `--panel #161618` · `--panel2 #1e1e21` · `--border rgba(255,255,255,.08)` ·
`--text #ededef` · `--muted #8e8e96` · `--accent #5b8def` · scrollbar `#2a2a2e`.
**Colors — light:**
`--bg #f6f6f7` · `--panel #ffffff` · `--panel2 #f1f1f3` · `--border rgba(0,0,0,.08)` ·
`--text #18181b` · `--muted #6b7280` · `--accent #2563eb` · scrollbar `#d4d4d8`.
**Risk system (both themes):** low `#16a34a` · medium `#f59e0b` · high `#dc2626`.
Badges/fills use `color-mix(in srgb, <risk> 14–16%, transparent)` with `…35–38%` borders.
**Accent secondaries:** violet `#8b5cf6`, green `#22c55e`; logo gradient `#5b8def→#8b5cf6`.
**Typography:** UI = **Inter Tight** (400/500/600/700); numbers, codes, dates = **JetBrains Mono**
(400/500/600). Scale: H1 26/700/-.02em · card titles 13.5/600 · KPI value 30/700/-.02em mono ·
body 14 · labels 11–11.5/600 · captions 10–11 mono. Kicker 11/600 uppercase, letter-spacing .09em.
**Radius:** cards 14 · controls/inputs 9–10 · chips/pills 999 · small chips 6–7 · heatmap cells 7.
**Spacing:** card padding 18px 20px · grid gaps 16 · controls gap 18 · section margins ~16–22.
**Shadows:** primary button `0 2px 12px rgba(91,141,239,.35)`; logo `0 2px 10px rgba(91,141,239,.35)`.
**Animations:** `rtsPulse` (status dot, 2s), `rtsShimmer` (skeletons, 1.3s linear), counter tween
750ms easeOutCubic, bar width transition .4s.

## Assets
None external — no image assets. Icons are typographic letter tiles (R / M / B / A) and unicode
glyphs (◒, ☀, ↑, ↓). No icon font required. Fonts load from Google Fonts (Inter Tight, JetBrains Mono).
Charts: **Recharts** (UMD in the prototype; npm `recharts` in production — it requires `prop-types`
as a peer when using the UMD build). Heatmap is pure CSS grid (no chart lib).

## Files
- `RTS Prediction Hub.dc.html` — the full prototype (shell, all three views, chart logic). Authored
  as a streaming Design Component; the markup/logic translate directly to a React component tree.
- `api.js` — the isolated data module (mock generators + documented live-fetch stub). This is the
  contract to implement against; port it to `api.ts` and swap mock bodies for `fetchPredict(...)`.
