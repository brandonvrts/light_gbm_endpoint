// ============================================================================
// RTS Prediction Hub — data layer
// All mock data lives here, shaped EXACTLY like the live API responses.
// To go live, replace each function body with a real call:
//   POST /predict/{contract}  with header  X-API-Key: <key>
// e.g.  return fetchPredict('multifamily-churn', body)
// ----------------------------------------------------------------------------
export const MODEL_VERSION = 'v0-standin';
export const RISK_COLORS = { low: '#16a34a', medium: '#f59e0b', high: '#dc2626' };

// ---- live-fetch stub (kept here so the swap is one line per function) -------
const API_BASE = ''; // e.g. 'https://dgx-spark.internal'
async function fetchPredict(contract, body, apiKey = 'REPLACE_ME') {
  const res = await fetch(`${API_BASE}/predict/${contract}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`predict/${contract} -> ${res.status}`);
  return res.json();
}
// keep the linter/bundler from tree-shaking it away while unused:
export const _live = { fetchPredict };

// ---- deterministic RNG so charts look stable per input ----------------------
function seeded(seedStr) {
  let h = 1779033703 ^ seedStr.length;
  for (let i = 0; i < seedStr.length; i++) {
    h = Math.imul(h ^ seedStr.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}
const delay = (ms) => new Promise((r) => setTimeout(r, ms));
function tierFor(p) { return p >= 0.6 ? 'high' : p >= 0.3 ? 'medium' : 'low'; }

const FIRST = ['James','Maria','Chen','Aisha','Robert','Linda','Diego','Sara','Tom','Nia','Omar','Grace','Kyle','Priya','Luis','Hana','Mark','Eva','Sven','Rosa','Cole','Mei','Jack','Iris'];
const LAST = ['Nguyen','Patel','Garcia','Smith','Johnson','Okafor','Lee','Brown','Rossi','Khan','Murphy','Singh','Adams','Torres','Cole','Park','Diaz','Reed','Bauer','Frank'];

// ============================================================================
// VIEW 1 — MULTIFAMILY: Tenant Move-Out Risk
// ============================================================================
export async function predictMultifamily({ buildings, asOf, maxUnits }) {
  await delay(650);
  const props = (buildings && buildings.length) ? buildings
    : ['Country Club Village', 'Sierra Pointe', 'Residences at 62W', 'Winwood'];
  const rng = seeded(`mf|${props.join(',')}|${asOf}|${maxUnits}`);
  const n = Math.max(8, Math.min(maxUnits || 120, 400));
  const results = [];
  for (let i = 0; i < n; i++) {
    const property = props[Math.floor(rng() * props.length)];
    // skew toward lower probabilities, with a meaningful high-risk tail
    let p = Math.pow(rng(), 1.7) * 0.55 + (rng() < 0.12 ? 0.35 : 0);
    p = Math.min(0.97, Math.max(0.01, p + (rng() - 0.5) * 0.08));
    const tier = tierFor(p);
    const has_notice = (tier === 'high' ? rng() < 0.55 : tier === 'medium' ? rng() < 0.18 : rng() < 0.04) ? 1 : 0;
    const actual_churn = (p > 0.5 ? rng() < p : rng() < p * 0.5) ? 1 : 0;
    const block = String.fromCharCode(65 + Math.floor(rng() * 6));
    results.push({
      scode: 't' + String(Math.floor(rng() * 9_000_00) + 21).padStart(7, '0'),
      full_name: `${FIRST[Math.floor(rng() * FIRST.length)]} ${LAST[Math.floor(rng() * LAST.length)]}`,
      unit: `${block}-${100 + Math.floor(rng() * 240)}`,
      property,
      days_to_lease_end: Math.floor(rng() * 330) + 5,
      has_notice,
      actual_churn,
      churn_probability: Math.round(p * 1000) / 1000,
      risk_tier: tier,
    });
  }
  return { model_version: MODEL_VERSION, model_source: 'real', as_of: asOf, results };
}

// ============================================================================
// VIEW 2 — BROKERAGE: Prospect Conversion / Signal Desk
// ============================================================================
const BK_COMPANIES = [
  { hMy: 840, display_name: 'Interiors Revived', market: 'DSM', industry: 'Retail', snapshots: 19 },
  { hMy: 1207, display_name: 'Prairie Dental Group', market: 'DSM', industry: 'Healthcare', snapshots: 22 },
  { hMy: 633, display_name: 'Cornerstone Capital', market: 'OMA', industry: 'Banking', snapshots: 15 },
  { hMy: 1455, display_name: 'Meridian Logistics', market: 'OTHER', industry: 'Manufacturing', snapshots: 24 },
  { hMy: 902, display_name: 'Birchwood Software', market: 'DSM', industry: 'Technology', snapshots: 17 },
  { hMy: 318, display_name: 'Heartland Outfitters', market: 'OMA', industry: 'Retail', snapshots: 12 },
];

export async function listBrokerageCompanies() {
  await delay(180);
  return BK_COMPANIES.map((c) => ({ hMy: c.hMy, display_name: c.display_name, snapshots: c.snapshots }));
}

export async function predictBrokerage({ companyId, query }) {
  await delay(600);
  let company = BK_COMPANIES.find((c) => String(c.hMy) === String(companyId));
  if (!company && query) {
    const q = String(query).toLowerCase();
    company = BK_COMPANIES.find((c) => c.display_name.toLowerCase().includes(q) || String(c.hMy) === q);
  }
  if (!company) company = BK_COMPANIES[0];
  const rng = seeded(`bk|${company.hMy}`);
  const months = company.snapshots;
  const history = [];
  let p = rng() * 0.06 + 0.01;
  const start = new Date(2024, 10, 30);
  for (let i = 0; i < months; i++) {
    const d = new Date(start.getFullYear(), start.getMonth() + i + 1, 0); // month-end
    p = Math.min(0.95, Math.max(0.01, p + (rng() - 0.32) * 0.06 + 0.012));
    history.push({
      snapshot_date: d.toISOString().slice(0, 10),
      conversion_probability: Math.round(p * 1000) / 1000,
      risk_tier: tierFor(p),
    });
  }
  const last = history[history.length - 1].conversion_probability;
  const decile = Math.max(1, Math.min(10, Math.round(last * 10) + 1));
  return {
    company: { hMy: company.hMy, display_name: company.display_name, market: company.market, industry: company.industry },
    current_rules_score: { Probability: Math.round((last * 0.85 + 0.04) * 100) / 100, Decile: decile, ScoredAt: history[history.length - 1].snapshot_date },
    history,
    model_version: MODEL_VERSION,
  };
}

// ============================================================================
// VIEW 3 — MARKETING: Cohort & Industry Analytics
// ============================================================================
export const MK_COHORTS = ['healthcare-vertical', 'growth-mode', 'tech-vertical', 'lease-expiring-12mo'];
export const MK_INDUSTRIES = ['Banking', 'Retail', 'Healthcare', 'Manufacturing', 'Technology'];
const VERTICALS = ['healthcare', 'finance-insurance', 'technology', 'retail', 'manufacturing', 'real-estate'];
const SIZE_BANDS = ['micro', 'small', 'mid', 'large', 'enterprise'];

export async function aggregateMarketing({ mode, key }) {
  await delay(550);
  const rng = seeded(`mk|${mode}|${key}`);
  const markets = ['DSM', 'OMA', 'OTHER'];
  const by_market = markets.map((Market) => {
    const n = 20 + Math.floor(rng() * 60);
    return {
      Market,
      n,
      avg_score: Math.round((rng() * 12 + 4) * 10) / 10,
      avg_win_prob: Math.round((rng() * 0.18 + 0.08) * 1000) / 1000,
    };
  });
  const decile_distribution = [];
  for (let d = 1; d <= 10; d++) {
    // bell-ish, weighted to mid deciles
    const base = Math.exp(-Math.pow(d - 4.5, 2) / 9) * 55;
    decile_distribution.push({ Decile: d, n: Math.max(3, Math.round(base + rng() * 12)) });
  }
  const cohort_win_rates = [];
  for (const Vertical of VERTICALS) {
    for (const SizeBand of SIZE_BANDS) {
      cohort_win_rates.push({
        Vertical,
        SizeBand,
        DealCount: Math.floor(rng() * 60) + 4,
        WinRate: Math.round((Math.pow(rng(), 1.3) * 0.34 + 0.02) * 1000) / 1000,
      });
    }
  }
  return { segment: key, mode, by_market, decile_distribution, cohort_win_rates, model_version: MODEL_VERSION };
}

export const SIZE_BAND_ORDER = SIZE_BANDS;
export const VERTICAL_ORDER = VERTICALS;
