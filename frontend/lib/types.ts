export interface Company {
  id: number;
  ticker: string;
  name: string;
  cik: string | null;
  sector: string | null;
  industry: string | null;
  sic_code: string | null;
  listing_status: string;
}

export interface LatestScore {
  value: number | null;
  status: "complete" | "provisional" | "unavailable";
  as_of: string;
  change_7d: number | null;
  model_version: string | null;
}

export type Direction = "a_advantaged" | "b_advantaged" | "neutral" | "uncertain" | "not_applicable";

export interface Pair {
  id: number;
  uid: string;
  slug: string;
  company_a: Company;
  company_b: Company;
  industry: string | null;
  status: "active" | "disabled" | "archived";
  monitoring_interval_minutes: number;
  alert_threshold: number;
  alert_change_threshold: number;
  notes: string | null;
  created_at: string;
  archived_at: string | null;
  watchlist_ids: number[];
  si_v1: LatestScore | null;
  si_v2: LatestScore | null;
  direction: Direction | null;
  confidence: number | null;
  explanation: string | null;
  last_update: string | null;
  is_demo: boolean;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface Component {
  component: string;
  value: number | null;
  available: boolean;
  weight: number;
  explanation: string;
  inputs: Record<string, unknown>;
  sources: unknown[];
}

export interface Score {
  id: number;
  index_name: "si_v1" | "si_v2";
  model_version: string;
  as_of: string;
  computed_at: string;
  value: number | null;
  status: string;
  coverage: number;
  direction: Direction;
  confidence: number | null;
  explanation: string;
  trigger: string;
  supporting_event_ids: number[];
  is_demo: boolean;
  components: Component[] | null;
  confidence_breakdown: { parts?: Record<string, number | null>; weights?: Record<string, number>; note?: string; baselines?: Record<string, number | null> } | null;
  direction_detail: Record<string, unknown> | null;
}

export interface Evidence {
  quote: string;
  supports: string;
  verified: boolean;
}

export interface IndexEffect {
  score_id: number;
  score_as_of: string;
  n_weight: number | null;
  si_v2_before: number | null;
  si_v2_after: number | null;
  si_v2_delta: number | null;
  counted: boolean;
}

export interface NewsEvent {
  id: number;
  pair_id: number;
  pair_slug: string;
  is_material: boolean;
  event_type: string;
  affected_tickers: string[];
  scope: string;
  relative_direction: string;
  severity: number;
  asymmetry: number;
  expected_persistence_days: number;
  extraction_confidence: number;
  source_reliability: number;
  is_new_information: boolean;
  thread_key: string;
  summary: string;
  company_exposure: Record<string, { exposure: string; effect: string; rationale: string }>;
  evidence: Evidence[];
  evidence_verified_ratio: number;
  uncertainties: string[];
  model_name: string;
  prompt_version: string;
  analyzed_at: string;
  changed_index: boolean;
  index_effect: IndexEffect | null;
}

export interface SourceDoc {
  id: number;
  provider: string;
  doc_type: string;
  url: string;
  title: string;
  publisher: string | null;
  published_at: string;
  ingested_at: string;
  source_reliability: number;
  filing_form: string | null;
  filing_items: string[];
}

export interface Article {
  id: number;
  headline: string;
  first_published_at: string;
  first_ingested_at: string;
  document_count: number;
  distinct_publishers: number;
  tickers: string[];
  analysis_status: string;
  analysis_error: string | null;
  is_demo: boolean;
  documents: SourceDoc[];
  events: NewsEvent[];
}

export interface Outcome {
  end_date: string;
  divergence: number;
  threshold: number;
  diverged: boolean;
  log_return_a: number;
  log_return_b: number;
  paper_trades: Record<string, any>;
  evaluated_at: string;
  is_total_return: boolean;
}

export interface Prediction {
  id: number;
  prediction_date: string;
  data_cutoff: string;
  horizon_days: number;
  maturity_date: string;
  target_version: string;
  target_definition: { threshold: number; measure: string };
  si_v1: number | null;
  si_v2: number | null;
  si_v2_status: string;
  si_v2_model_version: string | null;
  weights_version: string | null;
  confidence: number | null;
  direction: Direction;
  baselines: Record<string, number | null>;
  base_date: string;
  price_provider: string;
  is_total_return: boolean;
  is_demo: boolean;
  status: "pending" | "resolved";
  outcome: Outcome | null;
  error: number | null;
}

export interface JobRun {
  id: number;
  job_name: string;
  trigger: string;
  status: string;
  params: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  attempts: number;
  error: string | null;
  result: Record<string, unknown>;
}

export interface ScheduledJob {
  name: string;
  description: string;
  cron: string;
  timezone: string;
  trading_days_only: boolean;
  enabled: boolean;
  max_attempts: number;
  next_run: string | null;
  last_run: JobRun | null;
}

export interface SystemStatus {
  mode: "demo" | "production";
  version: string;
  now: string;
  pairs: { monitored: number; high_risk: number; avg_si_v2: number | null; avg_si_v1: number | null; scored_si_v2: number };
  significant_events_7d: number;
  bot: { state: string; paused: boolean; last_activity: string | null; running_jobs: string[]; queued: number };
  last_news_update: string | null;
  last_price_update: string | null;
  news_age_hours: number | null;
  price_age_hours: number | null;
  ai: { model: string; effort: string; month_cost_usd: number; monthly_limit_usd: number; input_tokens: number; output_tokens: number; calls: number; configured: boolean };
  providers: Record<string, boolean>;
  pending_analysis: number;
  open_alerts: number;
  demo_data_present?: boolean;
}

export interface Watchlist {
  id: number;
  name: string;
  description: string | null;
  is_default: boolean;
  pair_ids: number[];
}

export interface User {
  id: number;
  email: string;
  role: "admin" | "viewer";
}
