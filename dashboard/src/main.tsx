import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  ArrowLeft,
  ArrowUpRight,
  Check,
  ChevronLeft,
  ChevronRight,
  Database,
  Flag,
  Hexagon,
  Layers,
  RefreshCw,
} from "lucide-react";
import "./style.css";

type Metric = Record<string, unknown>;
type Config = Record<string, unknown>;
type Review = { status: string; note: string; created_at: string };
type Episode = {
  id: string;
  seed: number;
  seat: number;
  agent_color: string;
  outcome: string;
  turns: number;
  actions: number;
  reward: number;
  seconds: number;
  flags: { code: string; reason: string }[];
  action_counts: Record<string, number>;
  end_turn_eligible: number;
  end_turn_selected: number;
  review?: Review;
  steps?: Step[];
  board?: Board;
  final_state?: State;
  reviews?: Review[];
  trace_note?: string;
};
type Run = {
  id: string;
  title: string;
  algorithm: string;
  status: string;
  source: string;
  config: Config;
  provenance: Config;
  latest: Metric;
  metrics?: Metric[];
  episodes?: Episode[];
  episode_count?: number;
  flag_count?: number;
  warnings?: string[];
  notes?: string;
};
type State = {
  turn: number;
  phase: string;
  current_color: string;
  players: {
    color: string;
    vp: number;
    resources: number;
    development_cards: number;
  }[];
  buildings: [number, string, string][];
  roads: [number[], string][];
  robber: number[];
};
type Step = {
  index: number;
  actor: string;
  action: { type: string; value: unknown };
  legal_actions: { type: string; value: unknown }[];
  state: State;
  seconds: number;
  result: unknown;
  policy_output?: { text: string; fallback: boolean; reason: string | null; action_id: number };
};
type Board = {
  tiles: {
    coordinate: number[];
    tile: { type: string; resource?: string; number?: number };
  }[];
  nodes: Record<string, { tile_coordinate: number[]; direction: string }>;
};
const percent = (v: unknown) =>
  typeof v === "number" ? `${(v * 100).toFixed(0)}%` : "Not recorded";
const num = (v: unknown, n = 2) =>
  typeof v === "number" ? v.toFixed(n) : "Not recorded";
const titleCase = (s: string) => s.toLowerCase().replaceAll("_", " ");
const colors: Record<string, string> = {
  RED: "#cb644a",
  BLUE: "#527faa",
  WHITE: "#ece5cb",
  ORANGE: "#d6984d",
};

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok)
    throw new Error(
      `Request failed (${r.status}). Check the local API and try again.`,
    );
  return r.json();
}

function Chart({
  rows,
  series,
  percentScale = false,
}: {
  rows: Metric[];
  series: { key: string; name: string; color: string }[];
  percentScale?: boolean;
}) {
  const values = rows.flatMap(
    (r) =>
      series
        .map((s) => r[s.key])
        .filter((v) => typeof v === "number") as number[],
  );
  if (!values.length)
    return (
      <div className="empty-chart">This run did not record this metric.</div>
    );
  const max = percentScale ? 1 : Math.max(...values, 0.001) * 1.08;
  const min = percentScale ? 0 : Math.min(...values, 0) * 1.08;
  const x = (i: number) => 45 + (i / Math.max(rows.length - 1, 1)) * 635;
  const y = (v: number) => 192 - ((v - min) / (max - min)) * 166;
  return (
    <>
      <svg
        viewBox="0 0 710 235"
        className="chart"
        role="img"
        aria-label={
          series.map((s) => s.name).join(" and ") + " over recorded steps"
        }
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line
              x1="45"
              x2="680"
              y1={y(min + t * (max - min))}
              y2={y(min + t * (max - min))}
              stroke="#e4e7df"
            />
            <text x="33" y={y(min + t * (max - min)) + 4} textAnchor="end">
              {percentScale ? `${Math.round(t * 100)}%` : (min + t * (max - min)).toFixed(1)}
            </text>
          </g>
        ))}
        {series.map((s) => (
          <polyline
            key={s.key}
            fill="none"
            stroke={s.color}
            strokeWidth="2.4"
            strokeLinejoin="round"
            points={rows
              .flatMap((r, i) =>
                typeof r[s.key] === "number"
                  ? [`${x(i)},${y(r[s.key] as number)}`]
                  : [],
              )
              .join(" ")}
          />
        ))}
        {series.flatMap((s) =>
          rows.flatMap((r, i) =>
            typeof r[s.key] === "number"
              ? [
                  <circle
                    key={`${s.key}-${i}`}
                    cx={x(i)}
                    cy={y(r[s.key] as number)}
                    r="2.6"
                    fill={s.color}
                  >
                    <title>
                      {s.name}: {Number(r[s.key]).toFixed(3)} at step{" "}
                      {String(r.step)}
                    </title>
                  </circle>,
                ]
              : [],
          ),
        )}
        {[0, Math.floor((rows.length - 1) / 2), rows.length - 1].map((i, j) => (
          <text key={j} x={x(i)} y="219" textAnchor="middle">
            {String(rows[i]?.step ?? i)}
          </text>
        ))}
      </svg>
      <div className="legend">
        {series.map((s) => (
          <span key={s.key}>
            <i style={{ background: s.color }} />
            {s.name}
          </span>
        ))}
        <span className="axis-label">Recorded step</span>
      </div>
    </>
  );
}

function BoardView({ board, state }: { board: Board; state: State }) {
  const center = (c: number[]) => [
    ((Math.sqrt(3) * 27) / 2) * (c[0] - c[1]) + 215,
    -40.5 * (c[0] + c[1]) + 170,
  ];
  const angles: Record<string, number> = {
    NORTH: -90,
    NORTHEAST: -30,
    SOUTHEAST: 30,
    SOUTH: 90,
    SOUTHWEST: 150,
    NORTHWEST: 210,
  };
  const point = (id: number) => {
    const n = board.nodes[id];
    if (!n) return [0, 0];
    const [x, y] = center(n.tile_coordinate);
    const a = ((angles[n.direction] ?? -90) * Math.PI) / 180;
    return [x + 27 * Math.cos(a), y + 27 * Math.sin(a)];
  };
  const terrain: Record<string, string> = {
    WOOD: "#7c9c85",
    SHEEP: "#b6c58d",
    WHEAT: "#dbc98a",
    BRICK: "#bd8975",
    ORE: "#a5ada9",
  };
  return (
    <svg
      className="board"
      viewBox="0 0 430 340"
      role="img"
      aria-label="Public board at this decision"
    >
      {board.tiles
        .filter((t) => !["WATER", "PORT"].includes(t.tile.type))
        .map((t, i) => {
          const [x, y] = center(t.coordinate);
          return (
            <g key={i}>
              <polygon
                points={Array.from({ length: 6 }, (_, k) => {
                  const a = ((k * 60 - 90) * Math.PI) / 180;
                  return `${x + 26 * Math.cos(a)},${y + 26 * Math.sin(a)}`;
                }).join(" ")}
                fill={terrain[t.tile.resource ?? ""] ?? "#d8d0b0"}
                stroke="#f7f7f0"
                strokeWidth="2"
              />
              {t.tile.number && (
                <>
                  <circle cx={x} cy={y} r="9" fill="#f6f2e6" />
                  <text
                    x={x}
                    y={y + 3}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#314137"
                  >
                    {t.tile.number}
                  </text>
                </>
              )}
              {t.coordinate.join() == state.robber.join() && (
                <circle cx={x} cy={y - 13} r="4" fill="#26352e" />
              )}
            </g>
          );
        })}
      {state.roads.map(([edge, c], i) => {
        const a = point(edge[0]),
          b = point(edge[1]);
        return (
          <line
            key={i}
            x1={a[0]}
            y1={a[1]}
            x2={b[0]}
            y2={b[1]}
            stroke={colors[c]}
            strokeWidth="4"
            strokeLinecap="round"
          />
        );
      })}
      {state.buildings.map(([id, c, b]) => {
        const [x, y] = point(id);
        return (
          <g key={id}>
            <rect
              x={x - 4}
              y={y - 4}
              width={b === "CITY" ? 10 : 8}
              height={b === "CITY" ? 10 : 8}
              fill={colors[c]}
              stroke="#33443c"
              strokeWidth="1"
            />
            <title>
              {c} {b}, node {id}
            </title>
          </g>
        );
      })}
    </svg>
  );
}

function App() {
  const [runs, setRuns] = useState<Run[]>([]),
    [selected, setSelected] = useState(
      new URLSearchParams(location.search).get("run") ?? "",
    ),
    [run, setRun] = useState<Run | null>(null);
  const [tab, setTab] = useState("overview"),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true);
  const [episode, setEpisode] = useState<Episode | null>(null),
    [cursor, setCursor] = useState(0),
    [flagged, setFlagged] = useState(false);
  const [note, setNote] = useState(""),
    [reviewStatus, setReviewStatus] = useState("reviewed"),
    [saved, setSaved] = useState(false);
  const [compare, setCompare] = useState(""),
    [other, setOther] = useState<Run | null>(null);
  async function reload() {
    try {
      const r = await api<Run[]>("/api/runs");
      setRuns(r);
      setSelected((s) => s || r[0]?.id || "");
      setError("");
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    reload();
    const timer = setInterval(reload, 10000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    let active = true;
    if (selected)
      api<Run>(`/api/runs/${selected}`)
        .then((r) => {
          if (active) setRun(r);
        })
        .catch((e) => setError(String(e)));
    return () => {
      active = false;
    };
  }, [selected, runs]);
  useEffect(() => {
    let active = true;
    setOther(null);
    if (compare)
      api<Run>(`/api/runs/${compare}`)
        .then((r) => {
          if (active) setOther(r);
        })
        .catch((e) => setError(String(e)));
    return () => {
      active = false;
    };
  }, [compare, runs]);
  async function openEpisode(id: string) {
    try {
      const e = await api<Episode>(`/api/runs/${selected}/episodes/${id}`);
      setEpisode(e);
      setCursor(0);
      setSaved(false);
      setNote("");
    } catch (e) {
      setError(String(e));
    }
  }
  async function saveReview() {
    if (!episode) return;
    try {
      await api(`/api/runs/${selected}/episodes/${episode.id}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: reviewStatus, note }),
      });
      setSaved(true);
      setEpisode(
        await api<Episode>(`/api/runs/${selected}/episodes/${episode.id}`),
      );
      reload();
    } catch (e) {
      setError(String(e));
    }
  }
  function chooseRun(id: string) {
    setSelected(id);
    setEpisode(null);
    history.replaceState(null, "", `?run=${encodeURIComponent(id)}`);
  }
  const episodes = run?.episodes ?? [],
    metrics = run?.metrics ?? [],
    legacy = run?.source === "legacy_log",
    preflight = run?.algorithm === "llm_preflight",
    llmTraining = run?.algorithm === "grpo" || run?.algorithm === "sft";
  const sampledRewards = metrics.flatMap((m) => Array.isArray(m.rewards) ? m.rewards as number[] : []);
  const rewardBins = [
    { label: "Invalid", count: sampledRewards.filter((v) => v === -2).length },
    { label: "[-1, -0.5)", count: sampledRewards.filter((v) => v >= -1 && v < -.5).length },
    { label: "[-0.5, 0)", count: sampledRewards.filter((v) => v >= -.5 && v < 0).length },
    { label: "[0, 0.5)", count: sampledRewards.filter((v) => v >= 0 && v < .5).length },
    { label: "[0.5, 1]", count: sampledRewards.filter((v) => v >= .5 && v <= 1).length },
  ];
  const evaluated = [...metrics]
    .reverse()
    .find((m) => typeof m.win_rate === "number");
  const overviewEpisodes =
    run?.algorithm === "alphazero" && evaluated
      ? episodes.filter((e) =>
          e.id.startsWith(`${String(evaluated.step).padStart(3, "0")}-`),
        )
      : episodes;
  const counts = overviewEpisodes.reduce(
    (a, e) => {
      for (const [k, v] of Object.entries(e.action_counts ?? {}))
        a[k] = (a[k] ?? 0) + v;
      return a;
    },
    {} as Record<string, number>,
  );
  const countTotal = Object.values(counts).reduce((a, b) => a + b, 0);
  const outcomes = ["win", "loss", "timeout"].map((k) => ({
    key: k,
    count: overviewEpisodes.filter((e) => e.outcome === k).length,
  }));
  const step = episode?.steps?.[cursor];
  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Catan RL home">
          <Hexagon size={28} />
          <span>
            CATAN<span className="brand-light"> / RL</span>
          </span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <nav>
          {[
            ["overview", "Run overview", Activity],
            ["episodes", "Episode review", Flag],
            ["difficulty", "Compare & difficulty", Layers],
          ].map(([id, label, Icon]) => {
            const I = Icon as typeof Activity;
            return (
              <button
                key={String(id)}
                className={tab === id ? "nav active" : "nav"}
                onClick={() => {
                  setTab(String(id));
                  setEpisode(null);
                }}
              >
                <I size={17} />
                {String(label)}
                {id === "episodes" && (
                  <span className="nav-count">
                    {episodes.filter((e) => e.flags.length).length}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-label run-label">
          RUNS <span>{runs.length}</span>
        </div>
        <div className="run-list">
          {runs.map((r) => (
            <button
              key={r.id}
              className={`run-link ${selected === r.id ? "selected" : ""}`}
              onClick={() => chooseRun(r.id)}
            >
              <span className={`status-dot ${r.status}`} />
              <span>
                {r.title}
                <small>
                  {r.algorithm} ·{" "}
                  {r.source === "legacy_log" ? "historical log" : r.status}
                </small>
              </span>
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <span className="live-dot" /> Local workspace
          <small>Local evaluation traces</small>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <span>
            Research / <strong>Observatory</strong>
          </span>
          <button
            className="quiet-button"
            onClick={reload}
            aria-label="Refresh runs"
          >
            <RefreshCw size={14} />
            Refresh
          </button>
        </header>
        <div className="content">
          <div className="page-heading">
            <div className="eyebrow">CATAN REINFORCEMENT LEARNING</div>
            <div className="heading-row">
              <h1>
                {tab === "overview"
                  ? "Run overview"
                  : tab === "episodes"
                    ? "Episode review"
                    : "Run comparison & difficulty"}
              </h1>
              <span className="outline-badge">1v1 · no player trading</span>
            </div>
            <p>
              {tab === "overview"
                ? "Follow the learning signal. Check it against actual play."
                : tab === "episodes"
                  ? "Trace suspicious outcomes back to the state and available actions."
                  : "Separate changes in the agent from changes in the evaluation."}
            </p>
          </div>
          {error && (
            <div role="alert" className="notice danger">
              {error}
              <button onClick={reload}>Retry</button>
            </div>
          )}
          {loading ? (
            <div className="empty-chart">Loading local runs…</div>
          ) : !runs.length ? (
            <div className="panel empty-chart">
              <Database />
              <h2>No runs yet</h2>
              <p>Run an evaluation or import a training log to begin.</p>
              <code>
                uv run python -m catan_rl.evaluation random weighted --games 4
                --out runs/first-eval
              </code>
            </div>
          ) : (
            run && (
              <>
                <div className="run-header">
                  <div>
                    <span className="eyebrow">SELECTED RUN</span>
                    <h2>{run.title}</h2>
                  </div>
                  <span
                    className={`pill ${run.status === "failed" ? "red" : ""}`}
                  >
                    <span className="status-dot" />
                    {run.status}
                  </span>
                </div>
                {run.warnings?.map((w) => (
                  <div className="notice" key={w}>
                    {w}
                  </div>
                ))}
                {legacy && (
                  <div className="notice">
                    <Database size={17} />
                    <span>
                      <strong>Historical log, limited evidence.</strong> These
                      scores include VP leaders at the turn cutoff. Actual wins,
                      reward distributions, and episode traces were not
                      recorded.
                    </span>
                  </div>
                )}
                {run.config.smoke === true && (
                  <div className="notice">Engineering smoke test. Small budgets and shortened games; excluded from playing-strength claims.</div>
                )}
                {tab === "overview" && llmTraining && (
                  <>
                    <div className="stats">
                      <Stat label="TRAINING LOSS" value={num(run.latest.loss)} detail="Completion tokens only" />
                      <Stat label="MEAN REWARD" value={num(run.latest.mean_reward)} detail="MCTS proxy; invalid answers score -2" />
                      <Stat label="VALID OUTPUTS" value={percent(run.latest.valid_rate)} detail="Sampled training completions" />
                      <Stat label="DEV AGREEMENT" value={percent([...metrics].reverse().find((m) => typeof m.dev_agreement === "number")?.dev_agreement)} detail="Separate source games; no checkpoint selection" />
                    </div>
                    <div className="grid-two">
                      <section className="panel">
                        <PanelTitle label="Training reward" caption="A better proxy score does not establish better play" />
                        <Chart rows={metrics} series={[{ key: "mean_reward", name: "Mean MCTS reward", color: "#52765a" }]} />
                      </section>
                      <section className="panel">
                        <PanelTitle label="Output and group health" caption="Flat groups have no relative reward signal" />
                        <Chart rows={metrics} percentScale series={[
                          { key: "valid_rate", name: "Valid outputs", color: "#52765a" },
                          { key: "flat_group_rate", name: "Equal-reward groups", color: "#bd9366" },
                          { key: "dev_agreement", name: "Dev agreement", color: "#527faa" },
                        ]} />
                      </section>
                      <section className="panel">
                        <PanelTitle label="Sampled reward distribution" caption="All recorded training completions; fixed reward scale" />
                        {sampledRewards.length ? <div className="action-bars">
                          {rewardBins.map((bin) => <div key={bin.label}>
                            <span>{bin.label}</span><div><i style={{ width: `${100 * bin.count / sampledRewards.length}%` }} /></div><b>{bin.count}</b>
                          </div>)}
                        </div> : <div className="empty-chart">No sampled rewards. Supervised training uses teacher action labels.</div>}
                      </section>
                      <section className="panel">
                        <PanelTitle label="Latest sampled actions" caption="Training prompts only; compare game behavior in the final evaluation" />
                        <div className="action-bars">
                          {Object.entries((run.latest.action_counts ?? {}) as Record<string, number>).map(([action, count]) => {
                            const total = Object.values((run.latest.action_counts ?? {}) as Record<string, number>).reduce((a, b) => a + b, 0);
                            return <div key={action}><span>{titleCase(action)}</span><div><i style={{ width: `${100 * count / total}%` }} /></div><b>{percent(count / total)}</b></div>;
                          })}
                        </div>
                        <p className="footnote">KL: {num(run.latest.kl, 4)} · clipped tokens: {percent(run.latest.clip_fraction)} · gradient norm: {num(run.latest.gradient_norm)}</p>
                      </section>
                    </div>
                  </>
                )}
                {tab === "overview" && preflight && (
                  <>
                    <div className="notice">
                      Generation preflight only. No optimizer updates or
                      playing-strength evaluation.
                    </div>
                    <div className="stats">
                      <Stat
                        label="VALID LEGAL OUTPUTS"
                        value={percent(
                          metrics.length
                            ? metrics.reduce(
                                (a, m) => a + Number(m.valid_rate ?? 0),
                                0,
                              ) / metrics.length
                            : undefined,
                        )}
                        detail="Strict action-ID parser"
                      />
                      <Stat
                        label="PROMPTS"
                        value={String(metrics.length)}
                        detail={`${run.config.group} completions per prompt`}
                      />
                      <Stat
                        label="LAST GROUP TIME"
                        value={`${num(run.latest.generation_seconds)} s`}
                        detail="Includes prompt processing"
                      />
                      <Stat
                        label="MPS DRIVER ALLOCATION"
                        value={`${num(Number(run.latest.memory_bytes) / 1e9)} GB`}
                        detail="Not peak training memory"
                      />
                    </div>
                    <section className="panel">
                      <PanelTitle
                        label="Generation measurements"
                        caption="Small sample; first group includes warm-up"
                      />
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Prompt</th>
                              <th>Input tokens</th>
                              <th>Group time</th>
                              <th>Valid outputs</th>
                              <th>Model outputs</th>
                            </tr>
                          </thead>
                          <tbody>
                            {metrics.map((m, i) => (
                              <tr key={i}>
                                <td>{String(m.step)}</td>
                                <td>{String(m.prompt_tokens)}</td>
                                <td>{num(m.generation_seconds)} s</td>
                                <td>{percent(m.valid_rate)}</td>
                                <td>
                                  <code>{JSON.stringify(m.outputs)}</code>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </section>
                  </>
                )}
                {tab === "overview" && !preflight && !llmTraining && (
                  <>
                    <div className="stats">
                      <Stat
                        label={
                          legacy
                            ? "CUTOFF SCORE VS WEIGHTED"
                            : "ACTUAL WIN RATE"
                        }
                        value={percent(
                          legacy
                            ? run.latest.legacy_weighted_score
                            : evaluated?.win_rate,
                        )}
                        detail={
                          legacy
                            ? "Not an actual win rate"
                            : `${evaluated?.games ?? 0} games${run.algorithm === "alphazero" && evaluated ? ` at iteration ${evaluated.step}` : ""}`
                        }
                      />
                      <Stat
                        label={legacy ? "POLICY LOSS" : "TIMEOUT RATE"}
                        value={
                          legacy
                            ? num(run.latest.policy_loss)
                            : percent(evaluated?.timeout_rate)
                        }
                        detail={
                          legacy
                            ? "Lower loss need not mean better play"
                            : "No winner before the cap"
                        }
                      />
                      <Stat
                        label="EPISODES TO REVIEW"
                        value={
                          legacy
                            ? "Not recorded"
                            : String(
                                episodes.filter(
                                  (e) => e.flags.length && !e.review,
                                ).length,
                              )
                        }
                        detail="Flags are leads, not proven exploits"
                      />
                      <Stat
                        label="RECORDED STEPS"
                        value={String(metrics.length)}
                        detail={`${run.algorithm} · ${run.source === "legacy_log" ? "imported log" : "measured run"}`}
                      />
                    </div>
                    <div className="grid-two">
                      <section className="panel">
                        <PanelTitle
                          label={
                            legacy
                              ? "Historical evaluation scores"
                              : "Evaluation outcomes"
                          }
                          caption={
                            legacy
                              ? "VP leader at cutoff counted as a win"
                              : "Actual wins and timeouts; no cutoff wins"
                          }
                        />
                        <Chart
                          rows={metrics}
                          percentScale
                          series={
                            legacy
                              ? [
                                  {
                                    key: "legacy_weighted_score",
                                    name: "vs weighted (legacy)",
                                    color: "#52765a",
                                  },
                                  {
                                    key: "legacy_value_score",
                                    name: "vs value (legacy)",
                                    color: "#bd9366",
                                  },
                                ]
                              : [
                                  {
                                    key: "win_rate",
                                    name: "Actual win rate",
                                    color: "#52765a",
                                  },
                                  {
                                    key: "timeout_rate",
                                    name: "Timeout rate",
                                    color: "#bd9366",
                                  },
                                  ...(run.algorithm === "llm_evaluation" ? [{
                                    key: "agreement", name: "Fixed held-out agreement", color: "#527faa",
                                  }] : []),
                                ]
                          }
                        />
                        {run.algorithm === "llm_evaluation" && <p className="footnote">
                          Held-out MCTS agreement: {percent(run.latest.agreement)}. Agreement minus actual win rate: {num(typeof run.latest.agreement_win_gap === "number" ? run.latest.agreement_win_gap * 100 : undefined, 1)} percentage points. These measure different tasks; inspect their changes across agents.
                        </p>}
                      </section>
                      <section className="panel">
                        <PanelTitle
                          label={
                            legacy
                              ? "Training losses"
                              : "Outcome reward distribution"
                          }
                          caption={
                            legacy
                              ? "Training signal, not game strength"
                              : "Latest evaluation: win +1, loss -1, timeout 0"
                          }
                        />
                        {legacy ? (
                          <Chart
                            rows={metrics}
                            series={[
                              {
                                key: "policy_loss",
                                name: "Policy loss",
                                color: "#52765a",
                              },
                              {
                                key: "value_loss",
                                name: "Value loss",
                                color: "#bd9366",
                              },
                            ]}
                          />
                        ) : (
                          <div className="outcomes">
                            {outcomes.map((o, i) => (
                              <div className="outcome" key={o.key}>
                                <div className="outcome-number">{o.count}</div>
                                <div className="bar-track">
                                  <div
                                    style={{
                                      height: `${overviewEpisodes.length ? (o.count / overviewEpisodes.length) * 100 : 0}%`,
                                      background: [
                                        "#6a8867",
                                        "#bd8877",
                                        "#ceba8b",
                                      ][i],
                                    }}
                                  />
                                </div>
                                <span>{o.key}</span>
                                <small>{[1, -1, 0][i]}</small>
                              </div>
                            ))}
                            {!overviewEpisodes.length && (
                              <p>No episode outcomes recorded.</p>
                            )}
                          </div>
                        )}
                      </section>
                    </div>
                    {!legacy &&
                      metrics.some(
                        (m) => typeof m.policy_loss === "number",
                      ) && (
                        <section className="panel">
                          <PanelTitle
                            label="Training losses"
                            caption="Proxy objectives; compare against actual evaluation outcomes"
                          />
                          <Chart
                            rows={metrics}
                            series={[
                              {
                                key: "policy_loss",
                                name: "Policy loss",
                                color: "#52765a",
                              },
                              {
                                key: "value_loss",
                                name: "Value loss",
                                color: "#bd9366",
                              },
                            ]}
                          />
                        </section>
                      )}
                    <div className="grid-two">
                      <section className="panel">
                        <PanelTitle
                          label="Action frequencies"
                          caption="Latest evaluated checkpoint; forced actions excluded"
                        />
                        {countTotal ? (
                          <div className="action-bars">
                            {Object.entries(counts)
                              .sort((a, b) => b[1] - a[1])
                              .map(([k, v]) => (
                                <div key={k}>
                                  <span>{titleCase(k)}</span>
                                  <div>
                                    <i
                                      style={{
                                        width: `${(100 * v) / countTotal}%`,
                                      }}
                                    />
                                  </div>
                                  <b>{percent(v / countTotal)}</b>
                                </div>
                              ))}
                            <p className="footnote">
                              END_TURN when available with another choice:{" "}
                              {percent(
                                overviewEpisodes.reduce(
                                  (a, e) => a + e.end_turn_selected,
                                  0,
                                ) /
                                  Math.max(
                                    1,
                                    overviewEpisodes.reduce(
                                      (a, e) => a + e.end_turn_eligible,
                                      0,
                                    ),
                                  ),
                              )}
                              . Compare within the same game phase before
                              diagnosing collapse.
                            </p>
                          </div>
                        ) : (
                          <div className="empty-chart">
                            Action decisions were not recorded.
                          </div>
                        )}
                      </section>
                      <section className="panel">
                        <PanelTitle
                          label="Evidence & provenance"
                          caption="The conditions behind the numbers"
                        />
                        <dl className="metadata">
                          <dt>Evaluation</dt>
                          <dd>
                            {String(run.config.evaluation ?? "Not recorded")}
                          </dd>
                          <dt>Opponent</dt>
                          <dd>
                            {String(
                              run.config.opponent ?? "Not recorded in manifest",
                            )}
                          </dd>
                          <dt>
                            {run.algorithm === "alphazero"
                              ? "Self-play cap"
                              : "Evaluation cap"}
                          </dt>
                          <dd>
                            {run.config.smoke === true ? "Shortened smoke test" : String(
                              (run.algorithm === "alphazero"
                                ? run.latest.turn_cap
                                : run.config.turn_cap) ?? "Not recorded",
                            )}
                          </dd>
                          <dt>Source revision</dt>
                          <dd>
                            {String(
                              run.provenance.git_revision ?? "Unknown",
                            ).slice(0, 12)}
                            {run.provenance.dirty ? " + local changes" : ""}
                          </dd>
                          <dt>Source fingerprint</dt>
                          <dd>
                            {String(
                              run.provenance.source_sha256 ?? "Unknown",
                            ).slice(0, 16)}
                          </dd>
                          <dt>MCTS agreement</dt>
                          <dd>{percent(run.latest.agreement)}</dd>
                        </dl>
                        <p className="footnote">
                          Agreement is unavailable until a run records a
                          separate held-out search evaluation. It is never
                          inferred from win rate.
                        </p>
                      </section>
                    </div>
                  </>
                )}
                {tab === "episodes" &&
                  (episode ? (
                    <section className="panel episode-detail">
                      <button
                        className="quiet-button"
                        onClick={() => setEpisode(null)}
                      >
                        <ArrowLeft size={15} />
                        All episodes
                      </button>
                      <div className="heading-row">
                        <h2>Episode {episode.id}</h2>
                        <span
                          className={`pill ${episode.outcome === "timeout" ? "amber" : ""}`}
                        >
                          {episode.outcome}
                        </span>
                      </div>
                      <p className="muted">
                        Board seed {episode.seed} · seat {episode.seat + 1} ·{" "}
                        {episode.agent_color.toLowerCase()} agent ·{" "}
                        {episode.turns} turns
                      </p>
                      {episode.flags.map((f) => (
                        <div className="notice" key={f.code}>
                          <Flag size={16} />
                          {f.reason}
                        </div>
                      ))}
                      {episode.trace_note && <div className="notice">{episode.trace_note}</div>}
                      {step && episode.board && (
                        <div className="trajectory">
                          <div className="board-panel">
                            <BoardView
                              board={episode.board}
                              state={step.state}
                            />
                            <div className="players">
                              {step.state.players.map((p) => (
                                <div key={p.color}>
                                  <i style={{ background: colors[p.color] }} />
                                  <strong>{p.color.toLowerCase()}</strong>
                                  <span>
                                    {p.vp} VP · {p.resources} resources
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                          <div className="decision">
                            <span className="eyebrow">
                              BEFORE DECISION {cursor + 1} /{" "}
                              {episode.steps?.length}
                            </span>
                            <h3>{titleCase(step.action.type)}</h3>
                            <p>
                              Turn {step.state.turn} ·{" "}
                              {step.actor.toLowerCase()} · {step.state.phase}
                            </p>
                            <div className="decision-value">
                              <span>Action value</span>
                              <code>{JSON.stringify(step.action.value)}</code>
                            </div>
                            <div className="decision-value">
                              <span>Observed result</span>
                              <code>{JSON.stringify(step.result)}</code>
                            </div>
                            <div className="decision-value">
                              <span>Decision time</span>
                              <b>{step.seconds.toFixed(3)} s</b>
                            </div>
                            {step.policy_output && <div className="decision-value">
                              <span>Language model output</span>
                              <code>{JSON.stringify(step.policy_output.text)}</code>
                              {step.policy_output.fallback && <strong>Uniform legal fallback: {step.policy_output.reason}</strong>}
                            </div>}
                            <h4>{step.legal_actions.length} legal actions</h4>
                            <div className="legal-list">
                              {step.legal_actions.map((a, i) => (
                                <div
                                  key={i}
                                  className={
                                    JSON.stringify(a) ===
                                    JSON.stringify(step.action)
                                      ? "chosen"
                                      : ""
                                  }
                                >
                                  {titleCase(a.type)}{" "}
                                  <code>{JSON.stringify(a.value)}</code>
                                </div>
                              ))}
                            </div>
                          </div>
                          <div className="scrubber">
                            <button
                              aria-label="Previous decision"
                              disabled={!cursor}
                              onClick={() => setCursor((c) => c - 1)}
                            >
                              <ChevronLeft size={18} />
                            </button>
                            <input
                              aria-label="Decision in trajectory"
                              type="range"
                              min="0"
                              max={(episode.steps?.length ?? 1) - 1}
                              value={cursor}
                              onChange={(e) => setCursor(+e.target.value)}
                            />
                            <button
                              aria-label="Next decision"
                              disabled={
                                cursor === (episode.steps?.length ?? 1) - 1
                              }
                              onClick={() => setCursor((c) => c + 1)}
                            >
                              <ChevronRight size={18} />
                            </button>
                          </div>
                        </div>
                      )}
                      <div className="review-form">
                        <h3>Review this episode</h3>
                        <p className="muted">
                          Record evidence before confirming a reward exploit.
                          Reviews do not change the training data.
                        </p>
                        <label>
                          Finding
                          <select
                            value={reviewStatus}
                            onChange={(e) => {
                              setReviewStatus(e.target.value);
                              setSaved(false);
                            }}
                          >
                            <option value="reviewed">
                              Reviewed, needs investigation
                            </option>
                            <option value="dismissed">
                              Dismissed, expected behavior
                            </option>
                            <option value="confirmed">
                              Confirmed exploit, evidence below
                            </option>
                          </select>
                        </label>
                        <label>
                          Evidence
                          <textarea
                            value={note}
                            maxLength={2000}
                            onChange={(e) => {
                              setNote(e.target.value);
                              setSaved(false);
                            }}
                            placeholder="What happened? Which decision supports your conclusion?"
                          />
                        </label>
                        <button
                          className="primary"
                          disabled={!note.trim() || saved}
                          onClick={saveReview}
                        >
                          {saved ? (
                            <>
                              <Check size={15} />
                              Saved
                            </>
                          ) : (
                            "Save review"
                          )}
                        </button>
                        {episode.reviews?.map((r, i) => (
                          <div className="review-history" key={i}>
                            <strong>{r.status}</strong>
                            <p>{r.note}</p>
                            <small>
                              {new Date(r.created_at).toLocaleString()}
                            </small>
                          </div>
                        ))}
                      </div>
                    </section>
                  ) : (
                    <section className="panel">
                      <div className="panel-title">
                        <div>
                          <h3>Evaluation episodes</h3>
                          <p>
                            Flags identify behavior to inspect. They do not
                            establish reward hacking.
                          </p>
                        </div>
                        <label className="checkbox">
                          <input
                            type="checkbox"
                            checked={flagged}
                            onChange={(e) => setFlagged(e.target.checked)}
                          />
                          Flagged only
                        </label>
                      </div>
                      {episodes.length ? (
                        <div className="table-wrap">
                          <table>
                            <thead>
                              <tr>
                                <th>Episode</th>
                                <th>Outcome</th>
                                <th>Turns</th>
                                <th>Board / seat</th>
                                <th>Review</th>
                                <th />
                              </tr>
                            </thead>
                            <tbody>
                              {episodes
                                .filter((e) => !flagged || e.flags.length)
                                .map((e) => (
                                  <tr
                                    key={e.id}
                                    onClick={() => openEpisode(e.id)}
                                  >
                                    <td>
                                      <button
                                        className="text-button"
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          openEpisode(e.id);
                                        }}
                                      >
                                        #{e.id}
                                      </button>
                                    </td>
                                    <td>
                                      <span
                                        className={`pill ${e.outcome === "timeout" ? "amber" : ""}`}
                                      >
                                        {e.outcome}
                                      </span>
                                    </td>
                                    <td>{e.turns}</td>
                                    <td>
                                      {e.seed} / {e.seat + 1}
                                    </td>
                                    <td>
                                      {e.review ? (
                                        <span className="reviewed">
                                          <Check size={13} />
                                          {e.review.status}
                                        </span>
                                      ) : e.flags.length ? (
                                        <span className="flag-label">
                                          <Flag size={13} />
                                          {e.flags
                                            .map((f) => titleCase(f.code))
                                            .join(", ")}
                                        </span>
                                      ) : (
                                        "No flags"
                                      )}
                                    </td>
                                    <td>
                                      <ArrowUpRight size={16} />
                                    </td>
                                  </tr>
                                ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div className="empty-chart">
                          No episode traces were saved for this run. Evaluate a
                          checkpoint to collect fresh traces.
                        </div>
                      )}
                    </section>
                  ))}
                {tab === "difficulty" && (
                  <>
                    <section className="panel">
                      <PanelTitle
                        label="Fixed-cohort baseline references"
                        caption="20 games per agent against weighted; board seeds 10000–10009, both seats"
                      />
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Agent</th>
                              <th>Games completed</th>
                              <th>Actual wins</th>
                              <th>Timeouts</th>
                              <th>Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {runs
                              .filter(
                                (r) =>
                                  ["random", "value"].includes(
                                    String(r.config.agent),
                                  ) &&
                                  r.config.opponent === "weighted" &&
                                  r.config.seed === 10000 &&
                                  r.config.games === 20 &&
                                  r.config.turn_cap === 400 &&
                                  r.config.action_cap === 8000 &&
                                  r.config.evaluation === "actual_outcome_v1",
                              )
                              .map((r) => (
                                <tr key={r.id}>
                                  <td>
                                    <button
                                      className="text-button"
                                      onClick={() => chooseRun(r.id)}
                                    >
                                      {String(r.config.agent)}
                                    </button>
                                  </td>
                                  <td>{String(r.latest.games ?? 0)}</td>
                                  <td>{percent(r.latest.win_rate)}</td>
                                  <td>{percent(r.latest.timeout_rate)}</td>
                                  <td>{r.status}</td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                      <p className="footnote">
                        Reference performance depends on the agent. Empty rows
                        mean the reference runs have not been collected. Small
                        cohorts do not establish universal board difficulty.
                      </p>
                    </section>
                    <section className="panel">
                      <PanelTitle
                        label="Run comparison"
                        caption="Compare recorded outcomes without blending different evaluation definitions"
                      />
                      <label className="compare-label">
                        Compare with
                        <select
                          value={compare}
                          onChange={(e) => setCompare(e.target.value)}
                        >
                          <option value="">Choose another run</option>
                          {runs
                            .filter((r) => r.id !== selected)
                            .map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.title}
                              </option>
                            ))}
                        </select>
                      </label>
                      {other && (
                        <>
                          <div className="notice">
                            {[
                              "evaluation",
                              "opponent",
                              "turn_cap",
                              "games",
                              "seed",
                              "action_cap",
                              "smoke",
                              "evaluation_games",
                              "dataset_sha256",
                            ].every((k) => JSON.stringify(run.config[k]) === JSON.stringify(other.config[k]))
                              ? "Recorded evaluation settings match. Check model and search budgets before attributing differences to learning."
                              : "Evaluation settings differ. These results are descriptive, not a controlled comparison."}
                          </div>
                          <table>
                            <thead>
                              <tr>
                                <th>Measure</th>
                                <th>{run.title}</th>
                                <th>{other.title}</th>
                              </tr>
                            </thead>
                            <tbody>
                              {[
                                ["Actual win rate", "win_rate"],
                                ["Timeout rate", "timeout_rate"],
                                ["Held-out MCTS agreement", "agreement"],
                              ].map(([name, key]) => (
                                <tr key={key}>
                                  <td>{name}</td>
                                  <td>
                                    {percent(
                                      [...metrics]
                                        .reverse()
                                        .find(
                                          (m) => typeof m[key] === "number",
                                        )?.[key],
                                    )}
                                  </td>
                                  <td>
                                    {percent(
                                      [...(other.metrics ?? [])]
                                        .reverse()
                                        .find(
                                          (m) => typeof m[key] === "number",
                                        )?.[key],
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </>
                      )}
                    </section>
                    <section className="panel">
                      <PanelTitle
                        label="Difficulty indicators"
                        caption="Game length and baseline outcomes are observations, not a calibrated difficulty score"
                      />
                      {episodes.length ? (
                        <table>
                          <thead>
                            <tr>
                              <th>Observed length</th>
                              <th>Games</th>
                              <th>Actual wins</th>
                              <th>Timeouts</th>
                            </tr>
                          </thead>
                          <tbody>
                            {[
                              [0, 99],
                              [100, 199],
                              [200, 399],
                              [400, 9999],
                            ].map(([lo, hi]) => {
                              const group = episodes.filter(
                                (e) => e.turns >= lo && e.turns <= hi,
                              );
                              return (
                                <tr key={lo}>
                                  <td>
                                    {lo}–{hi === 9999 ? "end" : hi} turns
                                  </td>
                                  <td>{group.length}</td>
                                  <td>
                                    {group.length
                                      ? percent(
                                          group.filter(
                                            (e) => e.outcome === "win",
                                          ).length / group.length,
                                        )
                                      : "No samples"}
                                  </td>
                                  <td>
                                    {group.length
                                      ? percent(
                                          group.filter(
                                            (e) => e.outcome === "timeout",
                                          ).length / group.length,
                                        )
                                      : "No samples"}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      ) : (
                        <div className="empty-chart">
                          Difficulty breakdowns require recorded episodes.
                        </div>
                      )}
                      <p className="footnote">
                        These groups are defined after play and can reflect
                        agent weakness as well as environment difficulty. Fixed
                        board cohorts and independent baseline evaluations are
                        needed for calibration.
                      </p>
                    </section>
                  </>
                )}
                <footer>
                  <span>CATAN / RL</span>
                  <span>
                    Search scores are estimates. Game outcomes are measured
                    separately.
                  </span>
                </footer>
              </>
            )
          )}
        </div>
      </main>
    </div>
  );
}
function PanelTitle({ label, caption }: { label: string; caption: string }) {
  return (
    <div className="panel-title">
      <div>
        <h3>{label}</h3>
        <p>{caption}</p>
      </div>
    </div>
  );
}
function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="stat">
      <span className="eyebrow">{label}</span>
      <div
        className={
          value === "Not recorded" ? "stat-value missing" : "stat-value"
        }
      >
        {value}
      </div>
      <p>{detail}</p>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
