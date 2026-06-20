import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  ComposedChart, AreaChart, BarChart, LineChart,
  Area, Line, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

// ═══════════════════════════════════════════════════════════════════
//  API CONFIG  ←  point API_BASE at your deployed host
//  Each function below is a separate serverless endpoint / FastAPI route
// ═══════════════════════════════════════════════════════════════════
const API_BASE = "https://api.yourfund.io";

const EP = {
  summary:     `${API_BASE}/api/portfolio/summary`,   // GET
  history:     `${API_BASE}/api/portfolio/history`,   // GET
  positions:   `${API_BASE}/api/positions`,           // GET
  trades:      `${API_BASE}/api/trades`,              // GET
  risk:        `${API_BASE}/api/risk`,                // GET
  signals:     `${API_BASE}/api/signals`,             // GET
  attribution: `${API_BASE}/api/attribution`,         // GET
  backtest:    `${API_BASE}/api/backtest`,            // POST { config }
};

// ═══════════════════════════════════════════════════════════════════
//  DESIGN TOKENS
// ═══════════════════════════════════════════════════════════════════
const C = {
  bg: "#07090F", panel: "#0D1117", card: "#111827", card2: "#161F2E",
  border: "#1E2A3B", teal: "#00D4AA", tealDim: "#00D4AA18",
  red: "#FF4D6D", redDim: "#FF4D6D18",
  amber: "#F4B942", blue: "#58A6FF", violet: "#A78BFA",
  muted: "#4E617A", text: "#8FA3BC", bright: "#CDD9E5", white: "#E8F0F8",
};

// ═══════════════════════════════════════════════════════════════════
//  MOCK DATA
// ═══════════════════════════════════════════════════════════════════
function mkRng(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
}

const MOCK_HISTORY = (() => {
  const rng = mkRng(42); const rows = [];
  let nav = 10e6, bench = 10e6;
  for (let i = 0; i < 365; i++) {
    const d = new Date(2024, 0, 2); d.setDate(d.getDate() + i);
    if (d.getDay() === 0 || d.getDay() === 6) continue;
    nav   *= 1 + (rng() - 0.468) * 0.018;
    bench *= 1 + (rng() - 0.472) * 0.015;
    rows.push({ date: d.toISOString().slice(0, 10), nav: Math.round(nav), bench: Math.round(bench) });
  }
  return rows;
})();

const LAST_BAR = MOCK_HISTORY.at(-1), PREV_BAR = MOCK_HISTORY.at(-2);

const MOCK_SUMMARY = {
  strategy: "Cross-Sectional Momentum + Low-Vol",
  nav: LAST_BAR.nav, daily_pnl: LAST_BAR.nav - PREV_BAR.nav,
  daily_pnl_pct: (LAST_BAR.nav - PREV_BAR.nav) / PREV_BAR.nav,
  total_return: (LAST_BAR.nav - 10e6) / 10e6,
  cagr: 0.1186, cash: 1_234_567, gross_leverage: 1.48, net_leverage: 0.33,
  n_positions: 14, sharpe: 0.87, sortino: 1.24, calmar: 1.43, max_drawdown: -0.0827,
  as_of: "2024-12-27 16:00 UTC",
};

const MOCK_POSITIONS = [
  { symbol:"NVDA",side:"LONG", qty:1480,avg_cost:842.12,last_price:881.50,market_value: 1304620,weight: 0.112,pnl: 58298,pnl_pct: 0.0468,sector:"Technology"},
  { symbol:"AAPL",side:"LONG", qty:2847,avg_cost:182.34,last_price:189.12,market_value:  538554,weight: 0.047,pnl: 19308,pnl_pct: 0.0372,sector:"Technology"},
  { symbol:"MSFT",side:"LONG", qty:1230,avg_cost:378.44,last_price:396.70,market_value:  487941,weight: 0.042,pnl: 22460,pnl_pct: 0.0482,sector:"Technology"},
  { symbol:"COST",side:"LONG", qty: 620,avg_cost:689.20,last_price:724.80,market_value:  449376,weight: 0.039,pnl: 22072,pnl_pct: 0.0516,sector:"Consumer"},
  { symbol:"LLY", side:"LONG", qty: 340,avg_cost:712.50,last_price:762.30,market_value:  259182,weight: 0.022,pnl: 16932,pnl_pct: 0.0699,sector:"Healthcare"},
  { symbol:"META",side:"LONG", qty: 480,avg_cost:512.30,last_price:484.10,market_value:  232368,weight: 0.020,pnl:-13536,pnl_pct:-0.0551,sector:"Technology"},
  { symbol:"AMZN",side:"LONG", qty: 920,avg_cost:184.22,last_price:191.80,market_value:  176456,weight: 0.015,pnl:  6974,pnl_pct: 0.0411,sector:"Consumer"},
  { symbol:"JPM", side:"LONG", qty: 780,avg_cost:196.80,last_price:204.50,market_value:  159510,weight: 0.014,pnl:  6006,pnl_pct: 0.0393,sector:"Financials"},
  { symbol:"TSLA",side:"SHORT",qty:1200,avg_cost:248.40,last_price:261.30,market_value: -313560,weight:-0.027,pnl:-15480,pnl_pct:-0.0519,sector:"Automotive"},
  { symbol:"XOM", side:"SHORT",qty: 640,avg_cost:107.20,last_price:104.80,market_value:  -67072,weight:-0.006,pnl:  1536,pnl_pct: 0.0224,sector:"Energy"},
  { symbol:"CVX", side:"SHORT",qty: 580,avg_cost:156.80,last_price:153.40,market_value:  -88972,weight:-0.008,pnl:  1972,pnl_pct: 0.0217,sector:"Energy"},
  { symbol:"PFE", side:"SHORT",qty:3200,avg_cost: 28.40,last_price: 27.80,market_value:  -88960,weight:-0.008,pnl:  1920,pnl_pct: 0.0211,sector:"Healthcare"},
  { symbol:"BAC", side:"SHORT",qty:2100,avg_cost: 37.80,last_price: 36.90,market_value:  -77490,weight:-0.007,pnl:  1890,pnl_pct: 0.0238,sector:"Financials"},
  { symbol:"DIS", side:"SHORT",qty: 820,avg_cost: 92.40,last_price: 91.20,market_value:  -74784,weight:-0.007,pnl:   984,pnl_pct: 0.0130,sector:"Consumer"},
];

const MOCK_RISK = {
  position_max_pct:0.072, position_limit:0.10,
  gross_leverage:1.48,    gross_limit:2.00,
  net_leverage:0.33,      net_limit:1.00,
  current_dd:-0.041,      dd_limit:0.15,
  var_99_1d:0.0148,       var_limit:0.02,
  daily_loss_pct:0.0077,  daily_limit:0.03,
  portfolio_beta:0.84, portfolio_vol:0.112, tracking_error:0.071, sharpe:0.87,
};

const SYM_LIST = ["NVDA","AAPL","MSFT","COST","LLY","META","AMZN","JPM","TSLA","XOM","CVX","PFE","BAC","DIS"];
const ALPHA_MODELS = ["Momentum","MeanRev","Trend","LowVol"];

const MOCK_SIGNALS = (() => {
  const rng = mkRng(7);
  return SYM_LIST.map(sym => {
    const s = {}; ALPHA_MODELS.forEach(m => { s[m] = Math.round((rng()*2-1)*100)/100; });
    s.composite = Math.round(ALPHA_MODELS.reduce((a,m)=>a+s[m],0)/ALPHA_MODELS.length*100)/100;
    return { symbol: sym, ...s };
  }).sort((a,b) => b.composite - a.composite);
})();

const MOCK_TRADES = (() => {
  const base = new Date("2024-12-27T09:31:00Z");
  return [
    ["NVDA","BUY",120,878.40,"MomentumAlpha"],["COST","BUY",90,722.10,"LowVolAlpha"],
    ["META","SELL",240,487.30,"MomentumAlpha"],["TSLA","SELL",400,259.80,"MomentumAlpha"],
    ["AAPL","BUY",350,188.20,"TrendAlpha"],["LLY","BUY",60,760.10,"LowVolAlpha"],
    ["XOM","SELL",320,104.20,"MomentumAlpha"],["MSFT","BUY",180,394.10,"TrendAlpha"],
  ].map(([sym,side,qty,px,model],i) => ({
    time: new Date(base.getTime()+i*1740000).toISOString().slice(11,19),
    symbol:sym, side, qty, price:px, notional:Math.round(qty*px),
    commission:Math.round(qty*0.005*100)/100,
    slippage:Math.round(qty*px*0.0005*100)/100, model,
  }));
})();

const MOCK_MONTHLY = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
  .map((m,i) => { const rng = mkRng(i*31+17); return { month:m, ret:Math.round((rng()*9-4)*10)/10 }; });

// Attribution mock data
const SECTOR_COLORS = { Technology:C.teal, Consumer:C.blue, Healthcare:C.violet, Financials:C.amber, Energy:C.red, Automotive:"#EC4899" };
const MOCK_ATTRIBUTION = (() => {
  const bySector = {}; const bySymbol = [];
  MOCK_POSITIONS.forEach(p => {
    bySector[p.sector] = (bySector[p.sector] || 0) + p.pnl;
    bySymbol.push({ symbol: p.symbol, pnl: p.pnl, sector: p.sector });
  });
  return {
    by_symbol: bySymbol.sort((a,b) => b.pnl - a.pnl),
    by_sector: Object.entries(bySector).map(([sector, pnl]) => ({ sector, pnl })).sort((a,b) => b.pnl - a.pnl),
    by_model: [
      { model:"MomentumAlpha", pnl:82400, trades:312, win_rate:0.56 },
      { model:"TrendFollowing", pnl:31200, trades:198, win_rate:0.52 },
      { model:"LowVolatility",  pnl:21800, trades:147, win_rate:0.58 },
      { model:"MeanReversion",  pnl:-8300, trades:190, win_rate:0.44 },
    ],
    rolling: MOCK_HISTORY.slice(-60).map((h,i) => ({
      date: h.date,
      momentum: Math.round((mkRng(i*3+1)()*2-1)*8000),
      trend:    Math.round((mkRng(i*7+2)()*2-1)*4000),
      lowvol:   Math.round((mkRng(i*11+3)()*2-1)*3000),
    })),
  };
})();

// ═══════════════════════════════════════════════════════════════════
//  useFetch  — real API call, mock fallback on error
// ═══════════════════════════════════════════════════════════════════
function useFetch(url, mock) {
  const [data, setData] = useState(mock);
  const [apiOk, setApiOk] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error();
      setData(await r.json()); setApiOk(true);
    } catch { setApiOk(false); }
  }, [url]);

  useEffect(() => { refresh(); }, [refresh]);
  return { data, apiOk, refresh };
}

// ═══════════════════════════════════════════════════════════════════
//  Formatters
// ═══════════════════════════════════════════════════════════════════
const fmtD  = v => "$" + Math.abs(v).toLocaleString("en-US", { minimumFractionDigits:0 });
const pct   = (v,d=2) => (v*100).toFixed(d)+"%";
const sgn   = (v,d=2) => (v>=0?"+":"")+(v*100).toFixed(d)+"%";
const n2    = v => v.toFixed(2);
const pcol  = v => v>=0 ? C.teal : C.red;

// ═══════════════════════════════════════════════════════════════════
//  Shared micro-components
// ═══════════════════════════════════════════════════════════════════
const Card = ({children,style}) => (
  <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:8,...style}}>{children}</div>
);
const SecHead = ({title,hint}) => (
  <div style={{marginBottom:16}}>
    <div style={{color:C.white,fontSize:14,fontWeight:700}}>{title}</div>
    {hint && <div style={{color:C.muted,fontSize:11,marginTop:3}}>{hint}</div>}
  </div>
);
const Chip = ({label,color}) => (
  <span style={{display:"inline-block",padding:"2px 8px",borderRadius:4,fontSize:10,fontWeight:800,
    letterSpacing:"0.08em",background:color+"20",color,border:`1px solid ${color}40`}}>{label}</span>
);
const KPI = ({label,value,sub,vc}) => (
  <Card style={{padding:"18px 22px"}}>
    <div style={{color:C.muted,fontSize:10,fontWeight:700,letterSpacing:"0.14em",textTransform:"uppercase",marginBottom:8}}>{label}</div>
    <div style={{color:vc||C.white,fontSize:23,fontFamily:"monospace",fontWeight:800,letterSpacing:"-0.03em",lineHeight:1}}>{value}</div>
    {sub && <div style={{color:C.muted,fontSize:11,fontFamily:"monospace",marginTop:6}}>{sub}</div>}
  </Card>
);
const LimitBar = ({label,current,limit,isX=false}) => {
  const ratio = Math.min(Math.abs(current)/Math.abs(limit),1);
  const fill  = ratio>0.85?C.red:ratio>0.65?C.amber:C.teal;
  const fmt   = v => isX ? n2(v)+"×" : pct(v);
  return (
    <div style={{marginBottom:16}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{color:C.text,fontSize:13}}>{label}</span>
        <span style={{fontFamily:"monospace",fontSize:13,fontWeight:700,color:fill}}>
          {fmt(current)} <span style={{color:C.muted,fontWeight:400}}>/ {fmt(limit)}</span>
        </span>
      </div>
      <div style={{height:5,background:C.border,borderRadius:3}}>
        <div style={{width:`${ratio*100}%`,height:"100%",background:fill,borderRadius:3,transition:"width .5s ease"}}/>
      </div>
    </div>
  );
};
const Tip = ({active,payload,label,fv}) => {
  if (!active||!payload?.length) return null;
  return (
    <div style={{background:C.card2,border:`1px solid ${C.border}`,borderRadius:6,padding:"10px 14px",fontSize:11}}>
      <div style={{color:C.muted,marginBottom:6}}>{label}</div>
      {payload.map(p=>(
        <div key={p.dataKey} style={{color:p.color||C.text,fontFamily:"monospace"}}>
          {p.name}: {fv ? fv(p.value) : p.value}
        </div>
      ))}
    </div>
  );
};

// ═══════════════════════════════════════════════════════════════════
//  OVERVIEW PANEL
// ═══════════════════════════════════════════════════════════════════
function OverviewPanel({summary,history}) {
  const ddSeries = useMemo(() => {
    let hwm=0;
    return history.map(h=>{ hwm=Math.max(hwm,h.nav); return {date:h.date,dd:parseFloat(((h.nav-hwm)/hwm*100).toFixed(2))}; });
  },[history]);

  const sectorPnl = useMemo(() => {
    const m={}; MOCK_POSITIONS.forEach(p=>{ m[p.sector]=(m[p.sector]||0)+p.pnl; });
    return Object.entries(m).map(([sector,pnl])=>({sector,pnl})).sort((a,b)=>b.pnl-a.pnl);
  },[]);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:18}}>
      {/* KPIs */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:14}}>
        <KPI label="Net Asset Value"  value={fmtD(summary.nav)}       sub={`Total return ${sgn(summary.total_return)}`}                       vc={C.white} />
        <KPI label="Daily P&L"        value={(summary.daily_pnl>=0?"+":"−")+fmtD(Math.abs(summary.daily_pnl))} sub={sgn(summary.daily_pnl_pct)+" today"} vc={pcol(summary.daily_pnl)} />
        <KPI label="Sharpe Ratio"     value={n2(summary.sharpe)}       sub={`Sortino ${n2(summary.sortino)}`}  vc={summary.sharpe>1?C.teal:summary.sharpe>0.6?C.amber:C.red} />
        <KPI label="Max Drawdown"     value={pct(summary.max_drawdown)} sub={`Calmar ${n2(summary.calmar)}`}   vc={Math.abs(summary.max_drawdown)<0.10?C.teal:Math.abs(summary.max_drawdown)<0.15?C.amber:C.red} />
      </div>

      {/* NAV chart */}
      <Card style={{padding:"22px 24px"}}>
        <SecHead title="Portfolio NAV vs Benchmark" hint={`${history[0]?.date}  →  ${history.at(-1)?.date}  ·  ${history.length} trading days`} />
        <ResponsiveContainer width="100%" height={230}>
          <ComposedChart data={history} margin={{left:8,right:12,top:4,bottom:0}}>
            <defs>
              <linearGradient id="ng" x1="0" y1="0" x2="0" y2="1">
                <stop offset="4%"  stopColor={C.teal}  stopOpacity={0.28}/>
                <stop offset="96%" stopColor={C.teal}  stopOpacity={0.01}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" stroke={C.border}/>
            <XAxis dataKey="date" tick={{fill:C.muted,fontSize:10}} tickFormatter={d=>d.slice(5)} interval={24}/>
            <YAxis tick={{fill:C.muted,fontSize:10}} tickFormatter={v=>"$"+(v/1e6).toFixed(1)+"M"} domain={["auto","auto"]}/>
            <Tooltip content={<Tip fv={v=>fmtD(v)}/>}/>
            <Area dataKey="nav"   type="monotone" stroke={C.teal}  strokeWidth={2}   fill="url(#ng)" name="NAV"/>
            <Line dataKey="bench" type="monotone" stroke={C.amber} strokeWidth={1.5} dot={false} strokeDasharray="5 4" name="Benchmark"/>
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      {/* 3-col bottom row */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:16}}>
        {/* Drawdown */}
        <Card style={{padding:"20px 22px"}}>
          <SecHead title="Drawdown" />
          <ResponsiveContainer width="100%" height={160}>
            <AreaChart data={ddSeries} margin={{left:4,right:4}}>
              <defs>
                <linearGradient id="ddg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.red} stopOpacity={0.28}/>
                  <stop offset="95%" stopColor={C.red} stopOpacity={0.02}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke={C.border}/>
              <XAxis dataKey="date" tick={{fill:C.muted,fontSize:9}} tickFormatter={d=>d.slice(5)} interval={24}/>
              <YAxis tick={{fill:C.muted,fontSize:9}} tickFormatter={v=>v.toFixed(1)+"%"}/>
              <Tooltip content={<Tip fv={v=>v.toFixed(2)+"%"}/>}/>
              <ReferenceLine y={-15} stroke={C.red} strokeDasharray="4 2" strokeOpacity={0.4}/>
              <Area dataKey="dd" type="monotone" stroke={C.red} strokeWidth={1.5} fill="url(#ddg)" name="Drawdown"/>
            </AreaChart>
          </ResponsiveContainer>
        </Card>

        {/* Monthly returns */}
        <Card style={{padding:"20px 22px"}}>
          <SecHead title="Monthly Returns"/>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={MOCK_MONTHLY} margin={{left:4,right:4}}>
              <CartesianGrid strokeDasharray="2 4" stroke={C.border} vertical={false}/>
              <XAxis dataKey="month" tick={{fill:C.muted,fontSize:9}}/>
              <YAxis tick={{fill:C.muted,fontSize:9}} tickFormatter={v=>v+"%"}/>
              <Tooltip content={<Tip fv={v=>v.toFixed(2)+"%"}/>}/>
              <ReferenceLine y={0} stroke={C.border}/>
              <Bar dataKey="ret" radius={[3,3,0,0]} name="Return">
                {MOCK_MONTHLY.map((e,i)=><Cell key={i} fill={e.ret>=0?C.teal:C.red} fillOpacity={0.85}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Sector P&L */}
        <Card style={{padding:"20px 22px"}}>
          <SecHead title="Sector P&L"/>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={sectorPnl} layout="vertical" margin={{left:4,right:30}}>
              <CartesianGrid strokeDasharray="2 4" stroke={C.border} horizontal={false}/>
              <XAxis type="number" tick={{fill:C.muted,fontSize:9}} tickFormatter={v=>"$"+(v/1000).toFixed(0)+"K"}/>
              <YAxis type="category" dataKey="sector" tick={{fill:C.text,fontSize:10}} width={70}/>
              <Tooltip content={<Tip fv={v=>fmtD(v)}/>}/>
              <ReferenceLine x={0} stroke={C.border}/>
              <Bar dataKey="pnl" radius={[0,4,4,0]} name="P&L">
                {sectorPnl.map((e,i)=><Cell key={i} fill={SECTOR_COLORS[e.sector]||C.blue} fillOpacity={0.85}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════
//  POSITIONS PANEL
// ═══════════════════════════════════════════════════════════════════
function PositionsPanel({positions}) {
  const [sortK, setSortK] = useState("market_value");
  const [sortD, setSortD] = useState(-1);
  const rows = useMemo(() => [...positions].sort((a,b)=>(Math.abs(b[sortK])-Math.abs(a[sortK]))*sortD), [positions,sortK,sortD]);
  const tog = k => { if(k===sortK) setSortD(d=>-d); else { setSortK(k); setSortD(-1); } };

  const totalPnl = positions.reduce((s,p)=>s+p.pnl,0);
  const longMV   = positions.filter(p=>p.side==="LONG").reduce((s,p)=>s+p.market_value,0);
  const shortMV  = positions.filter(p=>p.side==="SHORT").reduce((s,p)=>s+Math.abs(p.market_value),0);

  const TH=({label,k,al="right"})=>(
    <th onClick={()=>tog(k)} style={{color:k===sortK?C.teal:C.muted,fontSize:10,fontWeight:700,
      letterSpacing:"0.10em",textTransform:"uppercase",padding:"10px 14px",textAlign:al,
      borderBottom:`1px solid ${C.border}`,cursor:"pointer",userSelect:"none",whiteSpace:"nowrap"}}>
      {label}{k===sortK?(sortD>0?" ↑":" ↓"):""}
    </th>
  );

  return (
    <div style={{display:"flex",flexDirection:"column",gap:14}}>
      <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12}}>
        {[
          {l:"Long Positions", v:positions.filter(p=>p.side==="LONG").length,  c:C.teal},
          {l:"Short Positions",v:positions.filter(p=>p.side==="SHORT").length, c:C.red},
          {l:"Long Exposure",  v:fmtD(longMV),                                 c:C.white},
          {l:"Short Exposure", v:fmtD(shortMV),                                c:C.white},
          {l:"Unrealised P&L", v:(totalPnl>=0?"+":"−")+fmtD(Math.abs(totalPnl)), c:pcol(totalPnl)},
        ].map(({l,v,c})=>(
          <Card key={l} style={{padding:"14px 18px"}}>
            <div style={{color:C.muted,fontSize:10,fontWeight:700,letterSpacing:"0.10em",textTransform:"uppercase",marginBottom:6}}>{l}</div>
            <div style={{color:c,fontSize:18,fontFamily:"monospace",fontWeight:800}}>{v}</div>
          </Card>
        ))}
      </div>

      <Card>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse