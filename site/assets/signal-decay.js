// Copied unchanged from prototypes/1_signal_decay/index.html (same model as the Python notebook).
// ---------- model (identical to the Python notebook) ----------
const EDGE = 0.0476, EVENTS = 4, OI_REV = 9.5e6, BAND = [100e3, 900e3], KMAX = 100;
const DECAY = [
  {name:"Slow",        f:k=>1/(1+0.1*(k-1)),     v:"--s1"},
  {name:"Fixed pie",   f:k=>1/k,                 v:"--s2"},
  {name:"Competition", f:k=>4/((k+1)**2),        v:"--s3"},
  {name:"Fast",        f:k=>Math.exp(-0.3*(k-1)),v:"--s4"},
];
const ks = Array.from({length:KMAX},(_,i)=>i+1);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = x => x>=1e6 ? "$"+(x/1e6).toFixed(1)+"M" : "$"+Math.round(x/1e3)+"k";
const $ = id => document.getElementById(id);
const BASE = {cap:50, proc:250, obs:500, nr:150};

function params(){ return {cap:+$("cap").value*1e6, proc:+$("proc").value*1e3, obs:+$("obs").value*1e3, nr:+$("nr").value*1e3}; }
const fundProfit = (k,d,p) => EDGE*d.f(k)*p.cap*EVENTS;
const price = (k,d,p) => Math.max(fundProfit(k,d,p)-p.proc, 0);
const revenue = (k,d,p) => k*price(k,d,p);

function analyse(d,p){
  let bi=0, br=-1; ks.forEach((k,i)=>{const r=revenue(k,d,p); if(r>br){br=r;bi=i;}});
  const kc = ks.find(k=>price(k,d,p)<=p.obs);
  return {bestK:ks[bi], bestP:price(ks[bi],d,p), bestR:br, kObs:kc, rObs:kc?revenue(kc,d,p):null};
}

// ---------- charts ----------
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.font.size = 11.5;
Chart.defaults.animation = false;
const axisMoney = {ticks:{callback:v=>v===0?"$0":fmt(v)}};
function baseOpts(ytitle, ymoney, extra={}){
  const ink2=css("--ink2"), line=css("--line");
  return Object.assign({responsive:true, maintainAspectRatio:false, interaction:{mode:"index",intersect:false},
    plugins:{legend:{display:false}, tooltip:{callbacks:{title:it=>"k = "+it[0].label+" funds",
      label:c=>" "+c.dataset.label+": "+(ymoney?fmt(c.parsed.y):c.parsed.y.toFixed(2)+"%")}}},
    scales:{x:{title:{display:true,text:"Number of funds buying the signal (k)",color:ink2},ticks:{color:ink2,maxTicksLimit:11},grid:{display:false},border:{color:line}},
            y:Object.assign({title:{display:true,text:ytitle,color:ink2},ticks:{color:ink2},grid:{color:line},border:{display:false},beginAtZero:true}, ymoney?axisMoney:{})}}, extra);
}
const line = (label,data,color,o={}) => Object.assign({label,data,borderColor:color,backgroundColor:color,borderWidth:2,pointRadius:0,pointHoverRadius:4,tension:0},o);

let c1,c2,c3,c4;
function draw(){
  const p = params(), cols = DECAY.map(d=>css(d.v)), ink2=css("--ink2");
  $("v-cap").textContent = fmt(p.cap); $("v-proc").textContent = fmt(p.proc);
  $("v-obs").textContent = fmt(p.obs); $("v-nr").textContent = fmt(p.nr);
  const A = DECAY.map(d=>analyse(d,p));

  // table
  $("tbl").querySelector("tbody").innerHTML = DECAY.map((d,i)=>`<tr><td><span class="sw" style="background:${cols[i]}"></span>${d.name}</td>
    <td class="num">${A[i].bestK}</td><td class="num">${fmt(A[i].bestP)}</td><td class="num">${fmt(A[i].bestR)}</td>
    <td class="num">${A[i].kObs ?? ">"+KMAX}</td><td class="num">${A[i].rObs!=null?fmt(A[i].rObs):"—"}</td></tr>`).join("");

  // verdict
  const fits = A.map((a,i)=>({a,d:DECAY[i]})).filter(x=>x.a.kObs);
  const ceil = Math.max(...A.map(a=>a.bestR));
  const fitRev = fits.map(x=>x.a.rObs);
  let v = `With these inputs, one exclusive fund earns <strong>${fmt(fundProfit(1,{f:()=>1},p))}</strong> a year from the signal. `;
  if (fits.length){
    v += `At the real price (${fmt(p.obs)}), the curves that reach it imply vendor revenue of <strong>${fmt(Math.min(...fitRev))}–${fmt(Math.max(...fitRev))}</strong> a year (Orbital Insight's real 2023 revenue: $9.5M). `;
  }
  const miss = A.map((a,i)=>a.kObs?null:DECAY[i].name).filter(Boolean);
  if (miss.length) v += `${miss.join(", ")} never reaches that price within ${KMAX} clients, so it does not fit the observed market. `;
  v += `The best case under any curve is <strong>${fmt(ceil)}</strong>/yr against <strong>$130M</strong> of venture capital raised.`;
  $("verdict").innerHTML = v;

  // chart data
  const ds1 = DECAY.map((d,i)=>line(d.name, ks.map(k=>100*EDGE*d.f(k)), cols[i]));
  const ds3 = DECAY.map((d,i)=>line(d.name, ks.map(k=>price(k,d,p)), cols[i]));
  const ds2 = DECAY.map((d,i)=>line(d.name, ks.map(k=>revenue(k,d,p)), cols[i], {
      pointRadius: ks.map(k=> k===A[i].bestK||k===A[i].kObs ? 6:0),
      pointBackgroundColor: ks.map(k=> k===A[i].bestK ? cols[i] : css("--card")),
      pointBorderColor: cols[i], pointBorderWidth:2}));
  ds2.push(line("OI 2023 revenue", ks.map(()=>OI_REV), ink2, {borderDash:[5,4],borderWidth:1}));
  ds3.push(line("Reported range (top)", ks.map(()=>BAND[1]), "transparent", {fill:false}));
  ds3.push(line("Reported range", ks.map(()=>BAND[0]), "transparent", {fill:"-1", backgroundColor:css("--s1")+"18"}));
  const n4 = Array.from({length:200},(_,i)=>i+1);
  const ds4 = DECAY.map((d,i)=>line("Alpha: "+d.name, n4.map(k=>revenue(k,d,p)), cols[i], {borderWidth:1.5}));
  ds4.push(line("Non-rival customers", n4.map(n=>n*p.nr), css("--ink"), {borderWidth:3}));

  const ymax3 = Math.min(fundProfit(1,{f:()=>1},p)*0.45, 2e7);
  const mk = (id, labels, ds, opts) => new Chart($(id), {type:"line", data:{labels, datasets:ds}, options:opts});
  [c1,c2,c3,c4].forEach(c=>c&&c.destroy());
  c1 = mk("c1", ks, ds1, baseOpts("Return over 3 days (%)", false));
  c2 = mk("c2", ks, ds2, baseOpts("Revenue per year", true));
  const o3 = baseOpts("Price per client", true); o3.scales.y.max = Math.max(ymax3, 1.5e6);
  o3.plugins.tooltip.filter = it => !it.dataset.label.startsWith("Reported");
  c3 = mk("c3", ks, ds3, o3);
  const o4 = baseOpts("Vendor revenue per year", true); o4.scales.x.title.text = "Number of clients";
  c4 = mk("c4", n4, ds4, o4);

  $("legend").innerHTML = DECAY.map((d,i)=>`<span><span class="sw" style="background:${cols[i]}"></span>${d.name}</span>`).join("")
    + `<span><span class="sw" style="background:${css("--ink")}"></span>Non-rival customers</span><span><span class="sw" style="background:${ink2}"></span>Orbital Insight 2023 revenue (dashed)</span>`;
}
["cap","proc","obs","nr"].forEach(id=>$(id).addEventListener("input",draw));
$("reset").addEventListener("click",()=>{for(const k in BASE) $(k).value=BASE[k]; draw();});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",draw);
draw();
