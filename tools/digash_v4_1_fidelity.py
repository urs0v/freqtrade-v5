#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,random,zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import digash_v4_stage0 as s0

RR=3.0; CT_MIN=15; PROTO_MIN=6; PROTO_MAX=30; RETEST=60; STOP_ATR=.80; SEED=4411

def args():
 p=argparse.ArgumentParser(description='Digash V4.1 sequential fidelity; no PnL')
 p.add_argument('--datadir',default='/freqtrade/user_data/data/binance/futures'); p.add_argument('--outdir',default='/freqtrade/user_data/digash_v4_1_fidelity')
 p.add_argument('--gold',default='/opt/rmv5/tools/digash_v4_1_gold.csv'); p.add_argument('--start',default='2025-11-01'); p.add_argument('--end',default='2026-08-19')
 p.add_argument('--pairs',default=','.join(s0.DEFAULT_PAIRS)); p.add_argument('--workers',type=int,default=12); p.add_argument('--sample',type=int,default=100); p.add_argument('--no-render',action='store_true')
 return p.parse_args()

def log(x): print(x,flush=True)
def active(r): return str(r.get('act_activity_class','INACTIVE')) in ('ACTIVE','SUPER_ACTIVE')
def htf(r,side):
 d1=int(r.get('h1_dir',0) if pd.notna(r.get('h1_dir',0)) else 0); d4=int(r.get('h4_dir',0) if pd.notna(r.get('h4_dir',0)) else 0)
 return (d1==side and d4!=-side, '1h+4h' if d1==side==d4 else '1h')

def consumed(levels,h1):
 dates=pd.to_datetime(h1.date,utc=True).reset_index(drop=True); c=h1.close.to_numpy(float); dt=dates.to_numpy(dtype='datetime64[ns]'); out=[]
 for q in levels:
  z=dict(q); j=int(np.searchsorted(dt,np.datetime64(pd.Timestamp(z['available_time']).to_datetime64()))); tol=max(.08*z['atr'],abs(z['price'])*.00015); ca=None
  if j<len(c)-2:
   b=c[j:]>z['price']+tol if z['kind']=='H' else c[j:]<z['price']-tol; hit=np.flatnonzero(b[:-1]&b[1:])
   if len(hit): ca=pd.Timestamp(dates.iloc[j+int(hit[0])+1])+pd.Timedelta(hours=1)
  z['consumed_at']=ca; out.append(z)
 return out

def targets(levels,t,entry,side):
 kind='H' if side>0 else 'L'; cut=t-pd.Timedelta(days=90); a=[]
 for z in levels:
  at=pd.Timestamp(z['available_time']); ca=z.get('consumed_at'); p=float(z['price'])
  if z['kind']!=kind or at>t or at<cut or (ca is not None and pd.Timestamp(ca)<=t): continue
  if side*(p-entry)<=0: continue
  a.append(z)
 a.sort(key=lambda z:side*(z['price']-entry)); u=[]
 for z in a:
  if u and abs(z['price']-u[-1]['price'])<=.15*max(z['atr'],u[-1]['atr']): continue
  u.append(z)
  if len(u)>=6: break
 if not u:return []
 out=[u[0]]
 for z in u[1:]:
  if abs(z['price']-out[-1]['price'])<=1.75*max(z['atr'],out[-1]['atr']): out.append(z)
  else: break
  if len(out)>=3: break
 return out

def gate(entry,stop,target,side,a):
 risk=side*(entry-stop); bps=risk/entry*1e4 if risk>0 else math.nan
 if not np.isfinite(risk) or risk<=0:return None,'NO_VALID_INVALIDATION'
 if np.isfinite(a) and a>0 and risk<STOP_ATR*a:return None,'MICRO_STOP'
 if bps<8 or bps>700:return None,'INVALID_STOP_GEOMETRY'
 r=side*(target-entry)/risk
 return ((r,bps),'OK') if np.isfinite(r) and r>=RR else (None,'AVAILABLE_R_LT_3')

def event(sym,x,i,eidx,side,fam,stop,bound,tg,r,bps,tf,episode,proto,sw=False,ret=None,note=''):
 row=x.iloc[i]; entry=float(x.iloc[eidx].open); t1,fin=tg[0],tg[-1]; asof=pd.Timestamp(row.asof_time)
 return {'pair':sym,'asof_time':asof,'entry_time':pd.Timestamp(x.iloc[eidx].date),'side':'LONG' if side>0 else 'SHORT','side_i':side,'entry_family':fam,
 'activity_class':str(row.get('act_activity_class','INACTIVE')),'activity_source':'OHLCV_PROXY','htf_direction':'LONG' if side>0 else 'SHORT','htf_tf':tf,
 'level_type':f"TARGET_{t1['kind']}",'level_id':t1['id'],'level_center':float(t1['price']),'level_width':float(.15*t1['atr']),'level_touch_count':1,'level_prominence_atr':float(t1['prominence_atr']),
 'cascade_count':len(tg),'local_structure_type':'SEQUENTIAL_PROTORGOVKA_BREAK' if fam!='LEVEL_BREAK' else 'MULTITOUCH_LEVEL_BREAK','protorgovka_duration_min':int((asof-proto)/pd.Timedelta(minutes=1)) if proto is not None else None,
 'bos_price':float(bound),'retest_price':ret,'sweep_depth_atr':None,'had_sweep':bool(sw),'entry_price':entry,'initial_stop':float(stop),'initial_risk_abs':float(side*(entry-stop)),'initial_risk_bps':float(bps),
 'target_1':float(t1['price']),'final_target':float(fin['price']),'available_R':float(r),'target_path':json.dumps([{'id':z['id'],'tf':z['tf'],'price':float(z['price'])} for z in tg]),
 'management_model':'NOT_TESTED_V4_1','future_outcome_used':False,'setup_episode_id':episode,'v41_notes':note,'_signal_idx':i,'_entry_idx':eidx}

def sweeps(x):
 h=x.high.to_numpy(float); l=x.low.to_numpy(float); c=x.close.to_numpy(float); a=x.atr1m.to_numpy(float); ph=x.high.shift(1).rolling(30,min_periods=20).max().to_numpy(float); pl=x.low.shift(1).rolling(30,min_periods=20).min().to_numpy(float)
 return np.isfinite(pl)&(l<pl-.05*a)&(c>pl), np.isfinite(ph)&(h>ph+.05*a)&(c<ph)

def scan_side(sym,x,levels,side,start,end,veto):
 n=len(x); c=x.close.to_numpy(float); h=x.high.to_numpy(float); l=x.low.to_numpy(float); a=x.atr1m.to_numpy(float); m=x.m5_dir.fillna(0).to_numpy(int); r20=x.prior_ret20.fillna(0).to_numpy(float); clean=x.clean_structure.fillna(False).to_numpy(bool); dates=pd.to_datetime(x.asof_time,utc=True); sl,sh=sweeps(x); sw=sl if side>0 else sh
 seg=np.cumsum(np.r_[True,m[1:]!=m[:-1]]); state=run=0; ct_i=seg_i=proto_i=-1; bound=stop=np.nan; had=False; seen=set(); bases=[]; out=[]
 for i in range(35,n-2):
  t=pd.Timestamp(dates.iloc[i]); ct=(m[i]==-side and side*r20[i]<0)
  if state==0:
   run=run+1 if ct else 0
   if run>=CT_MIN: state=1; ct_i=i; seg_i=int(seg[i]); had=bool(sw[max(0,i-45):i+1].any())
   continue
  if int(seg[i])!=seg_i and m[i]!=-side: state=0; run=0; continue
  had=had or bool(sw[i])
  if state==1:
   if i-ct_i>=PROTO_MIN and clean[i-PROTO_MIN+1:i+1].all():
    w=max(0,i-11); bound=float(np.max(h[w:i+1]) if side>0 else np.min(l[w:i+1])); stop=float(np.min(l[w:i+1]) if side>0 else np.max(h[w:i+1])); proto_i=i-PROTO_MIN+1; state=2
   continue
  if state==2:
   if i-proto_i>PROTO_MAX: state=3; continue
   cross=(c[i]>bound+.03*a[i] and c[i-1]<=bound) if side>0 else (c[i]<bound-.03*a[i] and c[i-1]>=bound)
   if not cross: continue
   ep=f"{sym}:{'L' if side>0 else 'S'}:m5seg{seg_i}"; state=3
   if ep in seen: continue
   seen.add(ep)
   if not(start<=t<end) or not active(x.iloc[i]): continue
   ok,tf=htf(x.iloc[i],side)
   if not ok: veto['STRUCTURE:NO_ACTIONABLE_HTF']+=1; continue
   entry=float(x.iloc[i+1].open); tg=targets(levels,t,entry,side)
   if not tg: veto['STRUCTURE:NO_FRESH_TARGET']+=1; continue
   g,why=gate(entry,stop,float(tg[-1]['price']),side,a[i]); fam='SWEEP_RETURN' if had else 'BOS_BREAK'
   if g: out.append(event(sym,x,i,i+1,side,fam,stop,bound,tg,g[0],g[1],tf,ep,pd.Timestamp(dates.iloc[proto_i]),had,note='countertrend -> later protorgovka -> frozen boundary -> first close break'))
   else:veto[f'{fam}:{why}']+=1
   bases.append((i,side,bound,ep,tf,pd.Timestamp(dates.iloc[proto_i]),had))
  elif state==3 and int(seg[i])!=seg_i: state=0;run=0
 return out,retests(sym,x,levels,bases,start,end,veto)

def retests(sym,x,levels,bases,start,end,veto):
 out=[]
 for bi,side,bound,ep,tf,proto,had in bases:
  touch=None; ext=None
  for j in range(bi+2,min(len(x)-1,bi+RETEST+1)):
   row=x.iloc[j]; a=float(row.atr1m)
   if not(np.isfinite(a) and a>0):continue
   tol=max(.2*a,abs(bound)*.00005)
   if touch is None:
    if float(row.low)<=bound+tol and float(row.high)>=bound-tol: touch=j;ext=float(row.low if side>0 else row.high)
    continue
   ext=min(ext,float(row.low)) if side>0 else max(ext,float(row.high)); react=(float(row.close)>bound+.35*a and row.close>row.open) if side>0 else (float(row.close)<bound-.35*a and row.close<row.open)
   if not react:continue
   t=pd.Timestamp(row.asof_time)
   if not(start<=t<end) or not active(row):break
   ok,tf2=htf(row,side)
   if not ok:veto['RETEST_REACTION:NO_ACTIONABLE_HTF']+=1;break
   entry=float(x.iloc[j+1].open);tg=targets(levels,t,entry,side)
   if not tg:veto['RETEST_REACTION:NO_FRESH_TARGET']+=1;break
   stop=float(ext-.03*a if side>0 else ext+.03*a);g,why=gate(entry,stop,float(tg[-1]['price']),side,a)
   if g:out.append(event(sym,x,j,j+1,side,'RETEST_REACTION',stop,bound,tg,g[0],g[1],tf2,ep,proto,had,ret=ext,note='BOS -> later touch -> later rejection close'))
   else:veto[f'RETEST_REACTION:{why}']+=1
   break
 return out

def mature(snaps):
 d={}
 for z in sorted(snaps,key=lambda q:pd.Timestamp(q['active_from'])):
  if int(z.get('touch_count',0))>=3:d.setdefault(str(z['level_id']),z)
 return list(d.values())

def level_breaks(sym,x,levels,snaps,start,end,veto):
 out=[]; dates=pd.to_datetime(x.asof_time,utc=True);dt=dates.to_numpy(dtype='datetime64[ns]');c=x.close.to_numpy(float);a=x.atr1m.to_numpy(float);clean=x.clean_structure.fillna(False).to_numpy(bool)
 for z in mature(snaps):
  side=1 if z['kind']=='H' else -1; at=pd.Timestamp(z['active_from']); center=float(z['center']); ss=max(35,int(np.searchsorted(dt,np.datetime64(max(at,start-pd.Timedelta(days=1)).to_datetime64())))); ee=min(len(x)-2,int(np.searchsorted(dt,np.datetime64(min(end,at+pd.Timedelta(days=14)).to_datetime64()))))
  for i in range(ss,ee):
   cross=(c[i]>center+.03*a[i] and c[i-1]<=center) if side>0 else (c[i]<center-.03*a[i] and c[i-1]>=center)
   if not cross:continue
   if i<8 or clean[i-8:i].sum()<6:veto['LEVEL_BREAK:NO_PREBREAK_PROTORGOVKA']+=1;continue
   row=x.iloc[i];t=pd.Timestamp(row.asof_time)
   if not(start<=t<end) or not active(row):continue
   ok,tf=htf(row,side)
   if not ok:veto['LEVEL_BREAK:NO_ACTIONABLE_HTF']+=1;continue
   entry=float(x.iloc[i+1].open);tg=[q for q in targets(levels,t,entry,side) if side*(q['price']-center)>0]
   if not tg:veto['LEVEL_BREAK:NO_FRESH_TARGET_BEYOND_LEVEL']+=1;continue
   pre=x.iloc[i-12:i];stop=float(pre.low.min() if side>0 else pre.high.max());g,why=gate(entry,stop,float(tg[-1]['price']),side,a[i])
   if not g:veto[f'LEVEL_BREAK:{why}']+=1;continue
   ev=event(sym,x,i,i+1,side,'LEVEL_BREAK',stop,center,tg,g[0],g[1],tf,f"{sym}:{side}:level:{z['level_id']}",pd.Timestamp(x.iloc[i-8].asof_time),note='3+ separated touches -> prebreak protorgovka -> clean close through')
   ev.update(level_type='MULTITOUCH_LEVEL_BREAK_V41',level_id=z['level_id'],level_center=center,level_width=float(z['width']),level_touch_count=int(z['touch_count']),level_prominence_atr=float(z['prominence_atr']));out.append(ev);break
 return out

def process(sym,datadir,act,start_s,end_s):
 start=s0._ts(start_s);end=s0._ts(end_s)+pd.Timedelta(days=1);v=Counter()
 try:x,h5,h1,h4,raw,sn=s0._prepare_pair(sym,Path(datadir),Path(act),start,s0._ts(end_s))
 except Exception as e:return [],{}, {'pair':sym,'status':'ERROR','error':repr(e)}
 lv=consumed(raw,h1);a,b=scan_side(sym,x,lv,1,start,end,v);c,d=scan_side(sym,x,lv,-1,start,end,v);e=level_breaks(sym,x,lv,sn,start,end,v);events=a+b+c+d+e
 dd={}
 for q in events:dd.setdefault((q['setup_episode_id'],q['entry_family']),q)
 events=sorted(dd.values(),key=lambda q:q['entry_time']);return events,dict(v),{'pair':sym,'status':'OK','events':len(events),'family_counts':dict(Counter(q['entry_family'] for q in events)),'bars_1m':len(x),'bars_5m':len(h5),'bars_1h':len(h1),'bars_4h':len(h4),'prominent_levels':len(lv),'multitouch_snapshots':len(sn),'mature_3touch_levels':len(mature(sn))}

def gold_report(ev,path):
 if not path.exists():return {'status':'MISSING_GOLD'}
 g=pd.read_csv(path);g.entry_time=pd.to_datetime(g.entry_time,utc=True);e=ev.copy() if not ev.empty else pd.DataFrame(columns=['pair','side','entry_family','entry_time']);
 if not e.empty:e.entry_time=pd.to_datetime(e.entry_time,utc=True)
 rows=[]
 for r in g.itertuples(index=False):
  q=e[(e.pair==r.pair)&(e.side==r.side)]; near=((q.entry_time-r.entry_time).abs()<=pd.Timedelta(minutes=15)).any() if len(q) else False; same=q[q.entry_family==r.family]; sf=((same.entry_time-r.entry_time).abs()<=pd.Timedelta(minutes=15)).any() if len(same) else False; rows.append({'sample_no':r.sample_no,'action':r.v41_gold_action,'any_match_15m':bool(near),'same_family_match_15m':bool(sf)})
 z=pd.DataFrame(rows);k=z[z.action=='KEEP'];r=z[z.action=='REJECT'];return rows,{'keep_total':len(k),'keep_match_15m':int(k.any_match_15m.sum()),'keep_recall_15m':float(k.any_match_15m.mean()) if len(k) else None,'reject_total':len(r),'reject_still_match_15m':int(r.any_match_15m.sum()),'reject_specificity_15m':float((~r.any_match_15m).mean()) if len(r) else None}

def main():
 A=args();datadir=Path(A.datadir);out=Path(A.outdir);out.mkdir(parents=True,exist_ok=True);start=s0._ts(A.start);end=s0._ts(A.end);pairs=[p.strip().upper() for p in A.pairs.split(',') if p.strip()];pairs=[p for p in pairs if s0._data_path(datadir,p,'1m').exists()];act=out/'activity_hourly.feather'
 log('=== DIGASH V4.1 SEQUENTIAL FIDELITY ===');log('Research only. NO PnL. Stage-0 unchanged.')
 if not act.exists():s0._build_activity(datadir,act,pairs,start,end)
 events=[];v=Counter();meta=[]
 with ProcessPoolExecutor(max_workers=max(1,min(A.workers,len(pairs)))) as ex:
  fs={ex.submit(process,p,str(datadir),str(act),A.start,A.end):p for p in pairs}
  for f in as_completed(fs):
   e,w,m=f.result();events+=e;v.update(w);meta.append(m);log(f"{m['pair']}: events={m.get('events',0)} {m.get('family_counts',{})}")
 ev=pd.DataFrame(events)
 if not ev.empty:ev.entry_time=pd.to_datetime(ev.entry_time,utc=True);ev.asof_time=pd.to_datetime(ev.asof_time,utc=True);ev=ev.sort_values(['entry_time','pair','entry_family']).reset_index(drop=True)
 s0._strip_internal(ev).to_csv(out/'events.csv',index=False);gr=gold_report(ev,Path(A.gold));gold={}
 if isinstance(gr,tuple):rows,gold=gr;pd.DataFrame(rows).to_csv(out/'gold_regression.csv',index=False)
 else:gold=gr
 zp=None
 if not A.no_render and not ev.empty:
  sample=s0._stratified_sample(ev,min(A.sample,len(ev)),seed=SEED).reset_index(drop=True);sample['_sample_no']=np.arange(1,len(sample)+1);zp=s0._render_review_bundle(sample,datadir,out,start,end);new=out/'digash_v4_1_fidelity_review.zip';zp.rename(new);zp=new
 summary={'stage':'Digash V4.1 sequential fidelity','pnl_computed':False,'start':str(start),'end':str(end),'events':len(ev),'family_counts':dict(Counter(ev.entry_family)) if not ev.empty else {},'pair_counts':dict(Counter(ev.pair)) if not ev.empty else {},'side_counts':dict(Counter(ev.side)) if not ev.empty else {},'veto_counts':dict(v.most_common()),'pair_meta':meta,'gold_regression':gold,'spec_contract':{'sequence':'confirmed 5m countertrend -> later 1m protorgovka -> frozen boundary -> actual close break','sweep_return':'sweep is setup context; entry only after later protorgovka+BOS','retest_reaction':'BOS -> later touch -> later rejection close','level_break':'3+ separated touches -> prebreak protorgovka -> clean close','targets':'unconsumed causal HTF targets','stop':'structural; microscopic stops rejected','rr_gate':3.0,'future_outcome_used':False},'review_bundle':str(zp) if zp else None}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str));log(f"RESULT events={len(ev)} families={summary['family_counts']} gold={gold}");log(f"summary={out/'summary.json'}")
if __name__=='__main__':main()
