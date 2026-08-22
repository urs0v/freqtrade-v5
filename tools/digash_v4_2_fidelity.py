#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import digash_v4_stage0 as s0
import digash_v4_1_fidelity as v41

RR_FINAL_MIN=3.0
CT_MIN=15
PROTO_MIN=6
PROTO_MAX=30
RETEST_WINDOW=75
STOP_ATR_MIN=1.0
MAX_ENTRY_EXTENSION_ATR=1.25
DEPARTURE_ATR=0.60
REACTION_ATR=0.35
SWEEP_MAX_AGE=60
HTF_CONFLICT_ATR=2.0
LEVEL_PRE=12
LEVEL_CLEAN_MIN=7
LEVEL_SIDE_RATIO=0.80
SEED=4422

def args():
 p=argparse.ArgumentParser(description='Digash V4.2 quality-first fidelity; no PnL')
 p.add_argument('--datadir',default='/freqtrade/user_data/data/binance/futures')
 p.add_argument('--outdir',default='/freqtrade/user_data/digash_v4_2_fidelity')
 p.add_argument('--gold',default='/opt/rmv5/tools/digash_v4_2_gold.csv')
 p.add_argument('--start',default='2025-11-01');p.add_argument('--end',default='2026-08-19')
 p.add_argument('--pairs',default=','.join(s0.DEFAULT_PAIRS));p.add_argument('--workers',type=int,default=12)
 p.add_argument('--sample',type=int,default=100);p.add_argument('--no-render',action='store_true')
 return p.parse_args()

def log(x): print(x,flush=True)

def add_htf_freshness(x,h1):
 f=h1[['date','close','atr14']].copy()
 f['h1_move6_atr']=(f.close-f.close.shift(6))/f.atr14.replace(0,np.nan)
 f['available_time']=f.date+pd.Timedelta(hours=1)
 return s0._merge_asof_feature(x,f[['available_time','h1_move6_atr']],['h1_move6_atr'],'').reset_index(drop=True)

def htf(row,side):
 d1=int(row.get('h1_dir',0) if pd.notna(row.get('h1_dir',0)) else 0)
 d4=int(row.get('h4_dir',0) if pd.notna(row.get('h4_dir',0)) else 0)
 if d1!=side or d4==-side:return False,'STRUCTURE_CONFLICT'
 impulse=float(row.get('h1_move6_atr',0) if pd.notna(row.get('h1_move6_atr',0)) else 0)
 if side*impulse < -HTF_CONFLICT_ATR:return False,'FRESH_HTF_IMPULSE_CONFLICT'
 return True,('1h+4h' if d4==side else '1h')

def gate(entry,stop,tg,side,a,bound=None):
 risk=side*(entry-stop)
 if not np.isfinite(risk) or risk<=0:return None,'NO_VALID_INVALIDATION'
 if np.isfinite(a) and a>0 and risk<STOP_ATR_MIN*a:return None,'MICRO_STOP'
 bps=risk/entry*1e4
 if bps<8 or bps>700:return None,'INVALID_STOP_GEOMETRY'
 ext=side*(entry-bound)/a if bound is not None and np.isfinite(a) and a>0 else 0.0
 if bound is not None and ext>MAX_ENTRY_EXTENSION_ATR:return None,'CHASE_ENTRY'
 rr1=side*(float(tg[0]['price'])-entry)/risk
 rrf=side*(float(tg[-1]['price'])-entry)/risk
 if not np.isfinite(rrf) or rrf<RR_FINAL_MIN:return None,'AVAILABLE_R_FINAL_LT_3'
 return {'rr_t1':float(rr1),'rr_final':float(rrf),'risk_bps':float(bps),'entry_extension_atr':float(ext)},'OK'

def event(sym,x,i,eidx,side,fam,stop,bound,tg,g,tf,episode,proto,had=False,ret=None,note=''):
 q=v41.event(sym,x,i,eidx,side,fam,stop,bound,tg,g['rr_final'],g['risk_bps'],tf,episode,proto,had,ret=ret,note=note)
 q['available_R_T1']=g['rr_t1'];q['available_R_final']=g['rr_final'];q['available_R']=g['rr_final']
 q['entry_extension_atr']=g['entry_extension_atr'];q['management_model']='NOT_TESTED_V4_2'
 q['v42_quality_contract']='QUALITY_FIRST_NO_FREQUENCY_OBJECTIVE'
 return q

def scan_side(sym,x,levels,side,start,end,veto):
 n=len(x);c=x.close.to_numpy(float);h=x.high.to_numpy(float);l=x.low.to_numpy(float);a=x.atr1m.to_numpy(float)
 m=x.m5_dir.fillna(0).to_numpy(int);r20=x.prior_ret20.fillna(0).to_numpy(float);clean=x.clean_structure.fillna(False).to_numpy(bool)
 dates=pd.to_datetime(x.asof_time,utc=True);sl,sh=v41.sweeps(x);sw=sl if side>0 else sh
 seg=np.cumsum(np.r_[True,m[1:]!=m[:-1]])
 state=run=0;ct_i=seg_i=proto_i=-1;bound=stop=np.nan;sweep_i=-1;seen=set();bases=[];sweeps_out=[]
 for i in range(35,n-2):
  t=pd.Timestamp(dates.iloc[i]);ct=(m[i]==-side and side*r20[i]<0)
  if state==0:
   run=run+1 if ct else 0
   if run>=CT_MIN:
    state=1;ct_i=i;seg_i=int(seg[i]);ct_start=i-run+1
    hits=np.flatnonzero(sw[max(0,ct_start):i+1]);sweep_i=(max(0,ct_start)+int(hits[-1])) if len(hits) else -1
   continue
  if int(seg[i])!=seg_i and m[i]!=-side:
   state=0;run=0;sweep_i=-1;continue
  if state==1:
   if sw[i]:sweep_i=i
   if i-ct_i>=PROTO_MIN and clean[i-PROTO_MIN+1:i+1].all():
    w=max(0,i-11);bound=float(np.max(h[w:i+1]) if side>0 else np.min(l[w:i+1]))
    stop=float(np.min(l[w:i+1]) if side>0 else np.max(h[w:i+1]));proto_i=i-PROTO_MIN+1;state=2
   continue
  if state==2:
   if i-proto_i>PROTO_MAX:state=3;continue
   cross=(c[i]>bound+.03*a[i] and c[i-1]<=bound) if side>0 else (c[i]<bound-.03*a[i] and c[i-1]>=bound)
   if not cross:continue
   ep=f"{sym}:{'L' if side>0 else 'S'}:m5seg{seg_i}";state=3
   if ep in seen:continue
   seen.add(ep)
   if not(start<=t<end) or not v41.active(x.iloc[i]):continue
   ok,tf=htf(x.iloc[i],side)
   if not ok:veto[f'STRUCTURE:NO_ACTIONABLE_HTF:{tf}']+=1;continue
   entry=float(x.iloc[i+1].open);tg=v41.targets(levels,t,entry,side)
   if not tg:veto['STRUCTURE:NO_FRESH_TARGET']+=1;continue
   veto['BOS_BREAK:DISABLED_AS_ENTRY']+=1
   bases.append({'bi':i,'side':side,'bound':bound,'stop':stop,'ep':ep,'tf':tf,'proto':pd.Timestamp(dates.iloc[proto_i]),'t1':float(tg[0]['price'])})
   valid_sweep=sweep_i>=0 and sweep_i<proto_i and 0<=i-sweep_i<=SWEEP_MAX_AGE
   if valid_sweep:
    g,why=gate(entry,stop,tg,side,a[i],bound)
    if g:sweeps_out.append(event(sym,x,i,i+1,side,'SWEEP_RETURN',stop,bound,tg,g,tf,ep,pd.Timestamp(dates.iloc[proto_i]),True,note='countertrend -> sweep/reclaim -> later protorgovka -> internal BOS'))
    else:veto[f'SWEEP_RETURN:{why}']+=1
  elif state==3 and int(seg[i])!=seg_i:
   state=0;run=0;sweep_i=-1
 return sweeps_out,retests(sym,x,levels,bases,start,end,veto)

def retests(sym,x,levels,bases,start,end,veto):
 out=[]
 for b in bases:
  bi=b['bi'];side=b['side'];bound=b['bound'];base_stop=b['stop'];touch=None;ext=None;depart=False
  for j in range(bi+1,min(len(x)-1,bi+RETEST_WINDOW+1)):
   row=x.iloc[j];a=float(row.atr1m)
   if not(np.isfinite(a) and a>0):continue
   if side>0 and float(row.low)<=base_stop:break
   if side<0 and float(row.high)>=base_stop:break
   if touch is None and ((side>0 and float(row.high)>=b['t1']) or (side<0 and float(row.low)<=b['t1'])):break
   if not depart:
    depart=(float(row.close)>=bound+DEPARTURE_ATR*a) if side>0 else (float(row.close)<=bound-DEPARTURE_ATR*a)
    if not depart:continue
    continue
   tol=max(.18*a,abs(bound)*.00005)
   if touch is None:
    touched=(float(row.low)<=bound+tol and float(row.high)>=bound-tol)
    if touched:touch=j;ext=float(row.low if side>0 else row.high)
    continue
   ext=min(ext,float(row.low)) if side>0 else max(ext,float(row.high))
   prev=x.iloc[j-1]
   react=(float(row.close)>bound+REACTION_ATR*a and float(row.close)>float(row.open) and float(row.close)>float(prev.high)) if side>0 else (float(row.close)<bound-REACTION_ATR*a and float(row.close)<float(row.open) and float(row.close)<float(prev.low))
   if not react:continue
   t=pd.Timestamp(row.asof_time)
   if not(start<=t<end) or not v41.active(row):break
   ok,tf=htf(row,side)
   if not ok:veto[f'RETEST_REACTION:NO_ACTIONABLE_HTF:{tf}']+=1;break
   entry=float(x.iloc[j+1].open);tg=v41.targets(levels,t,entry,side)
   if not tg:veto['RETEST_REACTION:NO_FRESH_TARGET']+=1;break
   stop=float(ext-.03*a if side>0 else ext+.03*a);g,why=gate(entry,stop,tg,side,a,bound)
   if g:out.append(event(sym,x,j,j+1,side,'RETEST_REACTION',stop,bound,tg,g,tf,b['ep'],b['proto'],False,ret=ext,note='internal BOS -> meaningful departure -> later touch -> rejection close through prior bar'))
   else:veto[f'RETEST_REACTION:{why}']+=1
   break
 return out

def level_breaks(sym,x,levels,snaps,start,end,veto):
 out=[];dates=pd.to_datetime(x.asof_time,utc=True);dt=dates.to_numpy(dtype='datetime64[ns]')
 c=x.close.to_numpy(float);a=x.atr1m.to_numpy(float);clean=x.clean_structure.fillna(False).to_numpy(bool)
 for z in v41.mature(snaps):
  side=1 if z['kind']=='H' else -1;at=pd.Timestamp(z['active_from']);center=float(z['center'])
  ss=max(LEVEL_PRE+2,int(np.searchsorted(dt,np.datetime64(max(at,start-pd.Timedelta(days=1)).to_datetime64()))))
  ee=min(len(x)-2,int(np.searchsorted(dt,np.datetime64(min(end,at+pd.Timedelta(days=14)).to_datetime64()))))
  first=None
  for i in range(ss,ee):
   if not(np.isfinite(a[i]) and a[i]>0):continue
   cross=(c[i]>center+.03*a[i] and c[i-1]<=center) if side>0 else (c[i]<center-.03*a[i] and c[i-1]>=center)
   if cross:first=i;break
  if first is None:continue
  i=first;row=x.iloc[i];aa=float(a[i]);pre=x.iloc[i-LEVEL_PRE:i]
  correct=((pre.close<=center+.05*aa).mean() if side>0 else (pre.close>=center-.05*aa).mean())
  if correct<LEVEL_SIDE_RATIO:veto['LEVEL_BREAK:NOT_APPROACHING_FROM_CORRECT_SIDE']+=1;continue
  if clean[i-8:i].sum()<LEVEL_CLEAN_MIN:veto['LEVEL_BREAK:NO_IMMEDIATE_PROTORGOVKA']+=1;continue
  near=((pre.high>=center-.35*aa).sum() if side>0 else (pre.low<=center+.35*aa).sum())
  if near<3:veto['LEVEL_BREAK:NOT_COMPRESSED_AT_BOUNDARY']+=1;continue
  last3=pre.tail(3).close.to_numpy(float)
  if np.mean(np.abs(last3-center))>.55*aa:veto['LEVEL_BREAK:LATE_OR_REMOTE_COMPRESSION']+=1;continue
  approach=side*(float(pre.close.iloc[-1])-float(pre.close.iloc[0]))/aa
  if approach<-.25:veto['LEVEL_BREAK:APPROACH_MOVING_AWAY']+=1;continue
  t=pd.Timestamp(row.asof_time)
  if not(start<=t<end) or not v41.active(row):continue
  ok,tf=htf(row,side)
  if not ok:veto[f'LEVEL_BREAK:NO_ACTIONABLE_HTF:{tf}']+=1;continue
  entry=float(x.iloc[i+1].open);tg=[q for q in v41.targets(levels,t,entry,side) if side*(q['price']-center)>0]
  if not tg:veto['LEVEL_BREAK:NO_FRESH_TARGET_BEYOND_LEVEL']+=1;continue
  stop=float(pre.low.min() if side>0 else pre.high.max());g,why=gate(entry,stop,tg,side,aa,center)
  if not g:veto[f'LEVEL_BREAK:{why}']+=1;continue
  ev=event(sym,x,i,i+1,side,'LEVEL_BREAK',stop,center,tg,g,tf,f"{sym}:{side}:level:{z['level_id']}",pd.Timestamp(x.iloc[i-8].asof_time),note='fresh 3+ touch level -> correct-side compression at boundary -> FIRST clean close through')
  ev.update(level_type='MULTITOUCH_LEVEL_BREAK_V42',level_id=z['level_id'],level_center=center,level_width=float(z['width']),level_touch_count=int(z['touch_count']),level_prominence_atr=float(z['prominence_atr']))
  out.append(ev)
 return out

def process(sym,datadir,act,start_s,end_s):
 start=s0._ts(start_s);end=s0._ts(end_s)+pd.Timedelta(days=1);v=Counter()
 try:x,h5,h1,h4,raw,sn=s0._prepare_pair(sym,Path(datadir),Path(act),start,s0._ts(end_s))
 except Exception as e:return [],{}, {'pair':sym,'status':'ERROR','error':repr(e)}
 x=add_htf_freshness(x,h1);lv=v41.consumed(raw,h1)
 a,b=scan_side(sym,x,lv,1,start,end,v);c,d=scan_side(sym,x,lv,-1,start,end,v);e=level_breaks(sym,x,lv,sn,start,end,v)
 events=a+b+c+d+e;dd={}
 for q in events:
  key=(q['setup_episode_id'],q['entry_family']);old=dd.get(key)
  if old is None or q['available_R_final']>old['available_R_final']:dd[key]=q
 events=sorted(dd.values(),key=lambda q:q['entry_time'])
 return events,dict(v),{'pair':sym,'status':'OK','events':len(events),'family_counts':dict(Counter(q['entry_family'] for q in events)),'bars_1m':len(x),'bars_5m':len(h5),'bars_1h':len(h1),'bars_4h':len(h4),'prominent_levels':len(lv),'multitouch_snapshots':len(sn),'mature_3touch_levels':len(v41.mature(sn))}

def gold_report(ev,path):
 if not path.exists():return {'status':'MISSING_GOLD'}
 g=pd.read_csv(path);g.entry_time=pd.to_datetime(g.entry_time,utc=True)
 e=ev.copy() if not ev.empty else pd.DataFrame(columns=['pair','side','entry_family','entry_time'])
 if not e.empty:e.entry_time=pd.to_datetime(e.entry_time,utc=True)
 rows=[]
 for r in g.itertuples(index=False):
  q=e[(e.pair==r.pair)&(e.side==r.side)];any60=((q.entry_time-r.entry_time).abs()<=pd.Timedelta(minutes=60)).any() if len(q) else False
  same=q[q.entry_family==r.family];sf60=((same.entry_time-r.entry_time).abs()<=pd.Timedelta(minutes=60)).any() if len(same) else False
  rows.append({'sample_no':r.sample_no,'quality':r.manual_quality,'family':r.family,'any_match_60m':bool(any60),'same_family_match_60m':bool(sf60)})
 z=pd.DataFrame(rows);va=z[z.quality=='VALID'];iv=z[z.quality=='INVALID'];bo=z[z.quality=='BORDERLINE']
 return rows,{'valid_total':len(va),'valid_match_60m':int(va.any_match_60m.sum()),'valid_recall_60m':float(va.any_match_60m.mean()) if len(va) else None,'invalid_total':len(iv),'invalid_still_match_60m':int(iv.any_match_60m.sum()),'invalid_specificity_60m':float((~iv.any_match_60m).mean()) if len(iv) else None,'borderline_total':len(bo),'borderline_match_60m':int(bo.any_match_60m.sum())}

def independent_sample(ev,gold_path,n):
 z=ev.copy()
 if gold_path.exists() and not z.empty:
  g=pd.read_csv(gold_path);g.entry_time=pd.to_datetime(g.entry_time,utc=True);z.entry_time=pd.to_datetime(z.entry_time,utc=True);keep=np.ones(len(z),dtype=bool)
  for r in g.itertuples(index=False):keep &= ~((z.pair==r.pair)&(z.side==r.side)&((z.entry_time-r.entry_time).abs()<=pd.Timedelta(minutes=120)))
  cand=z[keep]
  if len(cand)>=min(n,len(z)):z=cand
 return s0._stratified_sample(z,min(n,len(z)),seed=SEED)

def main():
 A=args();datadir=Path(A.datadir);out=Path(A.outdir);out.mkdir(parents=True,exist_ok=True);start=s0._ts(A.start);end=s0._ts(A.end)
 pairs=[p.strip().upper() for p in A.pairs.split(',') if p.strip()];pairs=[p for p in pairs if s0._data_path(datadir,p,'1m').exists()];act=out/'activity_hourly.feather'
 log('=== DIGASH V4.2 QUALITY-FIRST FIDELITY ===');log('Research only. NO PnL. Frequency is NOT an objective. Standalone BOS entries disabled.')
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
  sample=independent_sample(ev,Path(A.gold),min(A.sample,len(ev))).reset_index(drop=True);sample['_sample_no']=np.arange(1,len(sample)+1);s0._strip_internal(sample).to_csv(out/'review_sample.csv',index=False)
  zp=s0._render_review_bundle(sample,datadir,out,start,end);new=out/'digash_v4_2_fidelity_review.zip';zp.rename(new);zp=new
 summary={'stage':'Digash V4.2 quality-first fidelity','pnl_computed':False,'frequency_objective':False,'start':str(start),'end':str(end),'events':len(ev),'family_counts':dict(Counter(ev.entry_family)) if not ev.empty else {},'pair_counts':dict(Counter(ev.pair)) if not ev.empty else {},'side_counts':dict(Counter(ev.side)) if not ev.empty else {},'rr_t1':{'median':float(ev.available_R_T1.median()) if not ev.empty else None,'p10':float(ev.available_R_T1.quantile(.1)) if not ev.empty else None,'p90':float(ev.available_R_T1.quantile(.9)) if not ev.empty else None},'rr_final':{'median':float(ev.available_R_final.median()) if not ev.empty else None,'p10':float(ev.available_R_final.quantile(.1)) if not ev.empty else None,'p90':float(ev.available_R_final.quantile(.9)) if not ev.empty else None},'veto_counts':dict(v.most_common()),'pair_meta':meta,'gold_regression':gold,'spec_contract':{'objective':'quality/expectancy first; no trade-count reward or penalty','bos_break':'internal arming event only; NEVER standalone entry','sequence':'confirmed 5m countertrend -> later 1m protorgovka -> internal BOS','sweep_return':'sweep/reclaim must precede protorgovka in same countertrend episode; entry after later internal BOS','retest_reaction':'internal BOS -> >=0.60 ATR departure -> later touch -> rejection close through previous bar','level_break':'fresh 3+ touch level -> correct-side compression immediately at boundary -> FIRST clean close only','htf':'1h structure required; 4h may confirm/neutral, never oppose; fresh >=2 ATR 6h impulse against direction vetoes stale context','stop':'structural plus >=1.0 ATR1m noise floor','rr':'RR_FINAL >=3 gate; RR_T1 recorded separately, not tuned yet','entry':'chase veto if >1.25 ATR beyond boundary','future_outcome_used':False},'review_bundle':str(zp) if zp else None}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str));log(f"RESULT events={len(ev)} families={summary['family_counts']} gold={gold}");log(f"summary={out/'summary.json'}")
if __name__=='__main__':main()
