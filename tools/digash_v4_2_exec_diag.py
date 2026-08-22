#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from collections import Counter
from pathlib import Path
import numpy as np,pandas as pd
import digash_v4_stage0 as s0

FEE=5.0; SLIP=1.0; HOLD=24.0

def args():
 p=argparse.ArgumentParser(description='Digash V4.2 execution diagnostics; same detector events, alternative causal execution only')
 p.add_argument('--datadir',default='/freqtrade/user_data/data/binance/futures')
 p.add_argument('--events',default='/freqtrade/user_data/digash_v4_2_fidelity/events.csv')
 p.add_argument('--outdir',default='/freqtrade/user_data/digash_v4_2_fidelity/exec_diag')
 p.add_argument('--fee-bps-side',type=float,default=FEE);p.add_argument('--slippage-bps-side',type=float,default=SLIP)
 p.add_argument('--max-hold-hours',type=float,default=HOLD);p.add_argument('--limit-minutes',type=int,default=15)
 return p.parse_args()

def log(x):print(x,flush=True)

def load1m(root,sym):
 p=s0._data_path(root,sym,'1m');x=pd.read_feather(p)[['date','open','high','low','close']].copy();x.date=pd.to_datetime(x.date,utc=True)
 x=x.sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
 for c in 'open high low close'.split():x[c]=pd.to_numeric(x[c],errors='coerce')
 return x.dropna().reset_index(drop=True)

def pivots5(x):
 q=x.set_index('date').resample('5min',label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last')).dropna().reset_index()
 h=q.high.to_numpy(float);l=q.low.to_numpy(float);rows=[]
 for i in range(2,len(q)-2):
  av=pd.Timestamp(q.date.iloc[i])+pd.Timedelta(minutes=15)
  if h[i]>max(h[i-2:i]) and h[i]>=max(h[i+1:i+3]):rows.append((av,'H',h[i]))
  if l[i]<min(l[i-2:i]) and l[i]<=min(l[i+1:i+3]):rows.append((av,'L',l[i]))
 return pd.DataFrame(rows,columns=['available_time','kind','price'])

def local_target(piv,t,entry,side,final):
 if piv.empty:return None
 cut=t-pd.Timedelta(hours=48);kind='H' if side>0 else 'L'
 z=piv[(piv.available_time<=t)&(piv.available_time>=cut)&(piv.kind==kind)].copy()
 z=z[side*(z.price-entry)>0]
 z=z[side*(final-z.price)>=0]
 if z.empty:return None
 z['dist']=side*(z.price-entry)
 return float(z.sort_values('dist').iloc[0].price)

def touched_stop(side,lo,hi,p):return lo<=p if side>0 else hi>=p
def touched_target(side,lo,hi,p):return hi>=p if side>0 else lo<=p

def netret(entry,exitp,side,fee,slip):
 f=fee/1e4;s=slip/1e4;ee=entry*(1+side*s);xx=exitp*(1-side*s)
 return side*(xx-ee)/ee - f*(1+xx/ee)

def find_signal_close(x,entry_time):
 dt=x.date.to_numpy(dtype='datetime64[ns]');i=int(np.searchsorted(dt,np.datetime64(entry_time.to_datetime64()),side='left'))-1
 return (float(x.close.iloc[i]),i) if i>=0 else (None,None)

def find_limit_fill(x,entry_time,price,minutes):
 dt=x.date.to_numpy(dtype='datetime64[ns]');st=int(np.searchsorted(dt,np.datetime64(entry_time.to_datetime64()),side='left'))
 en=min(len(x),int(np.searchsorted(dt,np.datetime64((entry_time+pd.Timedelta(minutes=minutes)).to_datetime64()),side='left')))
 for i in range(st,en):
  if float(x.low.iloc[i])<=price<=float(x.high.iloc[i]):return i
 return None

def replay(x,start_i,entry_time,entry,stop,final,side,fee,slip,maxh,mode='FINAL',trigger=None,partial=0.0,fill_delay=0):
 risk=side*(entry-stop)
 base={'risk_abs':risk,'stop_bps':risk/entry*1e4 if entry else np.nan,'fill_delay_min':fill_delay}
 if not np.isfinite(risk) or risk<=0:return {**base,'status':'BAD_RISK'}
 dt=x.date.to_numpy(dtype='datetime64[ns]');end_t=entry_time+pd.Timedelta(hours=maxh);end_i=min(len(x),int(np.searchsorted(dt,np.datetime64(end_t.to_datetime64()),side='right')))
 if start_i>=end_i:return {**base,'status':'NO_WINDOW'}
 lo=x.low.to_numpy(float);hi=x.high.to_numpy(float);cl=x.close.to_numpy(float)
 active_stop=stop;be_at=None;triggered=False;realized_frac=0.0;realized_gross=0.0;realized_netret=0.0;remain=1.0
 mfe=-math.inf;mae=math.inf;exit_i=None;exitp=None;reason=None
 for i in range(start_i,end_i):
  if be_at is not None and i>=be_at:active_stop=entry
  if side>0:mfe=max(mfe,(hi[i]-entry)/risk);mae=min(mae,(lo[i]-entry)/risk)
  else:mfe=max(mfe,(entry-lo[i])/risk);mae=min(mae,(entry-hi[i])/risk)
  if touched_stop(side,lo[i],hi[i],active_stop):
   exit_i=i;exitp=active_stop;reason='BE' if abs(active_stop-entry)<1e-14*max(1,abs(entry)) else 'STOP';break
  target=final
  if mode=='TP':target=entry+side*risk*float(trigger)
  if touched_target(side,lo[i],hi[i],target):
   exit_i=i;exitp=target;reason='TP' if mode=='TP' else 'FINAL';break
  if mode in ('BE','PARTIAL') and not triggered:
   trigp=entry+side*risk*float(trigger)
   if touched_target(side,lo[i],hi[i],trigp):
    triggered=True;be_at=i+1
    if mode=='PARTIAL':
     frac=float(partial);realized_frac=frac;remain=1-frac;realized_gross=frac*float(trigger);realized_netret=frac*netret(entry,trigp,side,fee,slip)
    continue
 if exit_i is None:
  exit_i=end_i-1;exitp=float(cl[exit_i]);reason='TIMEOUT'
 gross2=side*(exitp-entry)/risk
 nr2=netret(entry,exitp,side,fee,slip)
 if mode=='PARTIAL' and realized_frac>0:
  gross=realized_gross+remain*gross2;nr=realized_netret+remain*nr2
 else:gross=gross2;nr=nr2
 stop_pct=risk/entry;netR=nr/stop_pct
 return {**base,'status':'OK','exit_time':pd.Timestamp(x.date.iloc[exit_i])+pd.Timedelta(minutes=1),'exit_reason':reason,'gross_R':float(gross),'net_R':float(netR),'mfe_R':float(mfe),'mae_R':float(mae),'triggered':bool(triggered)}

def one_event(ev,x,piv,A):
 t=pd.Timestamp(ev.entry_time);side=1 if ev.side=='LONG' else -1;entry=float(ev.entry_price);stop=float(ev.initial_stop);final=float(ev.final_target);bound=float(ev.bos_price)
 dt=x.date.to_numpy(dtype='datetime64[ns]');base_i=int(np.searchsorted(dt,np.datetime64(t.to_datetime64()),side='left'))
 local=local_target(piv,t,entry,side,final);risk=side*(entry-stop);localR=(side*(local-entry)/risk if local is not None and risk>0 else np.nan)
 variants=[]
 def add(name,ep,sp,st_i,et,mode='FINAL',trigger=None,partial=0.0,delay=0,extra=None):
  r=replay(x,st_i,et,ep,sp,final,side,A.fee_bps_side,A.slippage_bps_side,A.max_hold_hours,mode,trigger,partial,delay)
  q={'pair':ev.pair,'entry_time':t,'side':ev.side,'entry_family':ev.entry_family,'variant':name,'entry_price_used':ep,'stop_used':sp,'final_target':final,'local5_target':local,'local5_R_from_base':localR,'setup_episode_id':ev.setup_episode_id};q.update(r)
  if extra:q.update(extra)
  variants.append(q)
 add('BASE_FINAL',entry,stop,base_i,t)
 sc,si=find_signal_close(x,t)
 if sc is not None and side*(sc-stop)>0:add('ENTRY_SIGNAL_CLOSE_FINAL',sc,stop,base_i,t,extra={'entry_improvement_R_vs_base':side*(entry-sc)/risk if risk>0 else np.nan})
 fi=find_limit_fill(x,t,bound,A.limit_minutes)
 if fi is not None and side*(bound-stop)>0:
  ft=pd.Timestamp(x.date.iloc[fi]);add('ENTRY_BOUNDARY_LIMIT15_FINAL',bound,stop,fi,ft,delay=float((ft-t)/pd.Timedelta(minutes=1)),extra={'entry_improvement_R_vs_base':side*(entry-bound)/risk if risk>0 else np.nan})
 for mult in (1.25,1.50,2.00):
  sp=entry-side*risk*mult;add(f'STOP_X{mult:.2f}_FINAL',entry,sp,base_i,t)
 for rr in (0.5,1.0,1.5,2.0,3.0):add(f'TP_{rr:.1f}R',entry,stop,base_i,t,'TP',rr)
 for rr in (0.5,1.0,1.5,2.0):add(f'BE_{rr:.1f}R_FINAL',entry,stop,base_i,t,'BE',rr)
 add('PARTIAL50_1R_BE_FINAL',entry,stop,base_i,t,'PARTIAL',1.0,.5)
 add('PARTIAL50_2R_BE_FINAL',entry,stop,base_i,t,'PARTIAL',2.0,.5)
 if local is not None and localR>0:
  add('LOCAL5_EXIT',entry,stop,base_i,t,'TP',localR)
  add('LOCAL5_BE_FINAL',entry,stop,base_i,t,'BE',localR)
 return variants

def pf(s):
 pos=s[s>0].sum();neg=-s[s<0].sum();return float(pos/neg) if neg>0 else (math.inf if pos>0 else None)

def stats(df):
 rows=[]
 for (v,fam),g in df[df.status=='OK'].groupby(['variant','entry_family'],dropna=False):
  rows.append({'variant':v,'family':fam,'n':len(g),'gross_E_R':g.gross_R.mean(),'net_E_R':g.net_R.mean(),'winrate_net':(g.net_R>0).mean(),'PF_net':pf(g.net_R),'stop_rate':(g.exit_reason=='STOP').mean(),'BE_rate':(g.exit_reason=='BE').mean(),'median_MFE_R':g.mfe_R.median(),'median_stop_bps':g.stop_bps.median(),'median_local5_R':g.local5_R_from_base.median()})
 for v,g in df[df.status=='OK'].groupby('variant'):
  rows.append({'variant':v,'family':'ALL','n':len(g),'gross_E_R':g.gross_R.mean(),'net_E_R':g.net_R.mean(),'winrate_net':(g.net_R>0).mean(),'PF_net':pf(g.net_R),'stop_rate':(g.exit_reason=='STOP').mean(),'BE_rate':(g.exit_reason=='BE').mean(),'median_MFE_R':g.mfe_R.median(),'median_stop_bps':g.stop_bps.median(),'median_local5_R':g.local5_R_from_base.median()})
 return pd.DataFrame(rows)

def split_stats(df):
 cut=pd.Timestamp('2026-04-01',tz='UTC');z=df[df.status=='OK'].copy();z['split']=np.where(z.entry_time<cut,'EARLY','LATE');rows=[]
 for (v,sp),g in z.groupby(['variant','split']):rows.append({'variant':v,'split':sp,'n':len(g),'gross_E_R':g.gross_R.mean(),'net_E_R':g.net_R.mean(),'PF_net':pf(g.net_R),'winrate_net':(g.net_R>0).mean()})
 return pd.DataFrame(rows)

def main():
 A=args();root=Path(A.datadir);out=Path(A.outdir);out.mkdir(parents=True,exist_ok=True);ev=pd.read_csv(A.events);ev.entry_time=pd.to_datetime(ev.entry_time,utc=True);ev.asof_time=pd.to_datetime(ev.asof_time,utc=True)
 rows=[]
 for sym,g in ev.groupby('pair',sort=True):
  x=load1m(root,sym);p=pivots5(x);log(f'{sym}: events={len(g)} pivots5={len(p)}')
  for r in g.itertuples(index=False):rows.extend(one_event(r,x,p,A))
 z=pd.DataFrame(rows);z.entry_time=pd.to_datetime(z.entry_time,utc=True);z.to_csv(out/'execution_replay.csv',index=False)
 s=stats(z);s.to_csv(out/'variant_stats.csv',index=False);ss=split_stats(z);ss.to_csv(out/'variant_split_stats.csv',index=False)
 allstats=s[s.family=='ALL'].sort_values('net_E_R',ascending=False)
 base=allstats[allstats.variant=='BASE_FINAL'].iloc[0].to_dict() if (allstats.variant=='BASE_FINAL').any() else {}
 fills=z.groupby('variant').size().to_dict();total=len(ev)
 summary={'stage':'Digash V4.2 execution diagnostics','detector_changed':False,'events':total,'assumptions':{'fee_bps_side':A.fee_bps_side,'slippage_bps_side':A.slippage_bps_side,'max_hold_hours':A.max_hold_hours,'boundary_limit_cancel_minutes':A.limit_minutes,'intrabar':'adverse-first','note':'Diagnostic variants isolate entry/stop/target/management. They are not detector tuning and are not profitability claims.'},'baseline':base,'variant_fill_counts':{k:int(v) for k,v in fills.items()},'all_variant_stats':allstats.to_dict('records'),'files':{'execution_replay':str(out/'execution_replay.csv'),'variant_stats':str(out/'variant_stats.csv'),'variant_split_stats':str(out/'variant_split_stats.csv')}}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
 log('=== EXECUTION DIAGNOSTICS ===')
 for r in allstats.itertuples(index=False):log(f'{r.variant}: n={r.n} grossE={r.gross_E_R:.3f}R netE={r.net_E_R:.3f}R PF={r.PF_net:.3f} win={r.winrate_net:.1%}')
 log(f"summary={out/'summary.json'}")
if __name__=='__main__':main()
