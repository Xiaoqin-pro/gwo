import pandas as pd, numpy as np
from scipy import stats
mine=pd.concat([pd.read_csv("results/igwo29_20261016_085616/runs.csv")])
mine=mine[mine.Status=="ok"]
ext=pd.read_csv(r"D:\111\Desktop\噜噜\9.29\CEC2017_8Algorithm_30Seed\results\main_8algorithms\all_quality_runs.csv")
ext=ext[(ext.Status=="ok") & (ext.Seed.between(3001,3010)) & (ext.FunctionID!=2)]
keep=['HIDMS-PSO-IndexCorrected','HCLPSO-AuthorSource','DMS-PSO-SIS2005','SAQ-CSO','SPSO-2011-OfficialC','HPSO-TVAC-CEC','CLPSO-AuthorSource','Original-CSO-MR02']
ext=ext[ext.Condition.isin(keep)][['FunctionID','Seed','Condition','FinalError']].rename(columns={'Condition':'Algorithm'})
allr=pd.concat([mine[['FunctionID','Seed','Algorithm','FinalError']],ext],ignore_index=True)
m=allr.groupby(['FunctionID','Algorithm']).FinalError.mean().unstack()
r=m.rank(axis=1); ar=r.mean().sort_values()
print("=== 13 algorithms, 29 functions, seeds 3001-3010 (10 seeds, 100k FE) ===")
print("%-5s %-28s %8s %8s" % ('rank','algorithm','avg rank','--'))
for i,a in enumerate(ar.index,1):
    tag='  <== ours' if a.startswith('RASE') or a=='IGWO' else ''
    print("%5d %-28s %8.2f%s" % (i,a,ar[a],tag))
print()
def holm(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); adj=np.empty(n); run=0
    for rr,i in enumerate(o): run=max(run,(n-rr)*p[i]); adj[i]=min(1,run)
    return adj
me='RASE-IGWO-K25'; funcs=sorted(allr.FunctionID.unique())
print("RASE-IGWO-K25 vs each baseline (29 functions, 10 seeds):")
for a in ar.index:
    if a==me: continue
    w=int((m[me]<m[a]).sum()); l=int((m[me]>m[a]).sum()); ps=[]
    for f in funcs:
        x=allr[(allr.FunctionID==f)&(allr.Algorithm==me)].FinalError.values
        y=allr[(allr.FunctionID==f)&(allr.Algorithm==a)].FinalError.values
        ps.append(stats.mannwhitneyu(x,y,alternative='two-sided',method='asymptotic',use_continuity=True)[1])
    adj=holm(ps)
    sb=int(((adj<0.05)&(m[me].values<m[a].values)).sum()); sw=int(((adj<0.05)&(m[me].values>m[a].values)).sum())
    print("  %-28s W/L=%2d/%-2d ratio=%6.3f  Holm %2d/%2d" % (a,w,l,np.exp(np.median(np.log((m[me].values+1e-30)/(m[a].values+1e-30)))),sb,sw))
