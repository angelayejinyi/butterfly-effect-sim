# Interaction effect gap with Tit-for-Tat as the fixed-opponent reference (A_TfT = S_LLM - S_TfT),
# alongside the paper's A = S_LLM - mean over four fixed opponents. Paired seeds, 10,000 bootstrap replicates.
# Run from the ICLR_github root: python code/analysis/pd_tft_reference_gap.py
import pandas as pd, numpy as np, glob, os
F={'P':'P4_PLAIN_STRATEGIC','D':'P5_DESCRIPTIVE_STRATEGIC','T':'P6_TABULAR_STRATEGIC'}
ops=['TitForTat','Random','AlwaysCooperate','AlwaysDefect']
def single(m,f,o):
    r={}
    for p in glob.glob(f'{m}/single_agent/{f}/{o}/seed*/rounds.csv'):
        d=pd.read_csv(p); r[int(d.seed.iloc[0])]=(d.agent_action=='Cooperate').mean()
    return pd.Series(r)
def two(m,f):
    r={}
    for p in glob.glob(f'{m}/two_agent/{f}/seed*/rounds.csv'):
        d=pd.read_csv(p); r[int(d.seed.iloc[0])]=((d.agent_a_action=='Cooperate').mean()+(d.agent_b_action=='Cooperate').mean())/2
    return pd.Series(r)
ROOT="data/results/PD/persona"
os.chdir(ROOT)
rng=np.random.default_rng(0)
for m in ['gpt-5.2','claude-haiku-4-5-20251001','gemini-2.5-flash','deepseek-v3.2','deepseek-v4-flash']:
  if not os.path.isdir(f'{m}/two_agent'): continue
  for a,b in [('P','D'),('P','T'),('D','T')]:
    L=pd.concat([two(m,F[a]),two(m,F[b])],axis=1).dropna()
    S={o:pd.concat([single(m,F[a],o),single(m,F[b],o)],axis=1).dropna() for o in ops}
    seeds=sorted(set(L.index).intersection(*[set(S[o].index) for o in ops]))
    dl=(L.loc[seeds,1]-L.loc[seeds,0]).values
    ds={o:(S[o].loc[seeds,1]-S[o].loc[seeds,0]).values for o in ops}
    def stat(idx):
        sl=abs(dl[idx].mean()); sf=np.mean([abs(ds[o][idx].mean()) for o in ops]); st=abs(ds['TitForTat'][idx].mean())
        return sl, sf, st, sl-sf, sl-st
    full=stat(np.arange(len(seeds)))
    bs=np.array([stat(rng.integers(0,len(seeds),len(seeds))) for _ in range(10000)])
    lo,hi=np.percentile(bs[:,4],[2.5,97.5]); lo2,hi2=np.percentile(bs[:,3],[2.5,97.5])
    print(f"{m[:12]:12s} {a}-{b} n={len(seeds)} SLLM={full[0]:.3f} Sfix={full[1]:.3f} STfT={full[2]:.3f} A={full[3]:+.3f}[{lo2:+.2f},{hi2:+.2f}] A_TfT={full[4]:+.3f}[{lo:+.3f},{hi:+.3f}]")
