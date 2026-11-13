import pandas as pd, numpy as np, glob

mix='additional_experiments/results/PD_single_perturbation/gpt-5.2/two_agent_mixed'
P,D='P4_PLAIN_STRATEGIC','P5_DESCRIPTIVE_STRATEGIC'
def mixed_plain():
    r={}
    for a,b,col in [(P,D,'agent_a_action'),(D,P,'agent_b_action')]:
        for f in glob.glob(f'{mix}/A-{a}__B-{b}/seed*/rounds.csv'):
            d=pd.read_csv(f); r.setdefault(int(d.seed.iloc[0]),[]).append((d[col]=='Cooperate').mean())
    return pd.Series({k:np.mean(v) for k,v in r.items()})
def homo(f):
    r={}
    for p in glob.glob(f'data/results/PD/persona/gpt-5.2/two_agent/{f}/seed*/rounds.csv'):
        d=pd.read_csv(p); r[int(d.seed.iloc[0])]=((d.agent_a_action=='Cooperate').mean()+(d.agent_b_action=='Cooperate').mean())/2
    return pd.Series(r)
m=mixed_plain(); h=homo(P)
print(len(m),len(h), m.mean(), h.mean())
s=sorted(set(m.index)&set(h.index)); x=m[s].values-h[s].values
rng=np.random.default_rng(0); bs=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(10000)]
print('spill',x.mean(),np.percentile(bs,[2.5,97.5]))


obs=abs(x.mean()); signs=rng.choice([-1,1],size=(100000,len(x)))
print('perm p',( (np.abs((signs*x).mean(1))>=obs).sum()+1)/100001)
