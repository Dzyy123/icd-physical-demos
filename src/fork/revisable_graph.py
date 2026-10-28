"""Small, explicit revisable binary-DAG hypothesis layer (no simulator truth).

Truncated BDeu scores exclude a target's own intervened rows. Exact top-K
subset dynamic programming searches all DAGs of the declared max indegree.
Candidate support is conditional on the retained pool, not a calibrated CI.
"""
import itertools
import math
import hashlib
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=300000)
def tie_key(code):
    # Fixed before acquisition; do not privilege the display's node order.
    return hashlib.blake2b(f'42:{code}'.encode(),digest_size=8).digest()


def edges(code, n):
    return [(i,j) for i in range(n) for j in range(n) if code & (1 << (i*n+j))]


def parent_sets(n, j, max_parents):
    other=[i for i in range(n) if i!=j]
    return [tuple(p) for k in range(max_parents+1) for p in itertools.combinations(other,k)]


def counts_for(x, masks, j, parents):
    z=x[masks[:,j]==0]
    config=np.zeros(len(z),dtype=int)
    for k,p in enumerate(parents): config |= z[:,p].astype(int) << k
    counts=np.zeros((1<<len(parents),2),dtype=int)
    np.add.at(counts,(config,z[:,j].astype(int)),1)
    return counts


def fit_pool(x, masks, k=32, max_parents=2, ess=1.0):
    x=np.asarray(x,dtype=np.int8);masks=np.asarray(masks,dtype=np.int8)
    assert x.ndim==2 and masks.shape==x.shape
    assert np.isin(x,[0,1]).all() and np.isin(masks,[0,1]).all()
    n=x.shape[1];local={};cpds={};options={}
    for j in range(n):
        options[j]=[]
        for pars in parent_sets(n,j,max_parents):
            counts=counts_for(x,masks,j,pars)
            alpha=ess/len(counts);a=alpha/2
            score=sum(math.lgamma(alpha)-math.lgamma(alpha+int(row.sum()))+
                      sum(math.lgamma(a+int(c))-math.lgamma(a) for c in row)
                      for row in counts)
            key=(j,pars);local[key]=score
            cpds[key]=(counts[:,1]+a)/(counts.sum(1)+alpha)
            pmask=sum(1<<p for p in pars)
            ecode=sum(1<<(p*n+j) for p in pars)
            options[j].append((pmask,ecode,score,pars))
    dp={0:[(0.0,0)]}
    for size in range(1,n+1):
        for subset in itertools.combinations(range(n),size):
            mask=sum(1<<j for j in subset);candidates={}
            for sink in subset:
                rest=mask^(1<<sink)
                for pmask,ecode,score,_ in options[sink]:
                    if pmask & ~rest: continue
                    for prev,code in dp[rest]:
                        full=code|ecode;value=prev+score
                        if value>candidates.get(full,-math.inf):candidates[full]=value
            dp[mask]=sorted(((v,c) for c,v in candidates.items()),key=lambda z:(-round(z[0],10),tie_key(z[1])))[:k]
    best=dp[(1<<n)-1];scores=np.array([s for s,_ in best]);weights=np.exp(scores-scores.max());weights/=weights.sum()
    support=np.zeros((n,n));graphs=[]
    for (score,code),weight in zip(best,weights):
        es=edges(code,n)
        for i,j in es:support[i,j]+=weight
        graphs.append({'code':int(code),'score':float(score),'weight':float(weight),'edges':[list(e) for e in es]})
    return {'n':n,'graphs':graphs,'edge_support':support.tolist(),
            'map_edges':graphs[0]['edges'],'map_score':graphs[0]['score'],
            'support_scope':'relative support within retained top-K DAG pool'},cpds,local


def predictive(pool, cpds, target, value):
    n=pool['n'];states=np.array(list(itertools.product([0,1],repeat=n)),dtype=np.int8)
    states=states[states[:,target]==value];pred=[]
    for g in pool['graphs']:
        probability=np.ones(len(states))
        for j in range(n):
            if j==target:continue
            pars=tuple(sorted(i for i,q in g['edges'] if q==j))
            config=np.zeros(len(states),dtype=int)
            for bit,p in enumerate(pars):config|=states[:,p].astype(int)<<bit
            p1=cpds[(j,pars)][config]
            probability*=np.where(states[:,j]==1,p1,1-p1)
        assert np.isclose(probability.sum(),1.,atol=1e-8),probability.sum()
        pred.append(probability)
    return states,np.array(pred)


def select_experiment(pool, cpds):
    weights=np.array([g['weight'] for g in pool['graphs']]);scores=[];expectations={}
    for target in range(pool['n']):
        values=[];means=[];map_means=[]
        for value in [0,1]:
            states,pred=predictive(pool,cpds,target,value)
            mixture=weights@pred
            js=float(np.sum(weights[:,None]*pred*np.log(np.maximum(pred,1e-300)/np.maximum(mixture,1e-300))))
            values.append(max(0.,js));means.append((mixture@states).tolist());map_means.append((pred[0]@states).tolist())
        score=float(np.mean(values));scores.append(score)
        expectations[str(target)]={'mixture_mean_by_command':means,'map_mean_by_command':map_means,'disagreement_by_command':values}
    target=int(np.argmax(scores))
    support=np.array(pool['edge_support']);uncertain=[]
    for i,j in itertools.permutations(range(pool['n']),2):
        if i!=target and j!=target:continue
        p=float(support[i,j])
        if .02<p<.98:uncertain.append({'edge':[i,j],'candidate_support':p,'uncertainty':min(p,1-p)})
    uncertain.sort(key=lambda a:(-a['uncertainty'],a['edge']))
    return target,{'criterion':'mean predictive Jensen-Shannon disagreement across commands',
                  'scores_nats':scores,'selected_target':target,'selected_score_nats':scores[target],
                  'uncertain_incident_hypotheses':uncertain[:6],
                  'predictions_before_acquisition':expectations[str(target)]}


def revision(previous,current):
    old={tuple(e) for e in previous['map_edges']};new={tuple(e) for e in current['map_edges']}
    withdrawn=old-new;added=new-old
    reversed_edges=[list(e) for e in sorted(withdrawn) if (e[1],e[0]) in added]
    return {'withdrawn':[list(e) for e in sorted(withdrawn)],'added':[list(e) for e in sorted(added)],
            'reversed_from':reversed_edges,'unchanged':[list(e) for e in sorted(old&new)],
            'scope':'changes in best-scoring provisional DAG, not automatically certified causal facts'}
