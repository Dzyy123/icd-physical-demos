"""Portable entry points for the published chain/fork demonstration records."""
from pathlib import Path
import argparse, hashlib, html, json, sys, urllib.request

ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def module_path(scene):sys.path.insert(0,str(ROOT/'src'/scene))
def graph_dict(g):return {'directed':g.directed.tolist(),'skeleton':g.skeleton.tolist(),'undirected':[list(e) for e in sorted(g.undirected)]}
def check_graph(g,expected):
    d=graph_dict(g)
    for key in d:assert d[key]==expected[key],key

def refit(scene,out):
    import numpy as np
    r=ROOT/'records'/scene; batches=load(r/'readouts.json')['batches'];module_path(scene)
    checked=[]
    if scene=='fork':
        from revisable_graph import fit_pool,select_experiment
        xs=[];ms=[];pool=cpds=None
        for b in batches:
            n=b['round']
            if n:
                target,reason=select_experiment(pool,cpds)
                assert target==load(r/f'decision_{n:02d}.json')['target'],('target',n)
            xs.append(np.array(b['binary']));ms.append(np.array(b['masks']))
            pool,cpds,_=fit_pool(np.concatenate(xs),np.concatenate(ms))
            expected=load(r/f'state_{n:02d}.json')
            assert pool['map_edges']==expected['map_edges'],('map_edges',n)
            assert abs(pool['map_score']-expected['map_score'])<1e-8,('score',n)
            assert [g['code'] for g in pool['graphs']]==[g['code'] for g in expected['graphs']],('candidate_pool',n)
            checked.append({'round':n,'edges':pool['map_edges'],'score':pool['map_score']})
            print('verified fork round',n,flush=True)
    else:
        sys.path.insert(0,str(ROOT/'src/chain/backend'))
        import acq_closed as backend
        from variation_guard import install
        install(backend)
        ob=np.array(batches[0]['binary'],float);mean=ob.mean(0);std=ob.std(0)+1e-8
        encode=lambda x:(x-mean)/std
        g=backend.learn_observational_cpdag(encode(ob),7,alpha=.01,max_order=None,min_eff=.03,source='PC_from_frozen_RGB_ACC_binary_readouts')
        check_graph(g,load(r/'initial_graph.json'))
        xs=[ob];ms=[np.zeros_like(ob)];counts=np.zeros(7,int);rng=np.random.default_rng(42)
        trace=load(r/'graph_trace.json');checked.append({'round':0,**graph_dict(g)})
        for b in batches[1:]:
            n=b['round'];j,_=backend.select_target('coverage',g.undirected,g.directed,None,counts,1,7,rng,0)
            assert int(j)==load(r/f'round{n}_decision.json')['target']
            xs.append(np.array(b['binary']));ms.append(np.array(b['masks']));counts[int(j)]+=1
            g,ev=backend.discover(encode(np.concatenate(xs)),np.concatenate(ms),g.skeleton,7,alpha=.01,min_eff=.03,do_thr=.12,do_margin=0.,initial_graph=g,legacy_skeleton=False,return_graph=True)
            check_graph(g,trace[n-1]['graph']);checked.append({'round':n,**graph_dict(g)})
            print('verified chain round',n,flush=True)
    result={'status':'MATCHES_RECORDED_GRAPH_AND_SELECTION','scene':scene,'states':checked,
            'input':'Saved frozen-encoder/ACC binary readouts. This does not rerun pixel inference or estimate benchmark performance.'}
    save(out,result);return result

def simulate(scene,round_index,row,out):
    import numpy as np
    from PIL import Image
    module_path(scene)
    record=load(ROOT/f'records/{scene}/batches/round{round_index:02d}-launch.json')
    job=record['jobs'][row]
    if scene=='chain':
        from robot_seven import RobotSeven
        env=RobotSeven()
    else:
        if round_index>=13:
            from robot_repaired import RobotFork
        else:
            from robot_revision import RobotFork
        env=RobotFork()
    target=Path(out);target.mkdir(parents=True,exist_ok=False)
    try:im,labels,audit=env.sample_robot(**job,capture=str(target/'frames'))
    finally:env.close()
    Image.fromarray(im).save(target/'endpoint.png')
    digest=hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()
    batch=next(b for b in load(ROOT/f'records/{scene}/readouts.json')['batches'] if b['round']==round_index)
    expected=batch['input_image_sha256'][row]
    save(target/'audit.json',audit)
    result={'scene':scene,'round':round_index,'row':row,'job':job,'endpoint_sha256':digest,'recorded_endpoint_sha256':expected,
            'byte_identical':digest==expected,'target_realized':audit['target_realized'],'settled':audit['settled'],
            'min_contact_distance_m':audit['min_contact_distance'],'post_release_pose_resets':audit['post_release_pose_resets'],
            'note':'Physics replay, not inference. Exact pixels can depend on platform/library versions. Oracle measurements are audit-only.'}
    save(target/'result.json',result);print(json.dumps(result,indent=2))
    assert audit['target_realized'] and audit['settled'] and audit['min_contact_distance']>=-.001 and audit['post_release_pose_resets']==0
    return result

def verify():
    manifest=load(ROOT/'MANIFEST.json')
    for name,expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    print('Verified',len(manifest['sha256']),'public files; no model weights required.')

def report(scene,out):
    r=ROOT/'records'/scene; roles=['P','F1','F2','F3','F4','B','L']
    if scene=='chain':
        raw=[load(r/'initial_graph.json')]+[x['graph'] for x in load(r/'graph_trace.json')]
        states=[([(i,j) for i,row in enumerate(s['directed']) for j,v in enumerate(row) if v],s['undirected']) for s in raw]
        pos=[(80+i*130,180) for i in range(7)]
    else:
        states=[(load(r/f'state_{i:02d}.json')['map_edges'],[]) for i in range(21)]
        pos=[(80,180),(260,80),(260,280),(460,80),(460,280),(650,280),(830,280)]
    panels=[]
    for n,(edges,undirected) in enumerate(states):
        lines=[]
        for a,b in list(edges)+list(undirected):
            x,y=pos[a];xx,yy=pos[b];dx=xx-x;dy=yy-y;norm=(dx*dx+dy*dy)**.5;u=28/norm
            style='stroke-dasharray="7 5"' if [a,b] in undirected else 'marker-end="url(#arrow)"'
            lines.append(f'<line x1="{x+u*dx}" y1="{y+u*dy}" x2="{xx-u*dx}" y2="{yy-u*dy}" stroke="#477695" stroke-width="3" {style}/>')
        for label,(x,y) in zip(roles,pos):lines.append(f'<circle cx="{x}" cy="{y}" r="26" fill="#edf3f5" stroke="#477695"/><text x="{x}" y="{y+6}" text-anchor="middle">{label}</text>')
        panels.append(f'<section><h2>Round {n}</h2><svg viewBox="0 0 930 360"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#477695"/></marker></defs>{"".join(lines)}</svg></section>')
    doc=f'<!doctype html><meta charset="utf-8"><title>{scene.title()} — complete graph trace</title><style>body{{font:18px system-ui;max-width:1000px;margin:40px auto;color:#24394a}}section{{border-top:1px solid #ccd7dc}}svg{{width:100%}}</style><h1>{scene.title()}: every recorded graph state</h1><p>Recorded trace playback, not a new inference run. Blue arrows are learned hypotheses, not correctness labels. Dashed links are unresolved. Node identities were supervised. Fork rounds 13–20 were an exploratory continuation after inspecting round 12.</p>{"".join(panels)}'
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(doc);print(p)

def film(scene,out):
    data=load(ROOT/'records/media.json')[scene];p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(data['url'],p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==data['sha256']
    print('Downloaded verified public film:',p)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('verify')
    for name in ['refit','simulate','report','film']:
        q=sub.add_parser(name);q.add_argument('--scene',choices=['chain','fork'],required=True);q.add_argument('--out',required=True)
        if name=='simulate':q.add_argument('--round',type=int,default=1);q.add_argument('--row',type=int,default=0)
    a=p.parse_args()
    if a.cmd=='verify':verify()
    elif a.cmd=='simulate':simulate(a.scene,a.round,a.row,a.out)
    else:globals()[a.cmd](a.scene,a.out)
