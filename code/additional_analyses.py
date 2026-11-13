import sys, json, warnings
import os
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0,HERE)
import numpy as np, pandas as pd
from scipy.stats import spearmanr, norm as N_, rankdata
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings('ignore')
B=os.path.join(ROOT,'data')+os.sep
VARS=["cyclomatic_complexity","modularity","graph_density","entropy_wang","entropy_infoshannon","entropy_shannon","volume_area_ratio","volume_ratio","sphere_ratio","size"]
SEED=20260905; rng=np.random.default_rng(SEED)
raw=pd.read_excel(B+'Dataset_complejidad_con_size.xlsx').rename(columns={'succes_ratio':'success_ratio'})
num=raw.select_dtypes(include=[np.number]).columns
df=raw.groupby('file_name',as_index=False).agg({**{c:'mean' for c in num},'source':'first'})
print('n',len(df))
def qn(d,cols):
    Q=QuantileTransformer(output_distribution='normal',n_quantiles=len(d),random_state=SEED)
    return StandardScaler().fit_transform(Q.fit_transform(d[cols]))
X=qn(df,VARS)
pc=PCA(1,random_state=SEED).fit(X); s=pc.transform(X)[:,0]; sg=np.sign(spearmanr(s,df.cyclomatic_complexity).statistic); df['ICE']=s*sg
smap=pd.read_excel(B+'resultados_106.xlsx',sheet_name='solver original')
df['id']=df.file_name.str.extract(r'^(\d+)').astype(float)
M=df.merge(smap[['id','s_map']],on='id'); print('n smap',len(M))
R={}
R['rho_ice']=spearmanr(M.ICE,M.s_map).statistic; print('rho ICE',R['rho_ice'])
# profiles
sig=df[VARS].round(9).astype(str).agg('|'.join,axis=1); df['grp']=pd.factorize(sig)[0]
print('profiles',df.grp.nunique())
# --- correlations S_MAP vs success_ratio
R['rho_smap_sr']=spearmanr(M.s_map,M.success_ratio).statistic; R['p_smap_sr']=spearmanr(M.s_map,M.success_ratio).pvalue
print('S_MAP-SR',R['rho_smap_sr'],R['p_smap_sr'])
# --- AUC ICE only, grouped CV
y=(df.success_ratio>0).astype(int).values
cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED)
Z=df[['ICE']].values
p=np.zeros(len(df))
for tr,te in cv.split(Z,y,df.grp):
    # ICE refit in train would be stricter; use the standardized score from training-fold PCA
    Xtr=qn(df.iloc[tr],VARS)
    Q=QuantileTransformer(output_distribution='normal',n_quantiles=len(tr),random_state=SEED).fit(df.iloc[tr][VARS])
    sc=StandardScaler().fit(Q.transform(df.iloc[tr][VARS]))
    pcf=PCA(1,random_state=SEED).fit(sc.transform(Q.transform(df.iloc[tr][VARS])))
    f=lambda d: pcf.transform(sc.transform(Q.transform(d[VARS])))[:,0]
    str_,ste=f(df.iloc[tr]),f(df.iloc[te]); sgn=np.sign(spearmanr(str_,df.iloc[tr].cyclomatic_complexity).statistic)
    lr=LogisticRegression(max_iter=1000).fit((str_*sgn)[:,None],y[tr]); p[te]=lr.predict_proba((ste*sgn)[:,None])[:,1]
R['auc_ice_cv']=roc_auc_score(y,p); print('AUC ICE only grouped CV',R['auc_ice_cv'])
print('AUC raw (in-sample rank) ',roc_auc_score(y,df.ICE*-1) )
# --- paired bootstrap of rho differences & Steiger/Meng
def rho(a,b): return spearmanr(a,b).statistic
cand={'entropy_wang':M.entropy_wang.values,'cyclomatic_complexity':M.cyclomatic_complexity.values}
cand['binary']=None
# binary label from KMeans k2 on PCA k90 as in pipeline
pca_full=PCA(random_state=SEED).fit(X); k90=int(np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_),.90)+1)
Zp=PCA(n_components=k90,random_state=SEED).fit_transform(X)
lab=KMeans(2,n_init=50,random_state=SEED).fit(Zp).labels_
if df.cyclomatic_complexity[lab==0].median()>df.cyclomatic_complexity[lab==1].median(): lab=1-lab
df['bin']=lab; M=df.merge(smap[['id','s_map']],on='id')
print('rho bin',rho(M.bin,M.s_map),'k90',k90)
cand['binary']=M.bin.values
cand['mean']=(qn(df,VARS).mean(1)*sg)[df.id.isin(smap.id).values]
cand['graph6']=None
from sklearn.decomposition import PCA as P
G=['cyclomatic_complexity','modularity','graph_density','entropy_wang','entropy_infoshannon','entropy_shannon']
Xg=qn(df,G); sgp=P(1,random_state=SEED).fit(Xg).transform(Xg)[:,0]; sgp*=np.sign(spearmanr(sgp,df.cyclomatic_complexity).statistic); df['g6']=sgp
M=df.merge(smap[['id','s_map']],on='id'); cand['graph6']=M.g6.values
# NB: align orders
order=M.index
cand['mean']=pd.Series((qn(df,VARS).mean(1)*sg),index=df.index).loc[df.id.isin(smap.id)].values
# merge preserves df order for matched ids? check
assert (M.id.values==df.loc[df.id.isin(smap.id),'id'].values).all()
ice=M.ICE.values; sm=M.s_map.values; n=len(M)
for k,v in cand.items(): print(k, round(rho(v,sm),3))
out={}
for k,v in cand.items():
    if k=='cyclomatic_complexity': pass
    d=[]
    for _ in range(5000):
        i=rng.integers(0,n,n); d.append(rho(ice[i],sm[i])-rho(v[i],sm[i]))
    lo,hi=np.percentile(d,[2.5,97.5]); out[k]=(rho(ice,sm)-rho(v,sm),lo,hi)
    # Steiger (Meng, Rosenthal, Rubin 1992) z using spearman
    r1=rho(ice,sm); r2=rho(v,sm); r12=rho(ice,v)
    z1,z2=np.arctanh(r1),np.arctanh(r2); rm2=(r1**2+r2**2)/2
    f=min((1-r12)/(2*(1-rm2)),1); h=(1-f*rm2)/(1-rm2)
    z=(z1-z2)*np.sqrt((n-3)/(2*(1-r12)*h)); pv=2*(1-N_.cdf(abs(z)))
    out[k]+=(z,pv,r12)
    print(f'ICE - {k}: diff={out[k][0]:.3f} CI[{lo:.3f},{hi:.3f}] Meng z={z:.2f} p={pv:.3f} r12={r12:.3f}')
R['diffs']={k:[float(x) for x in v] for k,v in out.items()}
# --- N,E recovery
M_=df.cyclomatic_complexity.values; d_=df.graph_density.values
Nn=((d_+1)+np.sqrt((d_+1)**2+4*d_*(M_-2)))/(2*d_); E=M_+Nn-2
print('N mean/med',Nn.mean(),np.median(Nn),'E mean/med',E.mean(),np.median(E),'N dist',np.percentile(Nn,[0,25,50,75,100]))
print('N near-int frac',np.mean(np.abs(Nn-np.round(Nn))<0.05))
print('corr E~M',spearmanr(E,M_).statistic, np.corrcoef(E,M_)[0,1], 'E~N',np.corrcoef(E,Nn)[0,1])
R['N_mean']=Nn.mean();R['E_mean']=E.mean();R['N_near_int']=float(np.mean(np.abs(Nn-np.round(Nn))<0.05))
R['rho_E_M']=spearmanr(E,M_).statistic
print('rho N with size',spearmanr(Nn,df['size']).statistic)
# E vs ICE rho with S_MAP
df['E']=E;df['N']=Nn;M=df.merge(smap[['id','s_map']],on='id')
print('rho E smap',rho(M.E,M.s_map),'rho N smap',rho(M.N,M.s_map))
R['rho_E_smap']=rho(M.E,M.s_map);R['rho_N_smap']=rho(M.N,M.s_map)
# --- sensitivity log1p + z
def lz(d,cols): return StandardScaler().fit_transform(np.log1p(d[cols]))
XL=lz(df,VARS)
def structure(Xm,tag,nullkind='uniform'):
    pf=PCA(random_state=SEED).fit(Xm); k=int(np.searchsorted(np.cumsum(pf.explained_variance_ratio_),.90)+1)
    Zm=PCA(n_components=k,random_state=SEED).fit_transform(Xm)
    def W(Zq,kk):
        km=KMeans(kk,n_init=10,random_state=SEED).fit(Zq); return sum(((Zq[km.labels_==i]-Zq[km.labels_==i].mean(0))**2).sum() for i in range(kk))
    mn,mx=Zm.min(0),Zm.max(0); cov=np.cov(Zm.T); mu=Zm.mean(0)
    def ref(kind): return rng.uniform(mn,mx,Zm.shape) if kind=='uniform' else rng.multivariate_normal(mu,cov,Zm.shape[0])
    res={}
    for kind in ['uniform','gauss']:
        gap=[];sk=[]
        for kk in range(1,7):
            refs=[np.log(W(ref(kind),kk)) for _ in range(30)]
            gap.append(np.mean(refs)-np.log(W(Zm,kk))); sk.append(np.std(refs)*np.sqrt(1+1/30))
        kopt=next((kk for i,kk in enumerate(range(1,7)) if i+1<6 and gap[i]>=gap[i+1]-sk[i+1]),None)
        nul=[silhouette_score(u,KMeans(2,n_init=10,random_state=SEED).fit(u).labels_) for u in (ref(kind) for _ in range(30))]
        sil=silhouette_score(Zm,KMeans(2,n_init=50,random_state=SEED).fit(Zm).labels_)
        res[kind]=dict(kopt=kopt,gap1=gap[0],gap2=gap[1],sil=sil,nul=float(np.mean(nul)),nulsd=float(np.std(nul)),p_sil=float(np.mean(np.array(nul)>=sil)))
        print(tag,kind,res[kind])
    km=KMeans(2,n_init=50,random_state=SEED).fit(Zm).labels_; wd=AgglomerativeClustering(2,linkage='ward').fit(Zm).labels_
    res['ari']=adjusted_rand_score(km,wd);res['k90']=k;print(tag,'ARI',res['ari'],'k90',k)
    return res
R['qn']=structure(X,'quantile-normal'); R['log1p']=structure(XL,'log1p')
# marginal multimodality check: after quantile transform
# --- test-retest
d=raw[raw.duplicated('file_name',keep=False)][['file_name','success_ratio','time']]; print(d)
# --- counts / proportions
print('zeros',(df.success_ratio==0).sum(), 'profiles',df.grp.nunique())
print('time by source median'); print(df.groupby('source').time.median())
os.makedirs(os.path.join(ROOT,'results'),exist_ok=True)
json.dump(R,open(os.path.join(ROOT,'results','additional_analyses.json'),'w'),indent=1,default=float)
