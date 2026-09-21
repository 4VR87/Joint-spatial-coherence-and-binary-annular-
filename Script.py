#!/usr/bin/env python3

Study: joint spatial-coherence and binary-annular pupil optimization for
       defocus-robust weak-amplitude imaging.

All results are simulations. No experimental data are used.
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution, minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

SEED_PROP = 37
SEED_CUBIC = 41
SEED_QUARTIC = 43
LAMBDA0 = 550e-9
NA = 0.12
N_MED = 1.0
Z_UM = np.array([-100,-75,-50,-25,0,25,50,75,100], dtype=float)
Z_M = Z_UM*1e-6
WVALS = Z_M*NA**2/(N_MED*LAMBDA0)
BAND = (0.10,0.85)
SPAN = 1.5

# Fixed parameters produced by the disclosed searches on the N=96 design grid.
PARAMS = {
    "Clear": {"sigma": 0.5933496776994324},
    "Cubic": {"alpha": 6.56622221, "sigma": 0.6003584681130534},
    "Quartic": {"beta": -8.72958658, "sigma": 0.5026511498097901},
    "Proposed": {"r1": 0.6857834950155233, "r2": 0.8762796208289985,
                 "sigma": 0.5454812382609806, "phase": np.pi},
}

def grid(N:int, span:float=SPAN):
    x=np.linspace(-span,span,N,endpoint=False)
    dx=x[1]-x[0]
    X,Y=np.meshgrid(x,x,indexing='xy')
    R=np.sqrt(X**2+Y**2)
    return x,dx,X,Y,R

def mask_clear(X,Y,R):
    return np.zeros_like(R)

def mask_cubic(alpha):
    return lambda X,Y,R: alpha*(X**3+Y**3)

def mask_quartic(beta):
    return lambda X,Y,R: beta*R**4

def mask_ring(r1,r2,phase=np.pi):
    return lambda X,Y,R: np.where((R>r1)&(R<=r2), phase, 0.0)

def transfer2d(N:int, sigma:float, w:float, mask_func, span:float=SPAN):
    _,dx,X,Y,R=grid(N,span)
    aperture=(R<=1).astype(float)
    P=aperture*np.exp(1j*(mask_func(X,Y,R)+np.pi*w*R**2))
    S=(R<=sigma).astype(float)
    S/=S.sum()
    # Hopkins-type weak-object cross term C(f)=sum S(q) P(q+f) P*(q).
    C=fftconvolve(P, np.conj((S*P)[::-1,::-1]), mode='same')
    c=N//2
    H=np.real(C)/np.real(C[c,c])
    f=(np.arange(N)-c)*dx
    FX,FY=np.meshgrid(f,f,indexing='xy')
    FR=np.sqrt(FX**2+FY**2)
    return H,FR

def evaluate(mask_func,sigma,N=256,wvals=WVALS,band=BAND):
    hs=[]
    fr=None
    for w in wvals:
        h,fr=transfer2d(N,sigma,w,mask_func)
        hs.append(h)
    hs=np.stack(hs)
    bm=(fr>=band[0])&(fr<=band[1])
    vals=hs[:,bm]
    worst=vals.min(axis=0)
    m={
      'mean_worst':float(np.mean(worst)),
      'positive_worst':float(np.mean(np.maximum(worst,0))),
      'mean_axial_std':float(np.mean(np.std(vals,axis=0))),
      'infocus_mean':float(np.mean(vals[len(wvals)//2])),
      'neg_fraction':float(np.mean(vals<0)),
      'p10_worst':float(np.quantile(worst,0.10)),
      'p50_worst':float(np.quantile(worst,0.50)),
      'min_worst':float(np.min(worst)),
    }
    return m,hs,fr,bm

def score(m):
    # Fixed scalar merit function disclosed in the manuscript.
    return (m['positive_worst'] + 0.5*m['p10_worst'] + 0.08*m['infocus_mean']
            - 0.3*m['mean_axial_std'] - m['neg_fraction'])

def radial_average(img,fr,bins=np.linspace(0,0.9,91)):
    centers=0.5*(bins[:-1]+bins[1:])
    out=np.full_like(centers,np.nan,dtype=float)
    for i in range(len(centers)):
        b=(fr>=bins[i])&(fr<bins[i+1])
        if np.any(b): out[i]=np.mean(img[b])
    return centers,out

def masks_from_params():
    return {
      'Clear': (mask_clear, PARAMS['Clear']['sigma']),
      'Cubic': (mask_cubic(PARAMS['Cubic']['alpha']), PARAMS['Cubic']['sigma']),
      'Quartic': (mask_quartic(PARAMS['Quartic']['beta']), PARAMS['Quartic']['sigma']),
      'Proposed': (mask_ring(PARAMS['Proposed']['r1'],PARAMS['Proposed']['r2'],PARAMS['Proposed']['phase']), PARAMS['Proposed']['sigma'])
    }

def optimize_design_grid(N=96):
    def obj_ring(x):
        r1,r2,sigma=x
        if r2<=r1+0.05: return 10.0+(r1-r2)
        m,*_=evaluate(mask_ring(r1,r2),sigma,N=N)
        return -score(m)
    def obj_cubic(x):
        m,*_=evaluate(mask_cubic(x[0]),x[1],N=N)
        return -score(m)
    def obj_quartic(x):
        m,*_=evaluate(mask_quartic(x[0]),x[1],N=N)
        return -score(m)
    def obj_clear(s):
        m,*_=evaluate(mask_clear,float(s),N=N)
        return -score(m)
    rp=differential_evolution(obj_ring,[(0.1,0.8),(0.25,0.98),(0.15,1.0)],seed=SEED_PROP,maxiter=35,popsize=12,tol=1e-5,polish=True)
    rc=differential_evolution(obj_cubic,[(0,12),(0.15,1.0)],seed=SEED_CUBIC,maxiter=30,popsize=10,tol=1e-5,polish=True)
    rq=differential_evolution(obj_quartic,[(-12,12),(0.15,1.0)],seed=SEED_QUARTIC,maxiter=30,popsize=10,tol=1e-5,polish=True)
    r0=minimize_scalar(obj_clear,bounds=(0.15,1.0),method='bounded',options={'xatol':1e-5})
    return {
      'Proposed':{'r1':float(rp.x[0]),'r2':float(rp.x[1]),'sigma':float(rp.x[2]),'fun':float(rp.fun),'nfev':int(rp.nfev)},
      'Cubic':{'alpha':float(rc.x[0]),'sigma':float(rc.x[1]),'fun':float(rc.fun),'nfev':int(rc.nfev)},
      'Quartic':{'beta':float(rq.x[0]),'sigma':float(rq.x[1]),'fun':float(rq.fun),'nfev':int(rq.nfev)},
      'Clear':{'sigma':float(r0.x),'fun':float(r0.fun),'nfev':int(r0.nfev)}
    }

def make_fig1(outdir:Path):
    fig,ax=plt.subplots(figsize=(7.1,2.4))
    ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis('off')
    items=[(0.7,'Köhler\nsource'),(3.0,'Weak-amplitude\nobject'),(5.3,'Objective +\nphase pupil'),(8.2,'Sensor')]
    for x,label in items:
        if 'source' in label:
            ax.add_patch(Circle((x,1.5),0.55,fill=False,lw=1.5)); ax.add_patch(Circle((x,1.5),0.30,alpha=0.25))
        elif 'object' in label:
            ax.add_patch(Rectangle((x-0.55,0.9),1.1,1.2,fill=False,lw=1.5));
            for yy in np.linspace(1.05,1.95,5): ax.plot([x-0.4,x+0.4],[yy,yy],lw=0.8)
        elif 'phase' in label:
            ax.add_patch(Circle((x,1.5),0.60,lw=1.5)); ax.add_patch(Circle((x,1.5),0.52,fill=False,lw=1.2)); ax.add_patch(Circle((x,1.5),0.40,fill=False,lw=1.2))
        else:
            ax.add_patch(Rectangle((x-0.15,0.75),0.3,1.5,fill=False,lw=1.5))
        ax.text(x,0.35,label,ha='center',va='top',fontsize=9)
    for a,b in [(1.35,2.4),(3.65,4.65),(5.95,7.75)]:
        ax.add_patch(FancyArrowPatch((a,1.5),(b,1.5),arrowstyle='->',mutation_scale=12,lw=1.2))
    ax.text(0.7,2.45,r'$\sigma=\mathrm{NA}_{ill}/\mathrm{NA}$',ha='center',fontsize=9)
    ax.text(5.3,2.45,r'$\phi(r)=\pi$ for $r_1<r\leq r_2$',ha='center',fontsize=9)
    ax.text(5.3,2.15,r'$r_1=0.686,\ r_2=0.876$',ha='center',fontsize=9)
    fig.tight_layout(pad=0.5)
    fig.savefig(outdir/'Fig1_system_schematic.png',dpi=600,bbox_inches='tight')
    fig.savefig(outdir/'Fig1_system_schematic.pdf',bbox_inches='tight')
    plt.close(fig)

def make_fig2(outdir:Path):
    N=300; x=np.linspace(-1,1,N); X,Y=np.meshgrid(x,x); R=np.hypot(X,Y)
    phi=np.where((R>PARAMS['Proposed']['r1'])&(R<=PARAMS['Proposed']['r2']),np.pi,np.nan)
    base=np.where((R<=1)&~((R>PARAMS['Proposed']['r1'])&(R<=PARAMS['Proposed']['r2'])),0,np.nan)
    phase=np.where(np.isfinite(phi),phi,base)
    source=np.where(R<=PARAMS['Proposed']['sigma'],1,np.nan)
    fig,axs=plt.subplots(1,3,figsize=(7.1,2.55))
    im=axs[0].imshow(phase,extent=(-1,1,-1,1),origin='lower',vmin=0,vmax=np.pi,cmap='gray')
    axs[0].set_title('(a) pupil phase',fontsize=8); axs[0].set_xlabel('normalized pupil coordinate',fontsize=7.5); axs[0].set_ylabel('normalized pupil coordinate',fontsize=7.5)
    rr=np.linspace(0,1,1000); pp=np.where((rr>PARAMS['Proposed']['r1'])&(rr<=PARAMS['Proposed']['r2']),np.pi,0)
    axs[1].plot(rr,pp,lw=1.5); axs[1].axvline(PARAMS['Proposed']['r1'],ls='--',lw=0.9); axs[1].axvline(PARAMS['Proposed']['r2'],ls='--',lw=0.9)
    axs[1].set_ylim(-0.1,3.5); axs[1].set_xlabel('normalized radius',fontsize=7.5); axs[1].set_ylabel('phase (rad)',fontsize=7.5); axs[1].set_title('(b) radial phase profile',fontsize=8)
    axs[2].imshow(source,extent=(-1,1,-1,1),origin='lower',vmin=0,vmax=1,cmap='gray_r')
    axs[2].set_title('(c) circular source',fontsize=8); set_xlabel('normalized source coordinate',fontsize=7.5); axs[2].set_ylabel('normalized source coordinate',fontsize=7.5)
    for a in axs: a.tick_params(labelsize=6.5)
    fig.subplots_adjust(left=0.075,right=0.985,bottom=0.24,top=0.84,wspace=0.42)
    fig.savefig(outdir/'Fig2_design.png',dpi=600)
    fig.savefig(outdir/'Fig2_design.pdf')
    plt.close(fig)

def make_main_results(outdir:Path,datadir:Path,N=256):
    systems=masks_from_params(); summary=[]; radial_rows=[]; heat={}
    bins=np.linspace(0,0.9,91)
    for name,(mask,sig) in systems.items():
        m,hs,fr,bm=evaluate(mask,sig,N=N)
        summary.append({'Method':name,'sigma':sig,**m,'score':score(m)})
        # Radial-average each z and worst envelope of full 2D transfer.
        centers=None; radz=[]
        for z,h in zip(Z_UM,hs):
            centers,rv=radial_average(h,fr,bins)
            radz.append(rv)
            for fval,hval in zip(centers,rv): radial_rows.append({'Method':name,'z_um':z,'f_norm':fval,'H_A':hval})
        heat[name]=np.array(radz)
    sdf=pd.DataFrame(summary); sdf.to_csv(datadir/'Table1_summary_metrics.csv',index=False)
    pd.DataFrame(radial_rows).to_csv(datadir/'radial_transfer_curves.csv',index=False)
    # Fig 3: worst-case radial transfer curves calculated after radial averaging.
    fig,ax=plt.subplots(figsize=(5.7,3.5))
    styles=['-','--','-.',':']
    for (name,_),ls in zip(sys.items(),styles):
        zmat=heat[name]
        worst=np.nanmin(zmat,axis=0)
        ax.plot(centers,worst,label=name,ls=ls,lw=1.5)
    ax.axhline(0,lw=0.8)
    ax.set_xlim(0.1,0.85); ax.set_xlabel(r'normalized spatial frequency $|f|/f_c$'); ax.set_ylabel(r'worst-case signed transfer $H_{\rm worst}$')
    ax.legend(frameon=False,ncol=2,fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(outdir/'Fig3_worst_case_transfer.png',dpi=600,bbox_inches='tight'); fig.savefig(outdir/'Fig3_worst_case_transfer.pdf',bbox_inches='tight'); plt.close(fig)
    # Fig 4: heat maps of radial-average signed transfer.
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.2),sharex=True,sharey=True)
    for ax,(name,arr) in zip(axs.flat,heat.items()):
        im=ax.imshow(arr,aspect='auto',origin='lower',extent=(centers[0],centers[-1],Z_UM[0],Z_UM[-1]),vmin=-0.15,vmax=1.0,cmap='gray_r')
        ax.set_title(name,fontsize=9); ax.axvline(BAND[0],ls=':',lw=0.7); ax.axvline(BAND[1],ls=':',lw=0.7)
    for ax in axs[:,0]: ax.set_ylabel('defocus z (µm)')
    for ax in axs[-1,:]: ax.set_xlabel(r'normalized spatial frequency $|f|/f_c$')
    cbar=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=0.022,pad=0.03); cbar.set_label(r'radially averaged $H_A$')
    fig.subplots_adjust(left=0.10,right=0.91,bottom=0.10,top=0.93,wspace=0.12,hspace=0.20)
    fig.savefig(outdir/'Fig4_transfer_maps.png',dpi=600,bbox_inches='tight'); fig.savefig(outdir/'Fig4_transfer_maps.pdf',bbox_inches='tight'); plt.close(fig)
    return sdf

def make_sensitivity(outdir:Path,datadir:Path,N=192):
    r1=PARAMS['Proposed']['r1']; r2=PARAMS['Proposed']['r2']; sig=PARAMS['Proposed']['sigma']
    sweeps={
      'r1':np.linspace(r1-0.05,r1+0.05,11),
      'r2':np.linspace(r2-0.05,r2+0.05,11),
      'sigma':np.linspace(sig-0.10,sig+0.10,11),
      'phase_ratio':np.linspace(0.85,1.15,13)
    }
    rows=[]
    for key,vals in sweeps.items():
        for v in vals:
            if key=='r1': mask=mask_ring(v,r2); ss=sig
            elif key=='r2': mask=mask_ring(r1,v); ss=sig
            elif key=='sigma': mask=mask_ring(r1,r2); ss=v
            else: mask=mask_ring(r1,r2,np.pi*v); ss=sig
            m,*_=evaluate(mask,ss,N=N)
            rows.append({'parameter':key,'value':float(v),'mean_worst':m['mean_worst'],'p10_worst':m['p10_worst'],'neg_fraction':m['neg_fraction'],'mean_axial_std':m['mean_axial_std'],'infocus_mean':m['infocus_mean']})
    df=pd.DataFrame(rows); df.to_csv(datadir/'sensitivity.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(7.1,5.0))
    labels={'r1':r'$r_1$','r2':r'$r_2$','sigma':r'$\sigma$','phase_ratio':r'phase step / $\pi$'}
    for ax,key in zip(axs.flat,['r1','r2','sigma','phase_ratio']):
        d=df[df.parameter==key]
        ax.plot(d.value,d.mean_worst,marker='o',ms=3,lw=1.2,label='mean worst-case transfer')
        ax.plot(d.value,d.p10_worst,marker='s',ms=3,lw=1.0,ls='--',label='10th percentile')
        ax.set_xlabel(labels[key]); ax.set_ylabel('signed transfer'); ax.grid(alpha=0.2)
    axs[0,0].legend(frameon=False,fontsize=7)
    fig.tight_layout(); fig.savefig(outdir/'Fig5_sensitivity.png',dpi=600,bbox_inches='tight'); fig.savefig(outdir/'Fig5_sensitivity.pdf',bbox_inches='tight'); plt.close(fig)

def make_validation(outdir:Path,datadir:Path):
    r1=PARAMS['Proposed']['r1']; r2=PARAMS['Proposed']['r2']; sig=PARAMS['Proposed']['sigma']; mask=mask_ring(r1,r2)
    grows=[]
    for N in [128,192,256,384]:
        m,*_=evaluate(mask,sig,N=N)
        grows.append({'N':N,**m})
    gdf=pd.DataFrame(grows); gdf.to_csv(datadir/'grid_convergence.csv',index=False)
    drows=[]
    systems=masks_from_params()
    for D in [50,75,100,125,150,175]:
        wv=np.linspace(-D*1e-6,D*1e-6,9)*NA**2/(N_MED*LAMBDA0)
        for name,(mk,ss) in systems.items():
            m,*_=evaluate(mk,N=192,wvals=wv)
            drows.append({'half_range_um':D,'Method':name,**m})
    ddf=pd.DataFrame(drows); ddf.to_csv(datadir/'defocus_range_stress_test.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.0))
    axs[0].plot(gdf.N,gdf.mean_worst,marker='o'); axs[0].set_xlabel('grid size N'); axs[0].set_ylabel('mean worst-case transfer'); axs[0].grid(alpha=0.2)
    for name,ls in zip(systems.keys(),['-','--','-.',':']):
        d=ddf[ddf.Method==name]; axs[1].plot(d.half_range_um,d.mean_worst,marker='o',ms=3,ls=ls,label=name)
    axs[1].axvline(100,ls=':',lw=0.8); axs[1].set_xlabel('evaluated defocus half-range (µm)'); axs[1].set_ylabel('mean worst-case transfer'); axs[1].legend(frameon=False,fontsize=7); axs[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(outdir/'Fig6_validation_limits.png',dpi=600,bbox_inches='tight'); fig.savefig(outdir/'Fig6_validation_limits.pdf',bbox_inches='tight'); plt.close(fig)

def make_wavelength_robustness(datadir:Path,N=192):
    r1=PARAMS['Proposed']['r1']; r2=PARAMS['Proposed']['r2']; sig=PARAMS['Proposed']['sigma']; mask=mask_ring(r1,r2)
    rows=[]
    for wl_nm in [500,525,550,575,600]:
        wv=Z_M*NA**2/(N_MED*(wl_nm*1e-9))
        # Fixed physical phase-step thickness designed for pi at LAMBDA0;
        # refractive-index dispersion is neglected, so phase scales as lambda0/lambda.
        mask_wl=mask_ring(r1,r2,np.pi*(LAMBDA0/(wl_nm*1e-9)))
        m,*_=evaluate(mask_wl,sig,N=N,wvals=wv)
        rows.append({'wavelength_nm':wl_nm,**m})
    pd.DataFrame(rows).to_csv(datadir/'wavelength_sensitivity.csv',index=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--outdir',default=str(Path(__file__).resolve().parents[1])); p.add_argument('--optimize',action='store_true'); args=p.parse_args()
    root=Path(args.outdir); figdir=root/'figures'; datadir=root/'data'; figdir.mkdir(parents=True,exist_ok=True); datadir.mkdir(parents=True,exist_ok=True)
    meta={
      'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
      'lambda_nm':LAMBDA0*1e9,'NA':NA,'n_medium':N_MED,'z_um':Z_UM.tolist(),'normalized_frequency_band':list(BAND),'span':SPAN,
      'parameters':PARAMS,'random_seeds':{'proposed':SEED_PROP,'cubic':SEED_CUBIC,'quartic':SEED_QUARTIC}
    }
    (root/'software_environment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    make_fig1(figdir); make_fig2(figdir); summary=make_main_results(figdir,datadir); make_sensitivity(figdir,datadir); make_validation(figdir,datadir); make_wavelength_robustness(datadir)
    if args.optimize:
        opt=optimize_design_grid(96); (datadir/'optimization_reproduction.json').write_text(json.dumps(opt,indent=2),encoding='utf-8')
    print(summary.to_string(index=False))

if __name__=='__main__': main()
