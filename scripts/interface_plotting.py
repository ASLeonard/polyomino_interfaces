import sys
import numpy as np
if not np.any(['scripts' in pth for pth in sys.path]):
     sys.path.append('scripts/')
     sys.path.append('../polyomino_core/scripts')

from polyomino_visuals import VisualiseSingleShape as VSS
from interface_analysis import *
from interface_methods import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

from matplotlib.path import Path
from matplotlib.patches import Patch,Rectangle,PathPatch,ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pickle import load


from scipy.stats import linregress,binom,scoreatpercentile,t
from scipy.interpolate import splprep, splev
from itertools import combinations_with_replacement as cwr,product

from colorsys import hsv_to_rgb
from random import uniform,choice
from copy import deepcopy
from collections import defaultdict

def ZK(data):
     ax=plot2D(-2.5,2.5,250,True)
     c_grad_1 = LinearSegmentedColormap.from_list('grad_1', ['aqua','darkblue'], N=250)
     c_grad_2 = LinearSegmentedColormap.from_list('grad_2', ['mistyrose','firebrick'], N=250)
     
     phase_color=[]
     for _ in range(2):
          phase_color.extend(list(c_grad_1(np.linspace(0,1,250)))+list(c_grad_2(np.linspace(0,1,250))))


     
     for a in data:
          x=np.log10(a[:,0])
          y=np.log10(a[:,1])
          mag=np.sqrt(x**2+y**2)
          x_N=x.copy()
          y_N=y.copy()
          #x/=mag
          #y/=mag
          #print(mag[0])
          
          ax.RGB.quiver(x_N[:-1], y_N[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='width',scale=10, angles='xy',zorder=10,color=phase_color)
          ax.RGB.plot(x_N,y_N,alpha=0.5)
          #addArrows(ax.RGB,x,y,10)
     plt.show(block=False)

def addArrows(ax,x,y,num):
     for i in range(num):
          start_ind=0 if i==0 else int(len(x)//(num/i))
          print(start_ind)
          end_ind=start_ind+1
          ax.annotate('', xytext=(x[start_ind], y[start_ind]), xy=(x[end_ind], y[end_ind]), arrowprops=dict(arrowstyle="->"))

def gradient2D(low=-1,high=1,res=250):
     xx,yy=np.meshgrid(np.logspace(low,high,res),np.logspace(low,high,res))
     xxL,yyL=np.meshgrid(np.linspace(low,high,res),np.linspace(low,high,res))
     rgb=np.array(Twelve(xx,yy))
     ff,(axx,axxx)=plt.subplots(1,2)
     
     x=(rgb[1])
     vgrad = np.gradient(x)
     mag_grad=np.sqrt(vgrad[0]**2+vgrad[1]**2)#*np.sign(x)
     lw = 5*mag_grad / mag_grad.max()
     axxx.quiver(xxL,yyL,vgrad[1],vgrad[0])#,density=1,minlength=.25,linewidth=lw)
     axx.imshow(mag_grad,origin='lower',extent=[low,high]*2,norm=LogNorm(vmin=0.0001))
     #axxx.imshow(x,origin='lower',extent=[low,high]*2)
     #axxx.contour(x,20,colors='w',origin='lower',extent=[low,high]*2)
     plt.show(block=False)

     
def plot1D(low=-1,high=1,res=500,called=False):
     f,ax=plt.subplots()
     xs=np.logspace(low,high,res)
     plt.loglog(xs,FourOne(xs),'k')
     ax.set_xlabel(r'$\gamma$',fontsize=20)
     ax.set_ylabel('target phen fraction')
     if called:
          return ax
     plt.show(block=False)

def getBoundaries(maxes):
     change_points=defaultdict(list)
     for r_idx,row in enumerate(maxes):
          for change in [i for i in range(1,len(row)) if row[i]!=row[i-1]]:
               change_points[(min(row[change-1],row[change]),max(row[change-1],row[change]))].append((r_idx,change))
     return change_points


def plot2D(low=-1,high=1,res=1000,called=False):
     xx,yy=np.meshgrid(np.logspace(low,high,res),np.logspace(low,high,res))
     xxL,yyL=np.meshgrid(np.linspace(low,high,res),np.linspace(low,high,res))
     rgb=np.array(Twelve(xx,yy))
     fig = plt.figure()
     ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
     
     ax.imshow_rgb(*rgb[:,:],interpolation='none',origin='lower',extent=[low,high]*2)
     ax.RGB.contour(rgb[2], levels=np.linspace(.5,.95,10), colors='gray',linewidths=.5,alpha=1, origin='lower',extent=[low,high]*2)
     #cs=ax.RGB.contour(np.argmax(rgb,axis=0),levels=[0,1,2],colors='y',hatches=['+','////','o'],origin='lower',extent=[low,high]*2)
     boundaries=getBoundaries(np.argmax(rgb,axis=0))
     for bound in boundaries.values():
          ax.RGB.plot([xxL[b] for b in bound],[yyL[b] for b in bound],'w',ls=':',lw=3)
          

     
     for i,(channel,ax_c) in enumerate(zip(rgb,[ax.R,ax.G,ax.B])):
          CS=ax_c.contour(channel, levels=10, colors='w',linewidths=.5, origin='lower',extent=[low,high]*2)
          #ax_c.clabel(CS, inline=1, fontsize=10)
     ax.RGB.set_xlabel(r'$\log_{{10}}{\alpha}$',fontsize=20)
     ax.RGB.set_ylabel(r'$\log_{{10}}{\beta}$',fontsize=20)
     if called:
          return ax
     plt.show(block=False)        
     
"""PHYLOGENCY SECTION """
class EvoData(object):
     __slots__ = ('S_star','T','mu','gamma','transitions','N_runs','evo_strs','evo_samps','jumps')
     def __init__(self,S_star,T,mu,gamma,data):
          self.S_star=S_star
          self.T=T
          self.mu=mu
          self.gamma=gamma
          self.N_runs,self.transitions,self.jumps,self.evo_strs,self.evo_samps=data
          
     def __repr__(self):
          return r'Data struct for S*={:.3f},T={:.3g},mu={:.2g},gamma={:.2g}'.format(self.S_star,self.T,self.mu,self.gamma)
     
def loadPickle(S_star,t,mu,gamma):
     with open('/rscratch/asl47/Pickles/Y{}T{}Mu{}F{}.pkl'.format(S_star,*(float(i) for i in (t,mu,gamma))), 'rb') as f:
          return EvoData(S_star,t,mu,gamma,load(f))
def loadNPZ(S_star,t,mu,gamma):
     return np.load('/rscratch/asl47/Pickles/Y{}T{}Mu{}F{}.npz'.format(S_star,*(float(i) for i in (t,mu,gamma))), 'rb')['arr_0']
     
def plotExp(evo_data_struct):
     evo_data=evo_data_struct.evo_strs
     evo_data_samp=evo_data_struct.evo_samps
     S_star=evo_data_struct.S_star
     mu=evo_data_struct.mu
     
     phen_cols={(1,0):'black',(2,0):'firebrick',(4,0):'olivedrab',(4,1):'darkblue',(8,0):'chocolate',(3,0):'goldenrod',(12,0):'k',(16,0):'k',(12,1):'chartreuse'}
     ancestors={(2,0):{1:(1,0)}, (4,0):{2:(1,0)}, (4,1):{4:(2,0),3:(4,0)}, (8,0):{1:(4,0),2:(2,0)}, (12,0): {2:(4,1),4:(8,0)}, (16,0): {2:(4,1),4:(8,0)}}
     
     bond_marks={1:'o',2:'s',3:'v',4:'^'}
     bond_ls={1:'-',2:':',3:'-.',4:'--'}
     c_ls={1:'c',2:'m',3:'w.',4:'k'}
     N_markers=10
     MAX_G=200
     
     f,axarr=plt.subplots(2,3,sharey=True)
     ax_d={(4,0):axarr[0,0],(4,1):axarr[0,1],(12,0):axarr[0,2],(2,0):axarr[1,0],(8,0):axarr[1,1],(16,0):axarr[1,2]}
     null_exp=RandomWalk(64,None,.5,S_star,True)
     
     for phen,data in evo_data.items():
          plt.setp(ax_d[phen].spines.values(), color=phen_cols[phen],lw=(2 if phen[0]<10 else 1))
          if phen==(3,0) or phen==(12,1):
               continue
          #ax_d[phen].axhline(null_exp,c='k',ls=':')
          ax_d[phen].axhline(S_star,c='k',ls='-',lw=0.75)
          max_len=min(200,max(len(vals) for vals in data.values()))
          ax_d[phen].plot(range(max_len+1),RandomWalk(64,max_len,mu*1./6,S_star,False),'r--')

          for (bond,new),strs in data.items():
               if new not in ancestors[phen]:
                    continue
               strs=strs[:MAX_G]
               ax_d[phen].plot(range(len(strs)),strs,c=phen_cols[ancestors[phen][new]],marker=bond_marks[bond],markevery=max(1,len(strs)//N_markers),zorder=10,lw=.5)
               if evo_data_samp is not None:
                   for samp in evo_data_samp[phen][(bond,new)]:
                        samp=samp[:MAX_G]
                        ax_d[phen].plot(range(len(samp)),samp,c=phen_cols[ancestors[phen][new]],alpha=0.15,lw=.5)
               VSS(PhenTable()[phen],ax_d[phen],(.4,.95),.2)
               
     #axarr[0,0].set_ylim((0.71,.87))
     f.tight_layout()
     axarr[1,1].set_xlabel('generations')
     axarr[1,0].set_ylabel(r'$\langle \hat{S} \rangle$')
     plt.show(block=False)

def plotPhaseSpace(evo_data,low=-2,high=2,res=250):
     evo_strs=evo_data.evo_strs
     S_star=evo_data.S_star
     T=evo_data.T
     ax_1d=plot1D(low,high,res*4,True)
     ax_2d=plot2D(low,high,res,True)

     sx=[defaultdict(dict) for _ in range(10)]
     for k,v in evo_data.evo_samps.items():
          for k2,v2 in v.items():
               for i,v3 in enumerate(v2):
                    sx[i][k][k2]=v3
     
     
               
     phase_parameters=calcTransitionParams(evo_strs,evo_data.transitions,T,S_star)
     phen_mark={(2,0):'P',(4,0):'D',(4,1):'s',(8,0):'o',(16,0):'X',(12,0):'*'}
     jj=0
     indv_params=[]
     for sampx in sx:
          try:
               indv_params.append(calcTransitionParams(sampx,evo_data.transitions,T,S_star))
          except:
               print("incomplete data for indv params")
     
     for ctp in filter(None,sx):
          continue
          for (pure,phen,phen_source),phase_p in ctp.items():
               if pure:
                    continue
               ##print(jj)
               jj+=1
               fc='none' if pure else 'aqua'
               ZZ=10
               hatch='' #if pure else '////'
               m_size=100 if phen_source==(12,0) else 75
               if phen==(12,0):                    
                    ec='w' #if pure else 'k'
                    ax_2d.RGB.scatter(*np.log10(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=ec,s=m_size,lw=2,hatch=hatch,zorder=ZZ)
                    if phen_source==(8,0) and pure:
                         print(np.log10(phase_p))
               else:
                    col='firebrick' if phen==(16,0) else 'forestgreen'
                    pattern='////' if phen==(16,0) else ''
                    if fc=='aqua':
                         fc= 'orchid' if phen==(16,0) else fc
                    ax_1d.scatter(phase_p,FourOne(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=col,s=m_size*2,zorder=ZZ,hatch=pattern)
               
     for (pure,phen,phen_source),phase_p in phase_parameters.items():
          fc='none' if pure else 'aqua'
          ZZ=10
          hatch='' #if pure else '////'
          m_size=500 if phen_source==(12,0) else 120
          if phen==(12,0):                    
               ec='w' #if pure else 'k'
               ax_2d.RGB.scatter(*np.log10(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=ec,s=m_size,label=phen_source,lw=2,hatch=hatch,zorder=ZZ)
          else:
               col='firebrick' if phen==(16,0) else 'forestgreen'
               pattern='////' if phen==(16,0) else ''
               if fc=='aqua':
                    fc= 'orchid' if phen==(16,0) else fc
               ax_1d.scatter(phase_p,FourOne(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=col,s=m_size*2,label=phen_source,zorder=ZZ,hatch=pattern)
     ax_2d.RGB.legend()
     ax_1d.legend()
     plt.show(block=False)

def plotPathways(evo_data,norm_style=''):
     evo_trans=evo_data.transitions
     failed=evo_data.jumps
     f, ax1= plt.subplots(1,1)
     #ax1.set_title(r'S*={}, T={},$\gamma$={}'.format(evo_data.S_star,evo_data.T,evo_data.gamma))
     
     total_paths=float(evo_data.N_runs)

     print('Total runs: {}'.format(total_paths))
     label_coords={((2,0),(1,0)):(.5,-.5),
                   ((4,0),(1,0)):(.5,.5),
          
                   ((4,1),(2,0)):(1.75,.5),
                   ((4,1),(4,0)):(1.5,1),
                   ((8,0),(2,0)):(1.5,-1),
                   ((8,0),(4,0)):(1.75,-.5),
                   
                   ((12,0),(4,1)):(2.5,1),
                   ((12,0),(8,0)):(2.75,.5),
                   ((16,0),(4,1)):(2.75,-.5),
                   ((16,0),(8,0)):(2.5,-1)}

     phen_paths={(2,0):[(1,0)],(4,0):[(1,0)],(8,0):[(2,0),(4,0)],(4,1):[(2,0),(4,0)],(12,0):[(4,1),(8,0)],(16,0):[(4,1),(8,0)]}
     normalising=defaultdict(float)
     normalising[(1,0)]=total_paths
     d={}
     for phen_to, phen_froms in phen_paths.items():
          for phen_from in phen_froms:
               try:
                    d[(phen_to,phen_from)]=evo_trans[phen_to][phen_from]/(evo_trans[phen_to][phen_from]+failed[phen_to][phen_from])
               except:
                    if phen_to in evo_trans and phen_from in evo_trans[phen_to]:
                         d[(phen_to,phen_from)]=1
                    else:
                         continue
               ax1.text(*label_coords[(phen_to,phen_from)],s='{:.3f}'.format( d[(phen_to,phen_from)]),va='top',ha='right',color='r')
               
               
     for phen_in, phen_set in evo_trans.items():
          for phen_out,count in phen_set.items():
               if norm_style=='out':
                    normalising[phen_out]+=phen_set[phen_out]
               elif norm_style=='in':
                    normalising[phen_in]+=phen_set[phen_out]
               else:
                    normalising[phen_in]+=phen_set[phen_out]

     
     for phen_in, phen_set in evo_trans.items():
          for phen_out,count in phen_set.items():
               display_str=str(count)
               if norm_style=='total':
                    display_str='{:.3f}'.format(count/total_paths)
               elif norm_style=='out':
                    display_str='{:.3f}'.format(count/normalising[phen_in])
               elif norm_style=='in':
                    display_str='{:.3f}'.format(count/normalising[phen_out])
               elif norm_style=='full':
                    display_str='{}/{}'.format(count,int(normalising[phen_out]))
               elif norm_style=='prop':
                    display_str='{:.3f}'.format(count/failed[phen_in][phen_out])
               ax1.text(*label_coords[(phen_in,phen_out)],s=display_str,ha='left',va='bottom')
               
     for phen_in, phen_set in failed.items():
          for phen_out,count in phen_set.items():
               display_str=str(count)
               if norm_style=='ain':
                    display_str='{:.3f}'.format(count/normalising[phen_out])
               #ax1.text(*label_coords[(phen_in,phen_out)],s=display_str,va='top',ha='right',color='r')

     for coords in [((0,0),(1,1)),((0,0),(1,-1)),((1,1),(2,1)),((1,1),(2,-1)),((1,-1),(2,1)),((1,-1),(2,-1)),((2,1),(3,1)),((2,1),(3,-1)),((2,-1),(3,1)),((2,-1),(3,-1))]:
          con = ConnectionPatch(*coords, coordsA='data',coordsB='data', arrowstyle="-|>", shrinkA=5, shrinkB=5,mutation_scale=20, fc="w")
          ax1.add_artist(con)

     VSS(PhenTable()[(1,0)],ax1,(.02,.55),.1)
     VSS(PhenTable()[(2,0)],ax1,(.3,.15),.1)
     VSS(PhenTable()[(4,0)],ax1,(.3,.95),.1)
     VSS(PhenTable()[(8,0)],ax1,(.55,.15),.1)
     VSS(PhenTable()[(4,1)],ax1,(.55,.95),.1)
     VSS(PhenTable()[(16,0)],ax1,(.8,.15),.1)
     VSS(PhenTable()[(12,0)],ax1,(.8,.95),.1)
          
     ax1.set_xlim((-.5,3.5))
     ax1.set_ylim((-1.5,1.5))
     ax1.axis('off')
     plt.show(block=False)


    
def bootstrap(data, n_boot=10000, ci=68):
     boot_dist = []
     for i in range(int(n_boot)):
          resampler = np.random.randint(0, data.shape[0], data.shape[0])
          sample = data.take(resampler, axis=0)
          sample=sample.astype(float)
          boot_dist.append(np.nanmean(sample, axis=0))
  
     b= np.array(boot_dist)
     s1 = np.apply_along_axis(scoreatpercentile, 0, b, 50.-ci/2.)
     s2 = np.apply_along_axis(scoreatpercentile, 0, b, 50.+ci/2.)
     return (s1,s2)
    
def tsplotboot(ax,data,title='',**kw):
     gen_start=data[:,0]
     data=data[:,1:]

     x = np.arange(data.shape[1])
     est = np.nanmean(data, axis=0)
     cis = bootstrap(data,500)
     #ax.fill_between(x,*cis,alpha=0.3,color='k', **kw)
     c='g' if title=='I' else 'k'
     #ax.plot(x,est,c=c,lw=2.5,zorder=5)

     N_samples=min(data.shape[0],15)
     for selection in np.random.choice(data.shape[0],N_samples,replace=False):
          ax.plot(x+gen_start[selection],data[selection],alpha=0.5,c=c)
     ax.margins(x=0)
               
     ax.set_ylabel(r'$\langle \hat{S} \rangle$')
     if title=='S':
          ax.text(.95,.95,'symmetric',ha='right',va='top',transform=ax.transAxes)
     elif title=='E':
          ax.plot([], [], color=c, label='external')
          ax.text(.95,.95,'asymmetric',ha='right',va='top',transform=ax.transAxes)
     else:
          ax.plot([], [], color=c, label='internal')
     #if title!='':
     #     ax.set_title(title)
     ax.set_ylim([0.6875,.825])
     plt.show(block=False)


def plotData(data,I_size,S_star,mu=1,g_size=12,force_start=False):
     mu=float(mu)
     fig,axs = plt.subplots(2,sharex=True,sharey=True)
     for (k,v) in data.items():
          ax=axs[0] if k=='S' else axs[1]
          print(k,len(v))
          if type(v) is not list and v.size:
               v=np.array(v)
               
               if force_start:
                    v=v[v[:,0]==S_star]
               #v=v[:,:1000]
               
               tsplotboot(ax,v,k)
               if k=='I':
                    continue
 
               gen_length=v.shape[1]
               co_factor=2 if 'S' in k else 1
               step_length=1
               N_steps=int(np.ceil(gen_length/float(step_length)))
               pgs=RandomWalk(I_size/co_factor,N_steps,mu/(co_factor*6),S_star,1,1)
               ax.plot(range(0,(N_steps+1)*step_length,step_length),pgs[:-1],ls='--',lw=1.5,c='royalblue',zorder=10)

     axs[0].xaxis.set_ticks_position('none') 
     plt.xlabel(r'$\tau_{D}$',fontsize=24)
     plt.tight_layout(pad=0)
     
     #fig.suptitle(r'$l_I = %i , S^* = %.2f$' % (I_size,S_star))
     axs[1].legend(loc='upper left')
     plt.show(block=False)

         

""" Transition plots"""

faded_lines_alpha=0.3

def plotTransitionsDetailed(pt):
     phen_map_SIZE=defaultdict(dict)
     phen_map_SIZE[1][(1,1,1)]=None
     connection_subsets=defaultdict(list)
     for phen in pt.keys():
          if phen==(1,1,1):
               continue
          for sub_phen in pt[phen].keys():
               connection_subsets[phen].append((phen,sub_phen))

     for k,v in pt.items():
          phen_map_SIZE[np.count_nonzero(k[2:])][k]=v
     counts_map_X={i:len(phen_map_SIZE[v]) for i,v in enumerate(sorted(phen_map_SIZE.keys()))}
     phen_map_SIZE=dict(phen_map_SIZE)
     
     fig,ax = plt.subplots(1)
     connection_dict={}

     phen_dict={(1,1,1):AddPhenotypePatch(ax,(1,1,1),(0,0))}
     
     for i,c in enumerate(sorted(phen_map_SIZE.keys())[1:],1):
          offset=0 if len(phen_map_SIZE[c])%2==1 else .5
          for j,phen in enumerate(sorted(phen_map_SIZE[c])):
               valid_transition=False
               total_weight=sum(phen_map_SIZE[c][phen].values())
                    
               for connector,weight in phen_map_SIZE[c][phen].items():
                    con_size=np.count_nonzero(connector[2:])-1
                    if (con_size+1) not in phen_map_SIZE or connector not in sorted(phen_map_SIZE[con_size+1]):
                         continue
                    valid_transition=True
                    
                    offset2=0 if len(phen_map_SIZE[con_size+1])%2==1 else .5
                    con_y=len(phen_map_SIZE[con_size+1])/2-sorted(phen_map_SIZE[con_size+1]).index(connector)-offset2
                    con_x=sorted(phen_map_SIZE.keys()).index(con_size+1)
                    
                    spline_points=np.array([[con_x+.25,con_y],[con_x+.3,con_y],[i-.3,len(phen_map_SIZE[c])/2-j-offset],[i-.25,len(phen_map_SIZE[c])/2-j-offset]])
                    dx_f=i-con_x
                    rev_tran=dx_f<0
                    dy_f=spline_points[1,1]-spline_points[0,1]
                    dx=(1 if not rev_tran else -1)
                    dy= np.sign(dy_f) if abs(dy_f)>=1 else 0

                    ##special case of "reverse" transition
                    if rev_tran:
                         dx_f=abs(dx_f)
                         spline_points=np.insert(spline_points,2,[con_x,con_y+(0.5 if dy_f>0 else -0.5)],axis=0)
                         dy_f=spline_points[2,1]-spline_points[1,1]
                    steps=1
                    while steps<dx_f:
                         if int(spline_points[steps+(1 if rev_tran else 0),1]*2)%2==counts_map_X[con_x+dx]%2:
                              bump_factor= 0
                         else:
                              bump_factor=.5 if np.sign(spline_points[-1,1]-spline_points[steps+(1 if rev_tran<0 else 0),1])>0 else -.5
                         
                         adjustment_factor=dy+bump_factor
                         if abs(adjustment_factor)>1:
                              adjustment_factor=np.sign(adjustment_factor)*(adjustment_factor%1)
                         spline_points=np.insert(spline_points,steps+(2 if rev_tran else 1),[con_x+dx,spline_points[steps+(1 if rev_tran else 0),1]+adjustment_factor],axis=0)
                         
                         steps+=1
                         dx+=(1 if not rev_tran else -1)
                         dy=dy-(np.sign(dy_f)) if abs(dy)>=1 else 0
                         
                    if rev_tran:     
                         spline_points=np.insert(spline_points,-2,[i,len(phen_map_SIZE[c])/2-j-offset+(.5 if (spline_points[-3,1]-len(phen_map_SIZE[c])/2-j-offset)>0 else-.5)],axis=0)

                    connection_dict[(phen,connector)]=AddConnectionPatch(ax,spline_points,float(weight)/total_weight)

               if valid_transition:
                    phen_dict[phen]=AddPhenotypePatch(ax,phen,(i,len(phen_map_SIZE[c])/2 - j -offset))
        
                    
     ax.set_aspect(1)
     ax.relim()
     ax.autoscale_view()
     ax.grid(False)
     plt.axis('off')

     prev_artists_phens=[]
     prev_artists_lines=[]
     
     def onpick(event):
          coords=event.artist.get_bbox()
          mean_click=np.mean(coords.get_points(),axis=0)
          patch_coord=[int(np.round(mean_click[0])),np.round(mean_click[1]*2)/2.]

          vertical_index=int(counts_map_X[patch_coord[0]]/2-(0 if counts_map_X[patch_coord[0]]%2==1 else .5)-patch_coord[1])
          phen_slices=phen_map_SIZE[sorted(phen_map_SIZE.keys())[patch_coord[0]]]
          phen_key=sorted(phen_slices.keys())[vertical_index]
  
          #! reset colours !#
          for artist in prev_artists_phens:
               artist.set_alpha(.1)
          for artist in prev_artists_lines:
               artist.set_alpha(faded_lines_alpha)
               artist.set_color('gray')
          prev_artists_lines[:] = []
          prev_artists_phens[:] = []

          phen_set={phen_key}
          for phen_pairing, artists in connection_dict.items():
               if phen_key in phen_pairing:
                    phen_set.update(phen_pairing)
                    
                    for artist in artists: #! lines only !#
                         artist.set_alpha(1)
                         artist.set_color('r' if phen_key==phen_pairing[0] else 'b')
                         prev_artists_lines.append(artist)
                         
          for phen in phen_set:
               for artist in phen_dict[phen]:
                    artist.set_alpha(1 if phen==phen_key else 0.4)
                    prev_artists_phens.append(artist)
                    
          fig.canvas.draw() 
          return True

     fig.canvas.mpl_connect('pick_event', onpick)
     plt.show(block=False)




def AddPhenotypePatch(ax,shape,xy):
     ar_offsets={0:(0,-.25,0,.25),1:(-.25,0,.25,0),2:(0,.25,0,-.25),3:(.25,0,-.25,0)}
     cols=['darkgreen','royalblue','firebrick','goldenrod','mediumorchid']
     artists=[]
     dx,dy=shape[:2]
     scale=.5/max(shape[:2])
     
     for i,j in product(range(dx),range(dy)):
          if(shape[2+i+j*dx]):
               new_x=xy[0]+(i-dx/2.)*scale
               new_y=xy[1]+(dy/2.-j)*scale-(1.*scale)
               
               artists.append(ax.add_patch(Rectangle((new_x,new_y), scale, scale, fc=cols[(shape[2+i+j*dx]-1)/4],ec='k',fill=True,lw=2,picker=True,alpha=0.1)))
               artists.append(ax.arrow(*(np.array([new_x/scale+.5,new_y/scale+.5,0,0])+ar_offsets[(shape[2+i+j*dx]-1)%4])*scale, head_width=0.075*scale, head_length=0.15*scale, fc='k', ec='k',alpha=0.1))
               
     return artists


def AddConnectionPatch(ax,pts,weight):
     tck, u = splprep(pts.T, u=None, s=0.0,k=3, per=False)
     samples=100
     u_new = np.linspace(u.min(), u.max(), samples)
     x_new, y_new = splev(u_new, tck, der=0)

     ar=ax.arrow(x_new[samples/2],y_new[samples/2],np.diff(x_new[samples/2:samples/2+2])[0],np.diff(y_new[samples/2:samples/2+2])[0], shape='full', lw=0, length_includes_head=True, head_width=.075,alpha=faded_lines_alpha,color='gray')
     ln=ax.plot(x_new, y_new,c='gray', ls='--',lw=weight*2,alpha=faded_lines_alpha)[0]
     
     return (ln,ar)

def main():
     plotTransitionsDetailed(getD())
     return 

if __name__ == "__main__":
    main()

    
def MutualExclusion(n,S_c,L_I=64):
     return (binom(L_I/2,.5).cdf(int(round(S_c*L_I/2))-1)**n)*(binom(L_I,.5).cdf(int(S_c*L_I)-1)**(n*(n-1)/2.))

def plotExclusion(S_c,col='orangered'):
     xs=np.linspace(1,500,500)
     mut=MutualExclusion(xs,S_c)
     plt.plot(xs,mut,c=col,marker='h',ls='')
     #plt.plot(xs[:-1],-np.diff(mut),c='royalblue')
     #print -np.diff(mut),sum(-np.diff(mut))
     plt.show(block=False)
     
"""RANDOM THEORY SECTION """
    

def plotInterfaceProbability(l_I,l_g,Nsamps=False):

     def SF_sym(S_stars):
          return binom(l_I/2,.5).sf(np.ceil(l_I/2*S_stars)-1)#*(1./(l_g+1))
     def SF_asym(S_stars):
          return binom(l_I,.5).sf(np.ceil(l_I*S_stars)-1)#-sym(S_stars))/2*((l_g-1.)/(l_g+1))

     def sym_factor(A):
          return float(2)/(A+1)
     def asym_factor(A):
          return float(A-1)/(A+1)

     s_hats=np.linspace(0,1,l_I+1)

     fig, ax1 = plt.subplots()
     ax1.plot(s_hats[::2],np.log10(sym_factor(l_g)*SF_sym(s_hats[::2])),ls='',marker='^',c='royalblue')
     ax1.plot(s_hats,np.log10(asym_factor(l_g)*SF_asym(s_hats)),ls='',marker='o',c='firebrick')

     ax2 = ax1.twinx()
     
     ratios=np.log10((sym_factor(l_g)*SF_sym(s_hats))/(asym_factor(l_g)*SF_asym(s_hats)))
     ax2.plot(s_hats,ratios,c='darkseagreen',ls='',marker='h')
     crossover=np.where(ratios>0)[0][0]
     #ax2.axvline(s_hats[crossover],color='k',ls='--')
     #ax2.axhline(color='k',ls='-',lw=0.2)
     
     Is={8:np.uint8,16:np.uint16,32:np.uint32,64:np.uint64}
     if Nsamps:
          set_length(l_I)
          s_m=np.zeros(l_I+1)
          a_m=np.zeros(l_I+1)
          for _ in range(Nsamps):
               indices=choice(list(cwr(range(l_g),2)))
               if indices[0]!=indices[1]:
                    bases=np.random.randint(0,np.iinfo(Is[l_I]).max,dtype=Is[l_I],size=2)
                    
                    a_m[np.where(BindingStrength(*bases)>=s_hats)]+=1
               else:
                    base=np.random.randint(0,np.iinfo(Is[l_I]).max,dtype=Is[l_I])
                    s_m[np.where(BindingStrength(base,base)>=s_hats)]+=1
          s_m2=np.ma.log10(s_m/Nsamps)
          a_m2=np.ma.log10(a_m/Nsamps)
          ax1.plot(s_hats[::2],s_m2[::2],ls='--',c='royalblue')
          ax1.plot(s_hats,a_m2,ls='--',c='firebrick')
     

     crossover_height=np.log10(asym_factor(l_g)*SF_asym(1))/2.
     #ax1.text(crossover/float(l_I),crossover_height,'crossover',ha='right',va='center',rotation=90)
     scale_factor=np.log10(asym_factor(l_g)*SF_asym(s_hats))[0]-np.log10(asym_factor(l_g)*SF_asym(s_hats))[-1]
     ax1.text(.2,np.log10(sym_factor(l_g)*SF_sym(.2))-scale_factor*0.03,'symmetric',va='top')
     ax1.text(.2,np.log10(asym_factor(l_g)*SF_asym(.2)+scale_factor*0.05),'asymmetric',va='bottom')
     
     ax2.text(.1,(ratios[-1]-ratios[0])*.015+ratios[0],'ratio',ha='center',va='bottom')

     ax1.set_ylabel(r'$  \log  Pr $')
     ax2.set_ylabel(r'$\log \mathrm{ratio}$')
     ax1.set_xlabel(r'$\hat{S}$')

     ax1.spines['top'].set_visible(False)
     ax2.spines['top'].set_visible(False)
     
     plt.show(block=False) 

def PhenTable():
     return {(1,0):(1,1,1),(2,0):(2,1,1,5),(4,0):(2,2,1,2,4,3),(4,1):(2,2,1,2,4,5),(8,0):(4,4,0,0,1,0,4,5,6,0,0,8,7,2,0,3,0,0),(12,0):(4,4,0,1,2,0,1,5,6,2,4,8,7,3,0,4,3,0),(16,0):(4,4,1,2,1,2,4,5,6,3,1,8,7,2,4,3,4,3),(10,0):(4,3,0,1,2,0,1,5,6,2,4,3,7,3)}
