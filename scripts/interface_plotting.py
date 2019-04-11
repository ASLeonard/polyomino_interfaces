import sys
import numpy as np

#add local paths to load custom methods
if not any(('scripts' in pth for pth in sys.path)):
     sys.path.append('scripts/')
     sys.path.append('../polyomino_core/scripts')

from polyomino_visuals import VisualiseSingleShape as VSS
from interface_analysis import *
from interface_methods import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

from matplotlib.patches import Rectangle,PathPatch,ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pickle import load

from itertools import product
from collections import defaultdict



def plotParamGrid(S_hat,jump,Ts,Ys):
     _,axarr=plt.subplots(len(Ys),len(Ts),sharex=True,sharey=True)

     #plotting parameters
     bc={2:'chocolate',3:'royalblue',4:'forestgreen'}
     mc={2:'o',3:'v',4:'^'}
     mco={2:0,3:16,4:32}

     #truncate generations
     MAX_G=250

     #iterate over all temperature, nondetermism combinations
     for (gamma,T),ax in zip(product(Ys,Ts),axarr.flatten()):

          #load the relevant data
          d=loadPickle(S_hat,T,jump,gamma)
          ax.axhline(S_hat,c='k',ls='-',lw=0.75)
          ax.plot(range(MAX_G),RandomWalk(64,MAX_G-1,1./6,.671875,False),ls='--',c='k',zorder=20)
          if (16,0) not in d.evo_strs:
               continue
          for (bond,_),strs in d.evo_strs[(16,0)].items():
               
               if len(strs)!=MAX_G:
                    print(T,gamma,bond,len(strs))
                    continue
               ax.plot(range(MAX_G),strs,c=bc[bond],marker=mc[bond],markevery=(mco[bond],MAX_G//5))
     
     plt.show(block=False)
     
def BK(data):
     b2=np.log10(data[:,200:,:])
     b2=(b2.reshape(-1,2))

     sns.jointplot(x=b2[0],y=b2[0], kind="hex", color="k",joint_kws={'gridsize':100, 'bins':'log'})
     plt.show(block=False)

     
def ZK(data,period=100):
     ax=plot2D(-2.5,2.5,250,True)


     c_grad_1 = LinearSegmentedColormap.from_list('grad_1', ['aqua','darkblue'], N=period)
     c_grad_2 = LinearSegmentedColormap.from_list('grad_2', ['mistyrose','firebrick'], N=period)
     max_gen=data.shape[1]
     
     phase_color=[]
     for _ in range(max_gen//period//2):
          phase_color.extend(list(c_grad_1(np.linspace(0,1,period)))+list(c_grad_2(np.linspace(0,1,period))))

     q=data.copy()
     (x1,y1)=np.log10(np.mean(q[:,200:,:],axis=0)).T

     #ax.RGB.plot(x1,y1,c='indigo',lw=5)

     x2=x1[100:300]
     y2=y1[100:300]
     ax.RGB.quiver(x2[:-1], y2[:-1], x2[1:]-x2[:-1], y2[1:]-y2[:-1], scale_units='width',scale=5, angles='xy',zorder=10,color=phase_color)

     for sample in np.random.randint(0,100,20):
          (xs,ys)=np.log10(q[sample,0:50,:]).T
          (xL,yL)=np.log10(q[sample,500:700,:]).T
          ax.RGB.quiver(xs[:-1], ys[:-1], xs[1:]-xs[:-1], ys[1:]-ys[:-1], scale_units='width',scale=20, angles='xy',zorder=10,color='dimgrey',minlength=0,alpha=0.5)
          ax.RGB.plot(xs,ys,ls=':',c='k',alpha=0.5)

          for i in range(0,200,100):
               (ax.G if i==0 else ax.B).quiver(xL[i:i+100-1], yL[i:i+100-1], xL[i+1:i+100]-xL[i:i+100-1], yL[i+1:i+100]-yL[i:i+100-1], scale_units='width',scale=20, angles='xy',zorder=10,color=phase_color[i:i+100])
                    #ax.G.quiver(xL[i:i+100-1], yL[i:i+100-1], xL[i+1:i+100]-xL[i:i+100-1], yL[i+1:i+100]-yL[i:i+100-1], scale_units='width',scale=20, angles='xy',zorder=10,color=phase_color)
               
          

     for i in range(0,800,100):
          if i%200!=0:
               ax.G.plot(x1[i:i+100],y1[i:i+100],alpha=0.5,c='aqua')
          else:
               ax.B.plot(x1[i:i+100],y1[i:i+100],alpha=0.5,c='mistyrose')

                    
     
     plt.show(block=False)
     return

     
     for a in data:
          (x,y)=np.log10(a.T)


          
          mag=np.sqrt(x**2+y**2)
          x_N=x.copy()
          y_N=y.copy()
          #x/=mag
          #y/=mag
          #print(mag[0])
          q=0
          qq=100

          for i in range(0,200,100):
               ax.RGB.plot(x_N[i+q:i+qq],y_N[i+q:i+qq],alpha=0.5,c='k',ls=':')
          
          for i in range(200,1000,100):
               if i%200!=0:
                    ax.G.plot(x_N[i+q:i+qq],y_N[i+q:i+qq],alpha=0.5,c='y')
               else:
                    ax.B.plot(x_N[i+q:i+qq],y_N[i+q:i+qq],alpha=0.5,c='c')
          
          ax.RGB.quiver(x_N[:-1], y_N[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='width',scale=10, angles='xy',zorder=10,color=phase_color)
          #ax.RGB.plot(x_N,y_N,alpha=0.5)
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
     plt.plot(np.log10(xs),FourOne(xs),'k')
     ax.set_yticks([.5,.75,1])
     ax.set_yticklabels([r'$50\%$',r'$75\%$',r'$100\%$'])
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
     #ax.contour(rgb[2], levels=np.linspace(.5,.99,20), colors='lightgray',linewidths=.5,alpha=1, origin='lower',extent=[low,high]*2)

     #ax.contour(rgb[0]*(np.argmax(rgb,axis=0)==0), levels=np.linspace(.44,.5,10), colors='lightgray',linewidths=.5,alpha=1, origin='lower',extent=[low,high]*2)

     #ax.contour(rgb[1], levels=np.linspace(.5,.95,10), colors='lightgray',linewidths=.5,alpha=1, origin='lower',extent=[low,high]*2)

     
     #cs=ax.RGB.contour(np.argmax(rgb,axis=0),levels=[0,1,2],colors='y',hatches=['+','////','o'],origin='lower',extent=[low,high]*2)
     boundaries=getBoundaries(np.argmax(rgb,axis=0))
     for bound in boundaries.values():
          ax.RGB.plot([xxL[b] for b in bound],[yyL[b] for b in bound],'k',ls='-',lw=3)
          

     
     for i,(channel,ax_c) in enumerate(zip(rgb,[ax.R,ax.G,ax.B])):
          CS=ax_c.contour(channel, levels=20, colors='dimgrey',linewidths=.5, origin='lower',extent=[low,high]*2)
          #ax_c.clabel(CS, inline=1, fontsize=10)
     #ax.RGB.set_xlabel(r'$\log_{{10}}{\alpha}$',fontsize=20)
     #ax..set_ylabel(r'$\log_{{10}}{\beta}$',fontsize=20)

     #ax.set_xticks([-2,-1,0,1,2])
     #ax.set_yticks([-2,-1,0,1,2])

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
          
          for form in (self.transitions, self.evo_strs, self.evo_samps):
               if (10,0) in form:
                    print("Clearing out 10s")
                    del form[(10,0)]

          
     def __repr__(self):
          return r'Data struct for S*={:.3f},T={:.3g},mu={:.2g},gamma={:.2g}'.format(self.S_star,self.T,self.mu,self.gamma)
     
def loadPickle(S_star,t,mu,gamma):
     with open('Y{}T{}Mu{}F{}.pkl'.format(S_star,*(float(i) for i in (t,mu,gamma))), 'rb') as f:
          return EvoData(S_star,t,mu,gamma,load(f))

def joinPickle(p1,p2):
     for pid in ((2,0),(4,0)):
          p1.evo_strs[pid]=p2.evo_strs[pid]
     return p1
     
def loadNPZ(S_star,t,mu,gamma):
     return np.load('Y{}T{}Mu{}F{}.npz'.format(S_star,*(float(i) for i in (t,mu,gamma))), 'rb')['arr_0']
     
def plotExp(evo_data_struct,add_samps=False):

     #pull simulation parameters from data structure
     evo_data=evo_data_struct.evo_strs
     evo_data_samp=evo_data_struct.evo_samps
     S_star=evo_data_struct.S_star
     mu=evo_data_struct.mu

     #plotting parameters
     phen_cols={(1,0):'dimgrey',(2,0):'firebrick',(4,0):'olivedrab',(4,1):'darkblue',(8,0):'chocolate',(3,0):'goldenrod',(12,0):'k',(16,0):'k',(12,1):'chartreuse'}
     bond_marks={1:'o',2:'s',3:'v',4:'^'}
     bond_ls={1:'-',2:':',3:'-.',4:'--'}
     c_ls={1:'c',2:'m',3:'w.',4:'k'}
     N_markers=6
     MAX_G=250

     #possible ancestors of each pid
     ancestors={(2,0):{1:(1,0)}, (4,0):{2:(1,0)}, (4,1):{4:(2,0),3:(4,0)}, (8,0):{1:(4,0),2:(2,0)}, (12,0): {2:(4,1),4:(8,0)}, (16,0): {2:(4,1),4:(8,0)}}
     
     
     f,axarr=plt.subplots(2,3,sharey=True)

     #put each panel into dictionary accessed by pid
     ax_d={(4,0):axarr[0,0],(4,1):axarr[0,1],(12,0):axarr[1,2],(2,0):axarr[1,0],(8,0):axarr[1,1],(16,0):axarr[0,2]}

 
     null_expectations=RandomWalk(64,None,.5,S_star,True)
     
     for phen,data in evo_data.items():
          #ignore unused pids
          if phen not in ancestors:
               continue

          #set panel border colour
          plt.setp(ax_d[phen].spines.values(), color=phen_cols[phen],lw=(2 if phen[0]<10 else 1))
          
          #plot baseline and random walk expectation
          ax_d[phen].axhline(S_star,c='k',ls='-',lw=0.75)
          max_len=min(MAX_G,max(len(vals) for vals in data.values()))
          ax_d[phen].plot(range(max_len+1),RandomWalk(64,max_len,mu*1./6,S_star,False),ls='--',c='k',zorder=20)

          #iterate over each edge in the assembly graph
          for i,((bond,new),strs) in enumerate(data.items()):
               if new not in ancestors[phen]:
                    continue
               ancestor_p=ancestors[phen][new]

               #offset markers on edges with a top row ancestor
               top_row_ancestor=ancestor_p[0]==4
               strs=strs[:MAX_G]
               mark_tuple=[MAX_G//20*(top_row_ancestor),MAX_G//10]
               if phen==(8,0):
                    mark_tuple[0]+=MAX_G//40*(bond%2)
               mark_tuple=tuple(mark_tuple)

               #plot averaged edge trend given plotting parameters above
               ax_d[phen].plot(range(len(strs)),strs,c=phen_cols[ancestor_p],marker=bond_marks[bond],markevery=mark_tuple,zorder=10,lw=.5,fillstyle=('none' if top_row_ancestor else 'full'),mew=1.5)
               
               #if adding sampled individual runs, plot now
               if add_samps and evo_data_samp is not None:
                   for samp in evo_data_samp[phen][(bond,new)]:
                        samp=samp[:MAX_G]
                        ax_d[phen].plot(range(len(samp)),samp,c=phen_cols[ancestors[phen][new]],alpha=0.15,lw=.5)
               
     #add labels
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

     for (pure,phen,phen_source),phase_p in phase_parameters.items():
          fc='none' if pure else 'aqua'
          ZZ=10
          hatch='' #if pure else '////'
          m_size=500 if phen_source==(12,0) else 120
          if phen==(12,0):              
               ec='w' #if pure else 'k'
               ax_2d.scatter(*np.log10(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=ec,s=m_size,label=phen_source,lw=2,hatch=hatch,zorder=ZZ)
          else:
               col='firebrick' if phen==(16,0) else 'forestgreen'
               pattern='////' if phen==(16,0) else ''
               if fc=='aqua':
                    fc= 'orchid' if phen==(16,0) else fc
               ax_1d.scatter(np.log10(phase_p),FourOne(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=col,s=m_size*2,label=phen_source,zorder=ZZ,hatch=pattern)
     ax_2d.legend()
     ax_1d.legend()
     plt.show(block=False)

def plotPathways(evo_data,norm_style=''):
     evo_trans=evo_data.transitions
     failed=evo_data.jumps
     
     if (10,0) in failed:
          for phen_from,count in failed[(10,0)].items():
               if (12,0) in failed:
                    if phen_from in failed[(12,0)]:
                         failed[(12,0)][phen_from]+=count
                    else:
                         failed[(12,0)][phen_from]=count
               else:
                    failed[(12,0)]={phen_from:count}
                    
                    
     
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
               except KeyError:
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
