import sys
import numpy as np

#add local paths to load custom methods
if not any(('scripts' in pth for pth in sys.path)):
     sys.path.append('scripts/')
     sys.path.append('../polyomino_core/scripts')

from interface_analysis import *
from interface_methods import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

from matplotlib.colors import LinearSegmentedColormap

from itertools import product
from collections import defaultdict


     
##example sequence of loading data and plotting
def examplePlot():
    ##if data not generated and analysed, can uncomment and run below line
    #runEvolutionSequence()

    ##load the data from the pickle file (should be placed in this directory_
    data=loadPickle(.6875,25,1,5)

    ##plot the data as in Fig 4, and then print pathway success as in Fig 6
    calculatePathwaySucess(data)
    plotEvolution(data,True,False)
    



##plot evolution of phenotypes over time, used in Fig 4
##parameters
##evo_data (custom structure from loadPickle method): structure holding simulation results
##add_samps (bool): if false plots only bulk trends, if true plots a sample of individual simulations
def plotEvolution(evo_data_struct,add_samps=False,interactive_mode=True):

     ##pull simulation parameters from data structure
     evo_data=evo_data_struct.evo_strs
     evo_data_samp=evo_data_struct.evo_samps
     S_star=evo_data_struct.S_star
     mu=evo_data_struct.mu

     ##plotting parameters
     phen_cols={(1,0):'dimgrey',(2,0):'firebrick',(4,0):'olivedrab',(4,1):'darkblue',(8,0):'chocolate',(3,0):'goldenrod',(12,0):'k',(16,0):'k',(12,1):'chartreuse'}
     bond_marks={1:'o',2:'s',3:'v',4:'^'}

     MAX_G=250

     ##possible ancestors of each pid
     ancestors={(2,0):{1:(1,0)}, (4,0):{2:(1,0)}, (4,1):{4:(2,0),3:(4,0)}, (8,0):{1:(4,0),2:(2,0)}, (12,0): {2:(4,1),4:(8,0)}, (16,0): {2:(4,1),4:(8,0)}}

     f,axarr=plt.subplots(2,3,sharey=True)

     ##put each panel into dictionary accessed by pid
     ax_d={(4,0):axarr[0,0],(4,1):axarr[0,1],(12,0):axarr[1,2],(2,0):axarr[1,0],(8,0):axarr[1,1],(16,0):axarr[0,2]}

               
     
     for phen,data in evo_data.items():
          ##ignore unused pids
          if phen not in ancestors:
               continue

          ##set panel border colour
          plt.setp(ax_d[phen].spines.values(), color=phen_cols[phen],lw=(2 if phen[0]<10 else 1))

          ##plot baseline and random walk expectation
          ax_d[phen].axhline(S_star,c='k',ls='-',lw=0.75)
          max_len=min(MAX_G,max(len(vals) for vals in data.values()))
          ax_d[phen].plot(range(max_len+1),RandomWalk(64,max_len,mu*1./6,S_star,False),ls='--',c='k',zorder=20)


          ##iterate over each edge in the assembly graph
          for (bond,new),strs in data.items():
               if new not in ancestors[phen]:
                    continue
               ancestor_p=ancestors[phen][new]

               ##offset markers on edges with a top row ancestor
               top_row_ancestor=ancestor_p[0]==4
               strs=strs[:MAX_G]
               mark_tuple=[MAX_G//20*(top_row_ancestor),MAX_G//10]
               if phen==(8,0):
                    mark_tuple[0]+=MAX_G//40*(bond%2)
               mark_tuple=tuple(mark_tuple)

               def plotValues(values,alph):
                    values=values[:MAX_G]
                    ax_d[phen].plot(range(len(values)),values,c=phen_cols[ancestor_p],marker=bond_marks[bond],markevery=mark_tuple,zorder=10,lw=.5,fillstyle=('none' if top_row_ancestor else 'full'),mew=1.5,alpha=alph)


               ##plot averaged edge trend given plotting parameters above
               plotValues(strs,1)

               ##if adding sampled individual runs, plot now
               if add_samps and evo_data_samp is not None:
                   for samp in evo_data_samp[phen][(bond,new)]:
                        plotValues(samp,.2)

     ##add labels
     f.tight_layout()
     axarr[1,1].set_xlabel('generations')
     axarr[1,0].set_ylabel(r'$\langle \hat{S} \rangle$')
     plt.show(block= not interactive_mode)


##print the transition success rates used in Fig 6
##parameters
##evo_data (custom structure from loadPickle method): structure holding simulation results
def calculatePathwaySucess(evo_data):
     evo_trans=evo_data.transitions
     failed=evo_data.jumps

     ##count misassembled 10-mer as a failure to transition to 12-mer
     if (10,0) in failed:
          for phen_from,count in failed[(10,0)].items():
               if (12,0) in failed:
                    if phen_from in failed[(12,0)]:
                         failed[(12,0)][phen_from]+=count
                    else:
                         failed[(12,0)][phen_from]=count
               else:
                    failed[(12,0)]={phen_from:count}

     for child, ancestor_details in sorted(evo_trans.items()):
          for parent, count in ancestor_details.items():
               
               ##get total number of failed transitions
               fail_rate=0
               if child in failed:
                    if parent in failed[child]:
                         fail_rate=failed[child][parent]

               ##print formatted success/fail rates
               print('{}→{} with {} successes and {} failures ({:.2f}%)'.format(parent,child,count,fail_rate,100*count/(count+fail_rate)))

##plot grid of parameters for the 16-mer phenotype, used in S1 Fig
##parameters are
##S_hat (float): the critical interaction strength
##jump (float): the fitness jump value F
##Ts/Ys (list): lists of floats with all values of temperature and nondeterminism to plot
def plotParamGrid(S_hat,jump,Ts,Ys):
     _,axarr=plt.subplots(len(Ys),len(Ts),sharex=True,sharey=True)

     ##plotting parameters
     bc={2:'chocolate',3:'royalblue',4:'forestgreen'}
     mc={2:'o',3:'v',4:'^'}
     mco={2:0,3:16,4:32}

     ##truncate generations
     MAX_G=250

     ##iterate over all temperature, nondeterminism combinations
     for (gamma,T),ax in zip(product(Ys,Ts),axarr.flatten()):

          ##load the relevant data, skip if missing phenotype
          d=loadPickle(S_hat,T,jump,gamma)
          if (16,0) not in d.evo_strs:
               continue

          ##plot the baseline, random expectation
          ax.axhline(S_hat,c='k',ls='-',lw=0.75)
          ax.plot(range(MAX_G),RandomWalk(64,MAX_G-1,1./6,.671875,False),ls='--',c='k',zorder=20)

          ##plot the evolved values
          for (bond,_),strs in d.evo_strs[(16,0)].items():
               ax.plot(range(len(strs[:MAX_G])),strs[:MAX_G],c=bc[bond],marker=mc[bond],markevery=(mco[bond],MAX_G//5))
     
     plt.show(block=False)

##helper function to plot the phase space for the heterotetramer
##parameters
##low (float): lower log scale value to sample
##high (float): upper log scale value to sample
##res (int): number of data points to sample
##called (bool): default option to only display figure interactively
def plot1D(low=-1,high=1,res=500,called=False):
     _,ax=plt.subplots()

     ##use logspace based on parameters, and plot
     xs=np.logspace(low,high,res)
     plt.plot(np.log10(xs),FourOne(xs),'k')

     ##add labels
     ax.set_yticks([.5,.75,1])
     ax.set_yticklabels([r'$50\%$',r'$75\%$',r'$100\%$'])
     ax.set_xlabel(r'$\gamma$',fontsize=20)
     ax.set_ylabel('target phen fraction')

     ##return axis if used as helper, otherwise show plot
     if called:
          return ax
     plt.show(block=False)

#helper method to plot 2D phase space for the 12-mer
##parameters
##low (float): lower log scale value to sample
##high (float): upper log scale value to sample
##res (int): number of data points to sample
##called (bool): default option to only display figure interactively
def plot2D(low=-1,high=1,res=250,called=False):

     ##set up both log and linear scale grid of values
     xx,yy=np.meshgrid(np.logspace(low,high,res),np.logspace(low,high,res))
     xxL,yyL=np.meshgrid(np.linspace(low,high,res),np.linspace(low,high,res))

     ##calculate all 3 polyomino abundances for the grid
     rgb=np.array(Twelve(xx,yy))

     ##set up special RGB axis
     fig = plt.figure()
     ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

     ##plot the RGB values based on the polyomino abundance
     ax.imshow_rgb(*rgb[:,:],interpolation='none',origin='lower',extent=[low,high]*2)

     ##helper method to locate boundary between phenotypes in phase space
     def getBoundaries(maxes):
          change_points=defaultdict(list)
          for r_idx,row in enumerate(maxes):
               for change in [i for i in range(1,len(row)) if row[i]!=row[i-1]]:
                    change_points[(min(row[change-1],row[change]),max(row[change-1],row[change]))].append((r_idx,change))
          return change_points

     ##get boundaries and plot
     boundaries=getBoundaries(np.argmax(rgb,axis=0))
     for bound in boundaries.values():
          ax.RGB.plot([xxL[b] for b in bound],[yyL[b] for b in bound],'k',ls='-',lw=3)

     ##add contour lines for polyomino abundance on each colour channel
     for i,(channel,ax_c) in enumerate(zip(rgb,[ax.R,ax.G,ax.B])):
          ax_c.contour(channel, levels=20, colors='dimgrey',linewidths=.5, origin='lower',extent=[low,high]*2)

     if called:
          return ax
     plt.show(block=False)
     
##plot phase space and transition locations in Fig 6
##parameters
##evo_data (custom structure from loadPickle method): structure holding simulation results
##low (float): lower log scale value to sample
##high (float): upper log scale value to sample
##res (int): number of data points to sample
def plotPhaseSpace(evo_data,low=-2,high=2,res=250):

     ##extract simulation data from structure
     evo_strs=evo_data.evo_strs
     S_star=evo_data.S_star
     T=evo_data.T

     ##plot 1D and 2D phase spaces
     ax_1d=plot1D(low,high,res*4,True)
     ax_2d=plot2D(low,high,res,True)

     ##calculate transition locations in phase space
     phase_parameters=calcTransitionParams(evo_strs,evo_data.transitions,T,S_star)

     ##plotting parameter
     phen_mark={(2,0):'P',(4,0):'D',(4,1):'s',(8,0):'o',(16,0):'X',(12,0):'*'}
     fc='none'
     m_size=250
     
     for (pure,phen,phen_source),phase_p in phase_parameters.items():
          ##prefer to take prediction from initial conditions
          if pure:
               continue

          ##if 12-mer, plot on its 2D phase space
          if phen==(12,0):
               ec='w'
               ax_2d.RGB.scatter(*np.log10(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=ec,s=m_size,label=phen_source,lw=2,zorder=10)
          ##otherwise plot on 1D phase space (for heterotetramer and 16-mer)
          else:
               col='firebrick' if phen==(16,0) else 'forestgreen'
               ax_1d.scatter(np.log10(phase_p),FourOne(phase_p),marker=phen_mark[phen_source],c=fc,edgecolors=col,s=m_size*2,label=phen_source,zorder=10)

     ##add legends and show
     ax_2d.RGB.legend()
     ax_1d.legend()
     plt.show(block=False)

     

     
##plotting method for the dynamic landscape, used in part for Fig 8 and S2 Fig
##parameters
##data (numpy ndarray from loadNPZ method): bond strengths from simulations
##period (int): length of time between fitness switches
def plotDynamicLandscape(data,period=100):

     ##plot base phase space
     ax=plot2D(-2.5,2.5,250,True)

     ##great 2 colour gradients for the fitness landscape changing
     c_grad_1 = LinearSegmentedColormap.from_list('grad_1', ['aqua','darkblue'], N=period)
     c_grad_2 = LinearSegmentedColormap.from_list('grad_2', ['mistyrose','firebrick'], N=period)

     ##create colour list with changing landscape
     phase_color=[]
     for _ in range(data.shape[1]//period//2):
          phase_color.extend(list(c_grad_1(np.linspace(0,1,period)))+list(c_grad_2(np.linspace(0,1,period))))

     ##can the data burn in a few hundred generations
     burn_in=0
     (x1,y1)=np.log10(np.mean(data[:,burn_in:,:],axis=0)).T

     ##plot arrows in direction of population flow
     ax.RGB.quiver(x1[:-1], y1[:-1], x1[1:]-x1[:-1], y1[1:]-y1[:-1], scale_units='width',scale=5, angles='xy',zorder=10,color=phase_color)

     ##plot individual panels for each period
     for i in range(0,data.shape[1],period):
          if i%(2*period)==0:
               ax.G.plot(x1[i:i+period],y1[i:i+period],alpha=0.5,c='aqua')
          else:
               ax.B.plot(x1[i:i+period],y1[i:i+period],alpha=0.5,c='mistyrose')

     plt.show(block=False)
     return

if __name__ == '__main__':
     try:
          examplePlot()
     except Exception as e:
          print(e)
     else:
          print('Plotting sequence successful')
          sys.exit(0)
     print('Something went wrong, data loading or plotting potentially incomplete')
     sys.exit(1)
