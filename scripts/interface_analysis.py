import subprocess
import os
#subprocess.run('cd ..',shell=True)

#os.chdir('scripts')
from interface_methods import *
import numpy as np


from pickle import load,dump
from collections import defaultdict,Counter
from itertools import product
import math

import warnings

#GLOBAL PIDS
null_pid,init_pid=np.array([0,0],dtype=np.uint8),np.array([1,0],dtype=np.uint8)

          
def collateNPZs(S_star,t,mu,gamma,runs):
     full_data=[]
     for r in runs:
          full_data.append(np.load(open('Mu{}Y{}T{}F{}O{}.npz'.format(mu,S_star,t,gamma,r), 'rb'))['arr_0'])
     return np.asarray(full_data)

def collateAnalysis(S_star,t,mu,gamma,runs):
     N_samps=10
     full_data=[]
     for r in runs:
          try:
               full_data.append(load(open('Mu{}Y{}T{}F{}O{}.pkl'.format(mu,S_star,t,gamma,r), 'rb')))
          except:
               print('missing pickle for run ',r)
     N_runs=0
     full_transition=defaultdict(lambda: defaultdict(int))
     failed_jumps=defaultdict(lambda: defaultdict(int))
     raw_evolutions=defaultdict(lambda: defaultdict(list))
     average_evolutions=defaultdict(dict)
     sample_evolutions=defaultdict(dict)
     
     for str_evo,phen_tran,fails in filter(None,full_data):
          N_runs+=1
          for (phen_in,phen_out),count in phen_tran.items():
               full_transition[phen_in][phen_out]+=count
          for (phen_in,phen_out),count in fails.items():
               failed_jumps[phen_in][phen_out]+=count
          for phen,evo_list in str_evo.items():
               for (gen,evos) in evo_list:
                    for (bond_key,new_bond,_,_),strs in evos.items():
                         raw_evolutions[phen][(bond_key,new_bond)].append(strs)
          
     for phen in raw_evolutions.keys():
          samps=None
          for bond_key,evos in raw_evolutions[phen].items():
               str_arr=convertRaggedArray(evos)
               if samps is None or max(samps)>=str_arr.shape[0]:
                    samps=np.random.choice(str_arr.shape[0],min(str_arr.shape[0],N_samps),replace=False)
               average_evolutions[phen][bond_key]=np.nanmean(str_arr,axis=0)
               sample_evolutions[phen][bond_key]=str_arr[samps]
     return (N_runs,)+tuple(convertDoubleNestedDict(tr) for tr in (full_transition,failed_jumps)) + tuple(dict(res) for res in (average_evolutions,sample_evolutions))
     
def convertRaggedArray(list_of_lists):
     long_length=sorted([len(lis) for lis in list_of_lists],reverse=True)[0]
     rect_arr=np.empty((len(list_of_lists),long_length))
     for i,strs in enumerate(list_of_lists):
          rect_arr[i]=strs[:long_length]+[np.nan]*(long_length-len(strs[:long_length]))
     return rect_arr

def analysePhylogenetics(run,full_pIDs=False):
     s,p,st=loadAllFiles(run,0)
     ret_val=KAG(p,s,st)
     if not ret_val:
          return None
     transitions=ret_val[1]
     failed_transitions=ret_val[2]
     bond_data=treeBondStrengths(ret_val[0],st)
     #print(bond_data.keys())
     if full_pIDs:
          phen_table=loadPhenotypeTable(run)
          transitions={(phen_table[k[0]],phen_table[k[1]]):cnt for k,cnt in transitions.items()}
          failed_transitions={(phen_table[k[0]],phen_table[k[1]]):cnt for k,cnt in failed_transitions.items()}
          bond_data={phen_table[k]:v for k,v in bond_data.items()}
     return (bond_data,transitions,failed_transitions)

class Tree(object):
     __slots__ = ('pID','bonds','new_bond','gen','seq')
     def __init__(self,pid=None,bonds=None,new_bond=None,gen=None,seq=None):
          self.pID=pid
          self.bonds=bonds
          self.new_bond=new_bond
          self.gen=gen
          self.seq=seq
     def __repr__(self):
          return '{},{}'.format(self.pID,self.gen)
          
def KAG(phenotypes_in,selections,interactions):
     phenotypes=phenotypes_in.copy()
     max_gen,pop_size=selections.shape
     
     forest,temp_forest=[],[]
     transitions=defaultdict(int)
     
     def __growDescendentTree(tree,max_depth=float('inf')):
          gen_val=tree.gen
          descendents=tree.seq[0]
          
          def __valid_descendent(gen_idx,kid):
               return (np.array_equal(phenotypes_in[gen_idx+1,kid],tree.pID) and interactions[gen_idx+1,kid].bonds==tree.bonds)
          
          while gen_val<(max_gen-1):
               new_descendents=[child for descendent in descendents for child in np.where(selections[gen_val]==descendent)[0] if __valid_descendent(gen_val,child)]
                    
               if math.isinf(max_depth):
                    phenotypes[gen_val,descendents]=null_pid
               elif (gen_val-tree.gen)>=max_depth:
                    return True
               
               if not new_descendents:
                    break
               
               descendents=new_descendents
               tree.seq.append(descendents)
               gen_val+=1
          else:
               if math.isinf(max_depth):
                    phenotypes[gen_val,descendents]=null_pid
               elif (gen_val-tree.gen)>=max_depth:
                    return True
                              
     def __addBranch():        
          if g_idx:
               if len(bond_ref)<len(interactions[g_idx-1,p_idx].bonds):
                    return False;
               new_bond=list(set(bond_ref)-set(interactions[g_idx-1,p_idx].bonds))
               par_ref=phenotypes_in[g_idx-1,p_idx]

          else:
               new_bond=bond_ref
               par_ref=init_pid
               
          transitions[tuple(tuple(_) for _ in (pid_ref,par_ref))]+=1
          new_bond=[new_bond[0]]
          if len(new_bond)!=1:
               return False
          else:
               temp_forest.append((True,Tree(pid_ref.copy(),bond_ref,new_bond[0],g_idx,[[c_idx]])))
               return True
               
     for C_INDEX in range(pop_size):
          if np.array_equal(phenotypes[max_gen-1,C_INDEX],null_pid):
               continue
          
          c_idx=C_INDEX
          g_idx=max_gen-1
          p_idx=selections[g_idx-1,c_idx]
          pid_ref=phenotypes[g_idx,c_idx]
          bond_ref=interactions[g_idx,c_idx].bonds
          
          while g_idx>0:
               if np.array_equal(phenotypes[g_idx-1,p_idx],null_pid):
                    if np.array_equal(phenotypes_in[g_idx-1,p_idx],pid_ref):
                         temp_forest.append((False,Tree(pid_ref,bond_ref,(-1,-1),g_idx,[[c_idx]])))
                    else:
                         if not __addBranch():
                              return None
                         break
               
               elif not np.array_equal(phenotypes[g_idx-1,p_idx],pid_ref):
                    if not __addBranch():                         
                         return None
                    bond_ref=interactions[g_idx-1,p_idx].bonds
                    pid_ref=phenotypes[g_idx-1,p_idx]
                    
               elif interactions[g_idx-1,p_idx].bonds !=bond_ref:
                    temp_forest.append((False,Tree(pid_ref,bond_ref,(-1,-1),g_idx,[[c_idx]])))
                    bond_ref=interactions[g_idx-1,p_idx].bonds
          
               g_idx-=1
               c_idx=p_idx
               p_idx=selections[g_idx-1,p_idx]
          else:
               if not np.array_equal(pid_ref,init_pid) and not __addBranch():
                    return None
          
          while temp_forest:
               (alive,tree)=temp_forest.pop()
               __growDescendentTree(tree)
               if alive:
                    forest.append(tree)
                    
     evo_stage=np.vectorize(lambda x: len(x.bonds))(interactions)
     
     POPULATION_FIXATION=pop_size//20
     SURVIVAL_DEPTH=5
     failed_jumps=defaultdict(int)
     for g_idx,c_idx in product(range(max_gen-2,-1,-1),range(pop_size)):
          pid_c=phenotypes[g_idx,c_idx]
          if np.array_equal(pid_c,null_pid):
               continue
          
          pid_d=phenotypes_in[g_idx-1,selections[g_idx-1,c_idx]] if g_idx>0 else init_pid
          if not np.array_equal(pid_c,pid_d):
               if np.count_nonzero(evo_stage[g_idx]>=len(interactions[g_idx,c_idx].bonds))>POPULATION_FIXATION:
                    continue
               
               if not __growDescendentTree(Tree(pid_c,interactions[g_idx,c_idx].bonds,(-1,-1),g_idx,[[c_idx]]),SURVIVAL_DEPTH):
                    failed_jumps[tuple(tuple(_) for _ in (pid_c,pid_d))]+=1
     
     return (forest,dict(transitions),dict(failed_jumps))
     
def treeBondStrengths(KAG,interactions):
     bond_data=defaultdict(list)
     for tree in KAG: 
          bond_maps=defaultdict(list)
          max_pop=0
          for generation,populations in enumerate(tree.seq,tree.gen):
               if len(populations)<(max_pop//10) and max_pop>(interactions.shape[1]//10):
                    #print(len(populations),max_pop)
                    break
               max_pop=max(max_pop,len(populations))
               inner_bond_maps=defaultdict(list)
               for species in populations:
                    all_bonds=interactions[generation,species].bonds
                    new_bond_type=getBondType(tree.new_bond,all_bonds)
                    for bond,strength in interactions[generation,species]:
                         inner_bond_maps[(getBondType(bond,all_bonds),new_bond_type)+bond].append(strength)

               for k,v in inner_bond_maps.items():
                    bond_maps[k].append(np.mean(v))
          bond_data[tuple(tree.pID)].append((tree.gen,dict(bond_maps)))
     return dict(bond_data)     



def convertDoubleNestedDict(dict_in):
     return {k:dict(v) for k,v in dict_in.items()}

def getBondType(bond,bonds):
     if checkBranchingPoint(bond,bonds):
          if bond[0]//4==bond[1]//4:
               return 4 #same tile BP
          else:
               return 3 #external BP
     else:
          if bond[0]//4==bond[1]//4:
               return 2 #internal loop
          else:
               return 1 #external SIF

def checkBranchingPoint(bond,bonds):
     test_bonds=list(sum((b for b in bonds if b!= bond), ()))
     return any(x in test_bonds for x in bond)

   
def analyseHomogeneousPopulation(run,temperature):
     selections,phenotypes,st=loadAllFiles(run)
     
     max_gen,pop_size=selections.shape

     param_trajectory=[]
     for generation in range(max_gen):
          params=defaultdict(list)
          for species in range(pop_size):
               if np.array_equal(phenotypes[generation,species],null_pid):
                    continue
               if len(st[generation,species].bonds)!=3:
                    continue
               try:
                    bonds={getBondType(bond,st[generation,species].bonds):strength for bond,strength in st[generation,species]}
                    params['a'].append((bonds[4]/bonds[2])**temperature)
                    params['b'].append((bonds[2]/bonds[3])**temperature)
               except KeyError:
                    ##incomplete bonds, ignore and move on
                    continue

          param_trajectory.append([np.mean(params['a']),np.mean(params['b'])])
     return np.asarray(param_trajectory)


def runEvolutionSequence():

     default_parameters={'file_path' : '../bin/', 'N' : 2, 'P' : 100, 'K' : 1000, 'B' : 20, 'X': .75, 'F': 5, 'A' : 2, 'D' : 4, 'J': 5, 'M': 1, 'Y' : .6875, 'T': 10}
     
     print('Running evolution sequence')

     def generateParameterString():
          prm_str=''
          for param,value in default_parameters.items():
               if param == 'file_path':
                    continue
               prm_str+='-{} {} '.format(param,value)
          return prm_str

     ##run evolution simulation given parameters 
     subprocess.run(default_parameters['file_path']+'ProteinEvolution -E ' + generateParameterString()[:-1],shell=True)

     fname_params=tuple(float(default_parameters[k]) for k in ('Y','T','M','F'))

     for run in range(default_parameters['D']):
          analysisByMode(default_parameters['A'],run, fname_params)

     collateByMode(default_parameters['A'],range(default_parameters['D']),fname_params)
                             


def analysisByMode(mode,run, params):
     file_base='Mu{2}Y{0}T{1}F{3}O{4}'.format(*(params+(run,)))
     if mode < 2:
          analysis=analysePhylogenetics(run, mode==0)
          with open(file_base+'.pkl','wb') as f:
               dump(analysis,f)
     elif mode == 2:
          analysis=analyseHomogeneousPopulation(run,params[1])          
          np.savez_compressed(file_base,analysis)
     else:
          print('unknown mode request, set parameter \'A\'')

def collateByMode(mode,run_range, params):
     file_base='Y{}T{}Mu{}F{}'.format(*params)
     if mode < 2:
          with open(file_base+'.pkl','wb') as f:
               dump(collateAnalysis(*params,runs=run_range),f)

     elif mode == 2:
          np.savez_compressed(file_base,collateNPZs(*params,runs=run_range))
          
     else:
          print('unknown mode request, set parameter \'A\'')

if __name__ == '__main__':
    runEvolutionSequence()
