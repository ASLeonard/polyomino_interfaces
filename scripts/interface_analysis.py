import subprocess
import sys
#add local paths to load custom methods
if not any(('scripts' in pth for pth in sys.path)):
     sys.path.append('scripts/')
     
from interface_methods import *
import numpy as np

from pickle import load,dump
from collections import defaultdict
from itertools import product
import math


##GLOBAL PIDS
null_pid,init_pid=np.array([0,0],dtype=np.uint8),np.array([1,0],dtype=np.uint8)

##group together many dynamic simulations into one structure
def collateNPZs(S_star,t,mu,gamma,runs):
     full_data=[]
     for r in runs:
          try:
               full_data.append(np.load(open('Mu{}Y{}T{}F{}O{}.npz'.format(mu,S_star,t,gamma,r), 'rb'))['arr_0'])
          except FileNotFoundError:
               print('missing file for run ',r)
     return np.asarray(full_data)

##helper method to create numpy array with nans if ragged
def convertRaggedArray(list_of_lists):
     long_length=sorted([len(lis) for lis in list_of_lists],reverse=True)[0]
     rect_arr=np.empty((len(list_of_lists),long_length))
     for i,strs in enumerate(list_of_lists):
          rect_arr[i]=strs[:long_length]+[np.nan]*(long_length-len(strs[:long_length]))
     return rect_arr

##helper method to turn nested defaultdict to dict
def convertDoubleNestedDict(dict_in):
     return {k:dict(v) for k,v in dict_in.items()}

##group together many evolution simulations into one structure
def collateAnalysis(S_star,t,mu,gamma,runs):
     N_samps=10
     full_data=[]

     ##load the raw data
     for r in runs:
          try:
               full_data.append(load(open('Mu{}Y{}T{}F{}O{}.pkl'.format(mu,S_star,t,gamma,r), 'rb')))
          except FileNotFoundError:
               print('missing pickle for run ',r)

     ##declare objects
     N_runs=0
     full_transition=defaultdict(lambda: defaultdict(int))
     failed_jumps=defaultdict(lambda: defaultdict(int))
     raw_evolutions=defaultdict(lambda: defaultdict(list))
     average_evolutions=defaultdict(dict)
     sample_evolutions=defaultdict(dict)

     ##read raw data into objects
     for str_evo,phen_tran,fails in filter(None,full_data):
          N_runs+=1
          for data_strc, data_in in ((full_transitions,phen_tran),(failed_jumps,fails)):
               for (phen_in,phen_out),count in data_in.items():
                    data_strc[phen_in][phen_out]+=count
               
          for phen,evo_list in str_evo.items():
               for (_,evos) in evo_list:
                    for (bond_key,new_bond,_,_),strs in evos.items():
                         raw_evolutions[phen][(bond_key,new_bond)].append(strs)

     ##find average values and take samples
     for phen in raw_evolutions.keys():
          samps=None
          for bond_key,evos in raw_evolutions[phen].items():
               str_arr=convertRaggedArray(evos)
               if samps is None or max(samps)>=str_arr.shape[0]:
                    samps=np.random.choice(str_arr.shape[0],min(str_arr.shape[0],N_samps),replace=False)
                    
               average_evolutions[phen][bond_key]=np.nanmean(str_arr,axis=0)
               sample_evolutions[phen][bond_key]=str_arr[samps]

     ##return data in tuple converted back into plain dictionaries
     return (N_runs,)+tuple(convertDoubleNestedDict(tr) for tr in (full_transition,failed_jumps)) + tuple(dict(res) for res in (average_evolutions,sample_evolutions))

##analysis of evolution simulations
def analysePhylogenetics(run,full_pIDs=False):
     s,p,st=loadAllFiles(run,0)

     ##find ancestral groupings and transition success/fail
     ret_val=KAG(p,s,st)
     if not ret_val:
          return None
     transitions=ret_val[1]
     failed_transitions=ret_val[2]

     ##calculate strength evolution given ancestral trees
     bond_data=treeBondStrengths(ret_val[0],st)

     ##if full_pIDs, express in terms of phenotype not pid
     if full_pIDs:
          phen_table=loadPhenotypeTable(run)
          transitions={(phen_table[k[0]],phen_table[k[1]]):cnt for k,cnt in transitions.items()}
          failed_transitions={(phen_table[k[0]],phen_table[k[1]]):cnt for k,cnt in failed_transitions.items()}
          bond_data={phen_table[k]:v for k,v in bond_data.items()}
          
     return (bond_data,transitions,failed_transitions)

##custom tree object to store information on ancestor and descendents
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

##find all ancestral trees within an evolution simulation and group
def KAG(phenotypes_in,selections,interactions):
     phenotypes=phenotypes_in.copy()
     max_gen,pop_size=selections.shape
     
     forest,temp_forest=[],[]
     transitions=defaultdict(int)

     ##helper method to find all descendents from the root of a tree
     def __growDescendentTree(tree,max_depth=float('inf')):
          gen_val=tree.gen
          descendents=tree.seq[0]

          ##needs pid and edges to match to be a true descendent
          def __valid_descendent(gen_idx,kid):
               return (np.array_equal(phenotypes_in[gen_idx+1,kid],tree.pID) and interactions[gen_idx+1,kid].bonds==tree.bonds)

          ##iterate down tree while possible
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

     ##helper method to identify where a tree branches into a new ancestry
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

          ##if too rapid a change or loss, branch is not well formed and rejected
          if len(new_bond)!=1:
               return False
          else:
               temp_forest.append((True,Tree(pid_ref.copy(),bond_ref,new_bond[0],g_idx,[[c_idx]])))
               return True

     ##main loop over the entire generation/population
     for C_INDEX in range(pop_size):
          if np.array_equal(phenotypes[max_gen-1,C_INDEX],null_pid):
               continue
          
          c_idx=C_INDEX
          g_idx=max_gen-1
          p_idx=selections[g_idx-1,c_idx]
          pid_ref=phenotypes[g_idx,c_idx]
          bond_ref=interactions[g_idx,c_idx].bonds

          ##iterate backwards through tree finding new branches
          while g_idx>0:
               ##if parent is null, has been found already
               if np.array_equal(phenotypes[g_idx-1,p_idx],null_pid):
                    if np.array_equal(phenotypes_in[g_idx-1,p_idx],pid_ref):
                         temp_forest.append((False,Tree(pid_ref,bond_ref,(-1,-1),g_idx,[[c_idx]])))
                    else:
                         if not __addBranch():
                              return None
                         break
               ##if not equal, found a transition, add branch at this point
               elif not np.array_equal(phenotypes[g_idx-1,p_idx],pid_ref):
                    if not __addBranch():                         
                         return None
                    bond_ref=interactions[g_idx-1,p_idx].bonds
                    pid_ref=phenotypes[g_idx-1,p_idx]
                    
               elif interactions[g_idx-1,p_idx].bonds !=bond_ref:
                    temp_forest.append((False,Tree(pid_ref,bond_ref,(-1,-1),g_idx,[[c_idx]])))
                    bond_ref=interactions[g_idx-1,p_idx].bonds

               ##step back a generation and find parent
               g_idx-=1
               c_idx=p_idx
               p_idx=selections[g_idx-1,p_idx]
          else:
               if not np.array_equal(pid_ref,init_pid) and not __addBranch():
                    return None
          ##look back at roots of new branches and extend them if valid
          while temp_forest:
               (alive,tree)=temp_forest.pop()
               __growDescendentTree(tree)
               if alive:
                    forest.append(tree)

     ##find "complexity" of each phenotype
     evo_stage=np.vectorize(lambda x: len(x.bonds))(interactions)
     POPULATION_FIXATION=pop_size//20
     SURVIVAL_DEPTH=5
     failed_jumps=defaultdict(int)

     ##iterate over roots of branches
     for g_idx,c_idx in product(range(max_gen-2,-1,-1),range(pop_size)):
          pid_c=phenotypes[g_idx,c_idx]
          if np.array_equal(pid_c,null_pid):
               continue
          
          pid_d=phenotypes_in[g_idx-1,selections[g_idx-1,c_idx]] if g_idx>0 else init_pid
          ##if new phenotype wasn't actually more fit than neighbours, ignore
          if not np.array_equal(pid_c,pid_d):
               if np.count_nonzero(evo_stage[g_idx]>=len(interactions[g_idx,c_idx].bonds))>POPULATION_FIXATION:
                    continue
               ##only consider it failed if it "should" have survived based on fitness advantage 
               if not __growDescendentTree(Tree(pid_c,interactions[g_idx,c_idx].bonds,(-1,-1),g_idx,[[c_idx]]),SURVIVAL_DEPTH):
                    failed_jumps[tuple(tuple(_) for _ in (pid_c,pid_d))]+=1
     
     return (forest,dict(transitions),dict(failed_jumps))

##calculate strengths of each part of the tree
def treeBondStrengths(KAG,interactions):
     bond_data=defaultdict(list)
     for tree in KAG:
          bond_maps=defaultdict(list)
          max_pop=0
          for generation,populations in enumerate(tree.seq,tree.gen):
               ##if branch is dying off, cut off when statistics too low
               if len(populations)<(max_pop//10) and max_pop>(interactions.shape[1]//10):
                    break
               max_pop=max(max_pop,len(populations))
               inner_bond_maps=defaultdict(list)
               ##find strengths for all bonds, and get bond topology
               for species in populations:
                    all_bonds=interactions[generation,species].bonds
                    new_bond_type=getBondType(tree.new_bond,all_bonds)
                    for bond,strength in interactions[generation,species]:
                         inner_bond_maps[(getBondType(bond,all_bonds),new_bond_type)+bond].append(strength)

               for k,v in inner_bond_maps.items():
                    bond_maps[k].append(np.mean(v))
          bond_data[tuple(tree.pID)].append((tree.gen,dict(bond_maps)))
     return dict(bond_data)

##get bond topology based on edge properties
def getBondType(bond,bonds):

     ##check if multiple edges from this interface
     def checkBranchingPoint(bond,bonds):
          test_bonds=list(sum((b for b in bonds if b!= bond), ()))
          return any(x in test_bonds for x in bond)
     
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

##easier analysis for a fixed population, just tracking evolution of known edge strengths
def analyseHomogeneousPopulation(run,temperature):
     selections,phenotypes,st=loadAllFiles(run)
     
     max_gen,pop_size=selections.shape

     param_trajectory=[]
     for generation in range(max_gen):
          params=defaultdict(list)
          for species in range(pop_size):

               ##if the individual is wrong pid or lost a bond, move on
               if np.array_equal(phenotypes[generation,species],null_pid):
                    continue
               if len(st[generation,species].bonds)!=3:
                    continue

               ##try just for safety
               try:
                    ##get all the bonds, and they are of known hardcoded topology
                    bonds={getBondType(bond,st[generation,species].bonds):strength for bond,strength in st[generation,species]}
                    params['a'].append((bonds[4]/bonds[2])**temperature)
                    params['b'].append((bonds[2]/bonds[3])**temperature)
               except KeyError:
                    ##incomplete bonds, ignore and move on
                    continue
               
          ##append the per-generation means
          param_trajectory.append([np.mean(params['a']),np.mean(params['b'])])
     return np.asarray(param_trajectory)


def runEvolutionSequence():

     ##change parameters here for data generation
     
     ##defaults well-suited for generating evolution of fixed system
     default_parameters={'file_path' : '../bin/', 'N' : 2, 'P' : 100, 'K' : 400, 'B' : 100, 'X': .5, 'F': 5, 'A' : 1, 'D' : 1, 'J': 5, 'M': 1, 'Y' : .6875, 'T': 10, 'O' : 100, 'G' : 10} 

     ##defaults for dynamic fitness landscape 
     #default_parameters={'file_path' : '../bin/', 'N' : 2, 'P' : 100, 'K' : 600, 'B' : 150, 'X': 0, 'F': 1, 'A' : 2, 'D' : 1, 'J': 1, 'M': 1, 'Y' : .6875, 'T': 25, 'O' : 200, 'G' : 10} 


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

     ##run analysis
     for run in range(default_parameters['D']):
          analysisByMode(default_parameters['A'],run, fname_params)

     ##run compilation of results
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
