import numpy as np
from numpy import linalg as LA
import os

#############
## GLOBALS ##
#############
BASE_PATH=''
interface_length=64
interface_type=np.uint64

def setBasePath(path):
     default_file='/{}_Run{}.txt'
          
     global BASE_PATH
     
     BASE_PATH=path+default_file
         
def setLength(length):
     global interface_length
     interface_length=length
     global interface_type
     interface_type={8:np.uint8,16:np.uint16,32:np.uint32,64:np.uint64}[interface_length]

             
def BindingStrength(base1,base2):
     return 1-bin(np.bitwise_xor(interface_type(base1),reverseBits(base2))).count('1')/interface_length

def reverseBits(value):
    return ~interface_type(int(('{:0'+str(interface_length)+'b}').format(value)[::-1],2))

class Interactions(object):
     __slots__ = ('bonds','strengths')
     
     def __init__(self,bonds=None,strengths=None):
          self.bonds = bonds or []
          self.strengths = strengths or []
          
     def __iter__(self):
          for b,s in zip(self.bonds,self.strengths):
               yield (b,s)
               
     def __repr__(self):
          return "{} interactions".format(len(self.bonds))
          
def loadSelectionHistory(run):
     selections=[]
     for line in open(BASE_PATH.format('Selections',run)):
          converted=[int(i) for i in line.split()]
          selections.append(converted)
     return np.array(selections,np.uint16)

def loadPIDHistory(run):
     phenotype_IDs=[]
     for line in open(BASE_PATH.format('PIDs',run)):
          converted=[int(i) for i in line.split()]
          phenotype_IDs.append(list(zip(*(iter(converted),) * 2)))

     return np.array(phenotype_IDs,dtype=np.uint8)

def loadStrengthHistory(run):
     strengths=[]
     for line in open(BASE_PATH.format('Strengths',run)):
          row=[]
          for species in line.split(',')[:-1]:
               bond_list=[]
               str_list=[]
               for pairing in species.split('.'):
                    if pairing=='':
                         continue
                    pairing=pairing.split()
                    bond_list.append(tuple(int(i) for i in pairing[:2]))
                    str_list.append(1-int(pairing[2])/interface_length)
               row.append(Interactions(bond_list,str_list))
          strengths.append(row)
                
     return ObjArray(strengths)

def loadPhenotypeTable(run):
     phenotype_table= sorted([[int(i) for i in line.split()] for line in open(BASE_PATH.format('PhenotypeTable',run))],key=lambda z: z[0])
     return {tuple(px[:2]): tuple(px[2:]) for px in phenotype_table}

def loadAllFiles(run,cwd=None):
     setBasePath(cwd or os.getcwd())
     st=loadStrengthHistory(run)
     p=loadPIDHistory(run)
     s=loadSelectionHistory(run)
 
     return (s,p,st)

def ObjArray(data):
     shape=(len(data),len(data[0]))
     nparr=np.empty(shape,dtype=object)
     nparr[:]=data
     return nparr

#################
## RANDOM WALK ##
#################
def RandomWalk(I_size=64,n_steps=1000,phi=0.5,S_star=0.6,analytic=False):
     s_hats=np.linspace(0,1,I_size+1)
     N_states=int(I_size*(1-S_star))+1

     def __getSteadyStates(val):
          rows=[[1-phi,phi*(1-val[0])]+[0]*(N_states-2)]
          for i in range(1,N_states-1):
               rows.append([0]*(i-1)+[phi*val[i],1-phi,phi*(1-val[i])]+[0]*(N_states-2-i))
          rows.append([0]*(N_states-2)+[phi*val[-1],1-phi])
          matrix= np.vstack(rows).T
          eigval,eigvec=LA.eig(matrix)
          ve=eigvec.T[np.argmax(eigval)]
          return max(eigval),ve/sum(ve)
     
     if analytic:
          analytic_states=__getSteadyStates(s_hats[-N_states:])[1]
          return sum(s_hats[-N_states:]*analytic_states)
     
     states=np.array([1]+[0]*(N_states-1),dtype=float)
     progressive_states=[sum(s_hats[-N_states:]*states)]

     def __updateStates(states,val):
          states_updating=states.copy()
          for i in range(states.shape[0]):
               states_updating[i]-=states[i]*phi
               if i!=0:
                    states_updating[i]+=states[i-1]*phi*(1-val[i-1])
               if i!=states.shape[0]-1:
                    states_updating[i]+=states[i+1]*phi*val[i+1]
          return states_updating/sum(states_updating)

     for _ in range(n_steps):
          states=__updateStates(states,s_hats[-N_states:])
          progressive_states.append(sum(s_hats[-N_states:]*states))
     return progressive_states

################
## PHASE SPACE##
################

def HetTet(a,b):
     return 1/(1+2*b)*(2*b+2/(a+2)*(1+a*1/(a+2)))
     
def Branching1(a,b):
     return 2*b/(1+2*b)*HetTet(a,b) + 1/(1+2*b)*(HetTet(a,b)/(2+a)+2/(a+2)**2 +2*a/(2+a)**3+a*1/(2+a)**2*HetTet(a,b)+2*a/(2+a)**3)

def Branching2(a,b):
     return 1/(a+2)*Branching1(a,b)*(1+1*a/(2+a)) + (1/(2+a))**2 *HetTet(a,b) *(1+2*a/(2+a))+2/((2+a)**3)*(1+3*a/(2+a))

def goodSecondSeed(a,b):
     return 2*b/(2*b+1)*Branching1(a,b) + 1/(2*b+1)*Branching2(a,b)

def goodFirstSeed(a,b):
     return 1/(2*a*b+1)*(Branching2(a,b)+2*a*b/(2*a*b+1)/(a+2)*(Branching1(a,b)+1/(a+2)*(HetTet(a,b)+2/(a+2))))

def Twelve(a,b):
     x22=(1-FourOne(a*b,1))*.5
     full=goodSecondSeed(a,b)*.5+goodFirstSeed(a,b)*.5
     half=(1-x22*2-goodFirstSeed(a,b))*.5+(1-goodSecondSeed(a,b))*.5
     return (x22,half,full)

def FourOne(a,seed=None):
     def seed1():
          return sum((2*a)**n/(2*a+1)**(n+1) for n in range(3))
     def seed2():
          return 1
     if seed==1:
          return seed1()
     elif seed==2:
          return seed2()
     else:
           return .5*seed1()+.5*seed2()
     
def calcTransitionParams(evo_strs,transitions,T,S_star):
     param_dict={}

     if (4,1) in evo_strs:
          if (4,0) in transitions[(4,1)]:
               param_dict[(1,(4,1),(4,0))]=(evo_strs[(4,0)][(2,2)][-1]/S_star)**T
               param_dict[(0,(4,1),(4,0))]=(evo_strs[(4,1)][(4,3)][0]/evo_strs[(4,1)][(3,3)][0])**T
          if (2,0) in transitions[(4,1)]:
               param_dict[(1,(4,1),(2,0))]=(S_star/evo_strs[(2,0)][(1,1)][-1])**T
               param_dict[(0,(4,1),(2,0))]=(evo_strs[(4,1)][(4,4)][0]/evo_strs[(4,1)][(3,4)][0])**T

     if (16,0) in evo_strs:
          if (4,1) in transitions[(16,0)]:
               param_dict[(1,(16,0),(4,1))]=(evo_strs[(4,1)][(4,4)][-1]/evo_strs[(4,1)][(3,3)][-1])**T
               param_dict[(0,(16,0),(4,1))]=(evo_strs[(16,0)][(4,2)][0]/evo_strs[(16,0)][(3,2)][0])**T
               param_dict[(0,(16,0),(16,0))]=(evo_strs[(16,0)][(4,2)][-1]/evo_strs[(16,0)][(3,2)][-1])**T
          if (8,0) in transitions[(16,0)]:
               param_dict[(1,(16,0),(8,0))]=(S_star/evo_strs[(8,0)][(1,1)][-1])**T
               param_dict[(0,(16,0),(8,0))]=(evo_strs[(16,0)][(4,4)][0]/evo_strs[(16,0)][(3,4)][0])**T
               param_dict[(0,(16,0),(16,0))]=(evo_strs[(16,0)][(4,4)][-1]/evo_strs[(16,0)][(3,4)][-1])**T
          
          
     if (12,0) in evo_strs:
          if (4,1) in transitions[(12,0)]:
               param_dict[(1,(12,0),(4,1))]=((evo_strs[(4,1)][(4,4)][-1]/S_star)**T,(S_star/evo_strs[(4,1)][(3,3)][-1])**T)
               param_dict[(0,(12,0),(4,1))]=((evo_strs[(12,0)][(4,2)][0]/evo_strs[(12,0)][(2,2)][0])**T,(evo_strs[(12,0)][(2,2)][0]/evo_strs[(12,0)][(3,2)][0])**T)
               param_dict[(0,(12,0),(12,0))]=((evo_strs[(12,0)][(4,2)][-1]/evo_strs[(12,0)][(2,2)][-1])**T,(evo_strs[(12,0)][(2,2)][-1]/evo_strs[(12,0)][(3,2)][-1])**T)
          if (8,0) in transitions[(12,0)]:
               param_dict[(1,(12,0),(8,0))]=((S_star/evo_strs[(8,0)][(2,2)][-1])**T,(evo_strs[(8,0)][(2,2)][-1]/evo_strs[(8,0)][(1,2)][-1])**T)
               param_dict[(0,(12,0),(8,0))]=((evo_strs[(12,0)][(4,4)][0]/evo_strs[(12,0)][(2,4)][0])**T,(evo_strs[(12,0)][(2,4)][0]/evo_strs[(12,0)][(3,4)][0])**T)
               param_dict[(0,(12,0),(12,0))]=((evo_strs[(12,0)][(4,4)][-1]/evo_strs[(12,0)][(2,4)][-1])**T,(evo_strs[(12,0)][(2,4)][-1]/evo_strs[(12,0)][(3,4)][-1])**T)

     return param_dict
