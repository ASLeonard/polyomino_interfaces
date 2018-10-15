import numpy as np
from numpy import linalg as LA

def setBasePath(path):
     default_file='/{}_Y{:.6f}_T{:.6f}_Mu{:.6f}_Gamma{:.6f}_Run{}.txt'
     global BASE_PATH
     if path=='scratch':
          BASE_PATH='/scratch/asl47/Data_Runs/Bulk_Data'
     elif path=='rscratch':
         BASE_PATH='/rscratch/asl47/Pickles'
     else:
         BASE_PATH='../'
     BASE_PATH+=default_file
         
def setLength(length):
     global interface_length
     interface_length=length
     global interface_type
     interface_type={8:np.uint8,16:np.uint16,32:np.uint32,64:np.uint64}[interface_length]

setBasePath('scratch')
setLength(64)
             
def BindingStrength(base1,base2):
     return 1-bin(np.bitwise_xor(interface_type(base1),reverseBits(base2))).count('1')/interface_length

def reverseBits(value):
    return ~interface_type(int(('{:0'+str(interface_length)+'b}').format(value)[::-1],2))

class Interactions(object):
     __slots__ = ('bonds','strengths')
     def __init__(self,bonds=None,strengths=None):
          self.bonds=bonds
          self.strengths=strengths
     def __iter__(self):
          for b,s in zip(self.bonds,self.strengths):
               yield (b,s)

         
def LoadEvolutionHistory(S_star,t,mu,gamma,run):
     phenotype_IDs,selections=[],[]
     for lc,line in enumerate(open(BASE_PATH.format('PhenotypeHistory',S_star,t,mu,gamma,run))):
          converted=[int(i) for i in line.split()]
          if lc%2:
               selections.append(converted)
          else:
               phenotype_IDs.append(list(zip(*(iter(converted),) * 2)))

     return ObjArray(phenotype_IDs),np.array(selections,np.uint16)

def LoadGenotypeHistory(S_star,t,mu,gamma,run):
    genotypes=[]
    for line in open(BASE_PATH.format('GenotypeHistory',S_star,t,mu,gamma,run)):
        genotypes.append([[interface_type(int(f)) for f in genotype.split()] for genotype in line.split('x')][:-1])
    return ObjArray(genotypes)

def LoadStrengthHistory(S_star,t,mu,gamma,run):
     strengths=[]
     for line in open(BASE_PATH.format('Strengths',S_star,t,mu,gamma,run)):
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

def LoadPhenotypeTable(S_star,t,mu,gamma,run):
     phenotype_table= sorted([[int(i) for i in line.split()] for line in open(BASE_PATH.format('Phenotypes',S_star,t,mu,gamma,run))],key=lambda z: z[0])
     return {tuple(px[:2]): tuple(px[2:]) for px in phenotype_table}

def LoadAll(mu,S_star,t,gamma,run):
     st=LoadStrengthHistory(S_star,t,mu,gamma,run)
     p,s=LoadEvolutionHistory(S_star,t,mu,gamma,run)
     pt=LoadPhenotypeTable(S_star,t,mu,gamma,run)
     return (s,p,st,pt)

def ObjArray(data):
     shape=(len(data),len(data[0]))
     nparr=np.empty(shape,dtype=object)
     nparr[:]=data
     return nparr 

""" DRIFT SECTION """
def RandomWalk(I_size=64,n_steps=1000,phi=0.5,S_star=0.6,analytic=False):
     s_hats=np.linspace(0,1,I_size+1)
     N=int(I_size*(1-S_star))+1
     
     if analytic:
          analytic_states=__getSteadyStates(N,phi,s_hats[-N:])[1]
          return sum(s_hats[-N:]*analytic_states)
     
     states=np.array([1]+[0]*(N-1),dtype=float)
     progressive_states=[sum(s_hats[-N:]*states)]

     for i in range(n_steps):
          states=__updateStates(states,s_hats[-N:],phi)
          progressive_states.append(sum(s_hats[-N:]*states))
     return progressive_states

def __updateStates(states,val,phi=0.5):
     states_updating=states.copy()
     for i in range(states.shape[0]):
          states_updating[i]-=states[i]*phi
          if i!=0:
               states_updating[i]+=states[i-1]*phi*(1-val[i-1])
          if i!=states.shape[0]-1:
               states_updating[i]+=states[i+1]*phi*val[i+1]
     return states_updating/sum(states_updating)     

def __getSteadyStates(N_states,mu,val):
     rows=[[1-mu,mu*(1-val[0])]+[0]*(N_states-2)]
     for i in range(1,N_states-1):
          rows.append([0]*(i-1)+[mu*val[i],1-mu,mu*(1-val[i])]+[0]*(N_states-2-i))
     rows.append([0]*(N_states-2)+[mu*val[-1],1-mu])
     matrix= np.vstack(rows).T 
     eigval,eigvec=LA.eig(matrix)
     va=max(eigval)
     ve=eigvec.T[np.argmax(eigval)]
     return va,ve/sum(ve)

def good2(a,b):
     return 1/(1+2*b)*(2*b+2/(a+2)*(1+a*1/(a+2)))
     
def good1(a,b):
     return 2*b/(1+2*b)*good2(a,b) + 1/(1+2*b)*(good2(a,b)/(2+a)+2/(a+2)**2 +2*a/(2+a)**3+a*1/(2+a)**2*good2(a,b)+2*a/(2+a)**3)

def goodStar(a,b):
     return 1/(a+2)*good1(a,b)*(1+1*a/(2+a)) + (1/(2+a))**2 *good2(a,b) *(1+2*a/(2+a))+2/((2+a)**3)*(1+3*a/(2+a))

def goodSecondSeed(a,b):
     return 2*b/(2*b+1)*good1(a,b) + 1/(2*b+1)*goodStar(a,b)

def goodFirstSeed(a,b):
     return 1/(2*a*b+1)*(goodStar(a,b)+2*a*b/(2*a*b+1)/(a+2)*(good1(a,b)+1/(a+2)*(good2(a,b)+2/(a+2))))

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
     
def calcTransitionParams(evo_strs,T,S_star):
     param_dict={}

     if (4,1) in evo_strs:
          if (4,0) in evo_strs:
               param_dict[(1,(4,1),(4,0))]=(evo_strs[(4,0)][(2,2)][-1]/S_star)**T
               param_dict[(0,(4,1),(4,0))]=(evo_strs[(4,1)][(4,3)][0]/evo_strs[(4,1)][(3,3)][0])**T
          if (2,0) in evo_strs:
               param_dict[(1,(4,1),(2,0))]=(S_star/evo_strs[(2,0)][(1,1)][-1])**T
               param_dict[(0,(4,1),(2,0))]=(evo_strs[(4,1)][(4,4)][0]/evo_strs[(4,1)][(3,4)][0])**T

     if (16,0) in evo_strs:
          if (4,1) in evo_strs:
               param_dict[(1,(16,0),(4,1))]=(evo_strs[(4,1)][(4,4)][-1]/evo_strs[(4,1)][(3,3)][-1])**T
               param_dict[(0,(16,0),(4,1))]=(evo_strs[(16,0)][(4,2)][0]/evo_strs[(16,0)][(3,2)][0])**T
          if (8,0) in evo_strs:
               param_dict[(1,(16,0),(8,0))]=(S_star/evo_strs[(8,0)][(1,1)][-1])**T
               param_dict[(0,(16,0),(8,0))]=(evo_strs[(16,0)][(4,4)][0]/evo_strs[(16,0)][(3,4)][0])**T
          param_dict[(0,(16,0),(16,0))]=(evo_strs[(16,0)][(4,4)][-1]/evo_strs[(16,0)][(3,4)][-1])**T
          
     if (12,0) in evo_strs:
          if (4,1) in evo_strs:
               param_dict[(1,(12,0),(4,1))]=((evo_strs[(4,1)][(4,4)][-1]/S_star)**T,(S_star/evo_strs[(4,1)][(3,3)][-1])**T)
               param_dict[(0,(12,0),(4,1))]=((evo_strs[(12,0)][(4,2)][0]/evo_strs[(12,0)][(2,2)][0])**T,(evo_strs[(12,0)][(2,2)][0]/evo_strs[(12,0)][(3,2)][0])**T)
          if (8,0) in evo_strs:
               param_dict[(1,(12,0),(8,0))]=((S_star/evo_strs[(8,0)][(2,2)][-1])**T,(evo_strs[(8,0)][(2,2)][-1]/evo_strs[(8,0)][(1,2)][-1])**T)
               param_dict[(0,(12,0),(8,0))]=((evo_strs[(12,0)][(4,4)][0]/evo_strs[(12,0)][(2,4)][0])**T,(evo_strs[(12,0)][(2,4)][0]/evo_strs[(12,0)][(3,4)][0])**T)
          param_dict[(0,(12,0),(12,0))]=((evo_strs[(12,0)][(4,4)][-1]/evo_strs[(12,0)][(2,2)][-1])**T,(evo_strs[(12,0)][(2,2)][-1]/evo_strs[(12,0)][(3,2)][-1])**T)     
     
     
     return param_dict