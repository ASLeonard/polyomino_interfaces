#include "core_phenotype.hpp"

#include <functional>
#include <random>
#include <climits>


#include <set>
#include <array>

#include <iostream>


typedef uint64_t interface_type;
typedef std::vector<interface_type> BGenotype;
typedef std::pair<uint8_t,uint8_t> interaction_pair;

namespace model_params
{
  constexpr uint8_t interface_size=CHAR_BIT*sizeof(interface_type);
}

extern thread_local std::mt19937 RNG_Engine;
extern std::normal_distribution<double> normal_dist;
extern std::array<double,model_params::interface_size+1> binding_probabilities;

namespace simulation_params
{
  extern uint8_t model_type,n_tiles,samming_threshold;
  extern uint16_t dissociation_time;
  extern bool fixed_seed;
  extern double temperature,binding_threshold,fitness_factor;
}

std::array<double,model_params::interface_size+1> GenBindingProbsLUP();

void StripNoncodingGenotype(BGenotype& genotype);
uint8_t CountActiveInterfaces(const BGenotype& genotype);
std::vector<std::pair<interaction_pair,double> > GetEdgePairs(const BGenotype& genotype);
/* SPATIAL */
Phenotype SpatialGrid(std::vector<int8_t>& placed_tiles);


uint8_t PhenotypeSymmetryFactor(std::vector<uint8_t>& original_shape, uint8_t dx, uint8_t dy);


namespace interface_model
{
  struct PotentialTileSite {
    interaction_pair bonding_pair;
    std::array<int8_t,3> site_information;
    PotentialTileSite(interaction_pair bp,  int8_t x,int8_t y, int8_t f)
    {bonding_pair=bp;site_information={x,y,f};}
  };
  struct PotentialTileSites {
    std::vector<PotentialTileSite> sites;
    std::vector<double> strengths;
  };
  
  struct InterfacePhenotypeTable;
  
  interface_type ReverseBits(interface_type v);
  //uint8_t ArbitraryPopcount(interface_type face1);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  /* ASSEMBLY */
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, InterfacePhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions);
  std::vector<int8_t> AssemblePolyomino(const std::vector<std::pair<interaction_pair,double> > edges,const int8_t seed,const size_t UNBOUND_LIMIT, std::set<interaction_pair>& interacting_indices);
 
  void ExtendPerimeter(const std::vector<std::pair<interaction_pair,double> >& edges,uint8_t tile_detail, int8_t x,int8_t y, std::vector<int8_t>& placed_tiles,PotentialTileSites& perimeter_sites);
    

  struct InterfacePhenotypeTable : PhenotypeTable {
    std::unordered_map<uint8_t,std::vector<double> > phenotype_fitnesses{{0,{0}}};
         
    /*! Replace previously undiscovered phenotype IDs with new phenotype ID */
    void RelabelPhenotypes(std::vector<Phenotype_ID >& pids,std::map<Phenotype_ID, std::set<interaction_pair> >& p_ints) {
      for(auto x_iter=new_phenotype_xfer.begin();x_iter!=new_phenotype_xfer.end();x_iter+=3) {
        p_ints[std::make_pair(*x_iter,*(x_iter+2))].insert(p_ints[std::make_pair(*x_iter,*(x_iter+1))].begin(),p_ints[std::make_pair(*x_iter,*(x_iter+1))].end());
        phenotype_fitnesses[*x_iter].emplace_back(std::gamma_distribution<double>(*(x_iter)*2,.5*std::pow(*x_iter,-.5))(RNG_Engine));
      }
      PhenotypeTable::RelabelPhenotypes(pids);
    }

    std::map<Phenotype_ID,uint16_t> PhenotypeFrequencies(std::vector<Phenotype_ID >& pids) {
      std::map<Phenotype_ID, uint16_t> ID_counter;
      for(std::vector<Phenotype_ID >::const_iterator ID_iter = pids.begin(); ID_iter!=pids.end(); ++ID_iter) {
	if(ID_iter->second < known_phenotypes[ID_iter->first].size())
	  ++ID_counter[std::make_pair(ID_iter->first,ID_iter->second)];
        else
          ++ID_counter[NULL_pid];
      }
      return ID_counter;
    } 
    
    /* Add fitness contribution from each phenotype */
    double GenotypeFitness(std::map<Phenotype_ID,uint16_t> ID_counter) {
      double fitness=0;
      for(auto kv : ID_counter)
        if(kv.second>=ceil(model_params::UND_threshold*model_params::phenotype_builds))
          fitness+=phenotype_fitnesses[kv.first.first][kv.first.second] * std::pow(static_cast<double>(kv.second)/model_params::phenotype_builds,simulation_params::fitness_factor);     
      return fitness;
    }

    double SingleFitness(Phenotype_ID pid,uint16_t commonness) {
      return phenotype_fitnesses[pid.first][pid.second] * std::pow(static_cast<double>(commonness)/model_params::phenotype_builds,simulation_params::fitness_factor);     
    }

    void ReassignFitness() {
      for(std::unordered_map<uint8_t,std::vector<double> >::iterator fit_iter=phenotype_fitnesses.begin();fit_iter!=phenotype_fitnesses.end();++fit_iter) {
	if(fit_iter->first) {
	  std::gamma_distribution<double> fitness_dist(fit_iter->first*2,.5*std::pow(fit_iter->first,-.5));
	  for(double& fitness : fit_iter->second)
	    fitness=fitness_dist(RNG_Engine);
	}
      }
    }

  };/*! end struct */
  
}
