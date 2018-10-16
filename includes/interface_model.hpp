//#include "core_phenotype.hpp"
#include "core_assembly.hpp"

#include <functional>
#include <climits>


#include <set>
#include <array>

#include <iostream>


typedef uint64_t interface_type;
typedef std::vector<interface_type> BGenotype;


namespace model_params
{
  constexpr uint8_t interface_size=CHAR_BIT*sizeof(interface_type);
}

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
  struct InterfacePhenotypeTable : PhenotypeTable {
    std::unordered_map<uint8_t,std::vector<double> > phenotype_fitnesses{{0,{0}}};
         
    /*! Replace previously undiscovered phenotype IDs with new phenotype ID */
    void RelabelPhenotypes(std::vector<Phenotype_ID >& pids,std::map<Phenotype_ID, std::set<interaction_pair> >& p_ints);

    std::map<Phenotype_ID,uint16_t> PhenotypeFrequencies(std::vector<Phenotype_ID >& pids);
    
    /* Add fitness contribution from each phenotype */
    double GenotypeFitness(std::map<Phenotype_ID,uint16_t> ID_counter);

    double SingleFitness(Phenotype_ID pid,uint16_t commonness);
    /*
    void ReassignFitness() {
      for(std::unordered_map<uint8_t,std::vector<double> >::iterator fit_iter=phenotype_fitnesses.begin();fit_iter!=phenotype_fitnesses.end();++fit_iter) {
	if(fit_iter->first) {
	  std::gamma_distribution<double> fitness_dist(fit_iter->first*2,.5*std::pow(fit_iter->first,-.5));
	  for(double& fitness : fit_iter->second)
	    fitness=fitness_dist(RNG_Engine);
	}
      }
    }
    */

  };/*! end struct */
  
  
 
  
  interface_type ReverseBits(interface_type v);
  //uint8_t ArbitraryPopcount(interface_type face1);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  /* ASSEMBLY */
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, InterfacePhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions);
  //std::vector<int8_t> AssemblePolyomino(const std::vector<std::pair<interaction_pair,double> > edges,const int8_t seed,const size_t UNBOUND_LIMIT, std::set<interaction_pair>& interacting_indices);
 
  //void ExtendPerimeter(const std::vector<std::pair<interaction_pair,double> >& edges,uint8_t tile_detail, int8_t x,int8_t y, std::vector<int8_t>& placed_tiles,PotentialTileSites& perimeter_sites);
    

  
  
}
