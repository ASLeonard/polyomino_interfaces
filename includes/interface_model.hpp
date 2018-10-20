#pragma once
#include <cstdint>
#include <climits>


using interface_type = uint64_t;

#include "core_genotype.hpp"
#include "core_phenotype.hpp"
#include "core_evolution.hpp"

class InterfaceAssembly : public PolyominoAssembly<InterfaceAssembly> {
public:
  //thread_local std::vector<double> A::v {1,2,3};
  static double InteractionMatrix(const interface_type, const interface_type);
};

using BGenotype = std::vector<interface_type>;

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
  extern double temperature,binding_threshold;
}

namespace interface_model
{   
  interface_type ReverseBits(interface_type v);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  /* ASSEMBLY */
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, FitnessPhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions);
  
}
void RandomiseGenotype(BGenotype& genotype);

struct GenotypeMutator { 
  GenotypeMutator(double mu) : interface_indices(model_params::interface_size),b_dist(model_params::interface_size,mu) {std::iota(interface_indices.begin(),interface_indices.end(),0);}
  void operator()(BGenotype& binary_genotype) {
    for(interface_type& base : binary_genotype) {
      std::shuffle(interface_indices.begin(), interface_indices.end(), RNG_Engine);
      const uint8_t num_mutations=b_dist(RNG_Engine);
      for(uint8_t nth=0;nth<num_mutations;++nth) 
        base ^= (interface_type(1) << interface_indices[nth]);
    }
  }
      
private:
  std::vector<uint8_t> interface_indices;
  std::binomial_distribution<uint8_t> b_dist;
};

BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size);
void EnsureNeutralDisconnections(BGenotype& genotype,GenotypeMutator& mutator);

