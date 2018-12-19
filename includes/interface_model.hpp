#pragma once
#include <climits>


#include "core_genotype.hpp"
#include "core_phenotype.hpp"
#include "core_evolution.hpp"

using interface_type = uint64_t;
constexpr uint8_t interface_size=CHAR_BIT*sizeof(interface_type);
using BGenotype = std::vector<interface_type>;

class InterfaceAssembly : public PolyominoAssembly<InterfaceAssembly> {
  
public:  
  static double InteractionMatrix(const interface_type, const interface_type);
  static void Mutation(BGenotype& genotype);
  
};


void PrintBindingStrengths();
//extern std::normal_distribution<double> normal_dist;


namespace simulation_params
{
  extern uint8_t model_type,n_tiles,samming_threshold;
  extern uint16_t dissociation_time;
  extern double temperature,binding_threshold,mu_prob;
}

namespace interface_model
{   
  interface_type ReverseBits(interface_type v);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  /* ASSEMBLY */
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, FitnessPhenotypeTable* pt,Phenotype_ID& pid,std::set<InteractionPair>& pid_interactions);
  
}
void RandomiseGenotype(BGenotype& genotype);

BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size);
void EnsureNeutralDisconnections(BGenotype& genotype);

