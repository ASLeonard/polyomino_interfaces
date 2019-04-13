#pragma once
#include <climits>

#include "core_genotype.hpp"
#include "core_phenotype.hpp"
#include "core_evolution.hpp"

//using interfaces of 64 bits length, can use any width of unsigned integer
using interface_type = uint64_t;

//set genotype to be a vector of these elements
constexpr uint8_t interface_size=CHAR_BIT*sizeof(interface_type);
using BGenotype = std::vector<interface_type>;

//extension of base assembly (in polyomino_core/include/core_genotype)
class InterfaceAssembly : public PolyominoAssembly<InterfaceAssembly> {

protected:
  //binding likelihood for a given strength
  inline static std::array<double,interface_size+1> binding_probabilities{};
  
public:
  //helper methods for binding strengths
  static void SetBindingStrengths();
  static void PrintBindingStrengths();
  
  //produces a random interface according to its type
  inline static thread_local auto GenRandomSite = []() {return std::uniform_int_distribution<interface_type>()(RNG_Engine);};
  
  //methods requiring extension
  static double InteractionMatrix(const interface_type, const interface_type);
  static void Mutation(BGenotype& genotype);  
};

//parameters used
namespace simulation_params
{
  extern uint8_t model_type,n_tiles,samming_threshold;
  extern uint16_t dissociation_time;
  extern double temperature,binding_threshold,mu_prob;
}


namespace interface_model
{
  //main helper functions to manipulate interfaces in this model, reversing an interface and calculate the unnormalised strength between them
  interface_type ReverseBits(interface_type v);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  //ASSEMBLY wrapper
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, FitnessPhenotypeTable* pt,Phenotype_ID& pid,std::set<InteractionPair>& pid_interactions); 
}

//helper functions for simulations
BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size);
void EnsureNeutralDisconnections(BGenotype& genotype);

