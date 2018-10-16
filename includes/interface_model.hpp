#include "core_evolution.hpp"
#include "core_assembly.hpp"
#include "core_genotype.hpp"


#include <functional>
#include <climits>

using interface_type = uint64_t;
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

std::array<double,model_params::interface_size+1> GenBindingProbsLUP();

bool InteractionMatrix(const interface_type face_1,const interface_type face_2);
double BindingStrength(const interface_type face_1,const interface_type face_2);


namespace interface_model
{   
  interface_type ReverseBits(interface_type v);
  uint8_t SammingDistance(interface_type face1,interface_type face2);

  /* ASSEMBLY */
  double PolyominoAssemblyOutcome(BGenotype& binary_genome, FitnessPhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions);
  
}
