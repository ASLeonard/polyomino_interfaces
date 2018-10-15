#include "interface_model.hpp"
#include <functional>

namespace simulation_params
{
  uint16_t population_size=100,fitness_period=100;
  uint32_t generation_limit=100,independent_trials=1,run_offset=0;
  bool random_initilisation=true;
  double mu_prob=1,fitness_jump=2,fitness_rise=10;
}


struct PopulationGenotype {
  std::vector<interface_type> genotype;
  Phenotype_ID pid;
  PopulationGenotype(void) : genotype(simulation_params::n_tiles*4), pid{1,0} {};
};

void ReducedModelTable(interface_model::InterfacePhenotypeTable* pt);

//void RandomStrings();

/* Fitness selection */
std::vector<uint16_t> RouletteWheelSelection(std::vector<double>& fitnesses);

/* Main evolution runners */
void EvolvePopulation(std::string run_details); 
void EvolutionRunner();
void SetRuntimeConfigurations(int argc, char* argv[]);

BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size);
void SampleSupport();
void SampleMutual();

struct DynamicFitnessLandscape {
  DynamicFitnessLandscape(interface_model::InterfacePhenotypeTable* pt_in,uint16_t period, uint16_t rise) : pt_iter(pt_in),period(period) {
    sharpness=rise/std::log(20);
  }
  void operator()(uint32_t generation) {
    if(generation%period==0) 
      std::rotate(pid_cyc.begin(),pid_cyc.begin()+1,pid_cyc.end());
    pt_iter->phenotype_fitnesses[pid_cyc[0]][0]=std::exp(-double(generation%period)/sharpness);
    pt_iter->phenotype_fitnesses[pid_cyc[1]][0]=1-std::exp(-double(generation%period)/sharpness);
  }

private:
  std::array<uint8_t,2> pid_cyc{12,10};
  interface_model::InterfacePhenotypeTable* pt_iter;
  uint16_t period;
  double sharpness;
};

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
 
void EnsureNeutralDisconnections(BGenotype& genotype, GenotypeMutator& mutator);
