#include "interface_model.hpp"
#include <functional>

namespace simulation_params
{
  PopulationSize population_size=100;
  uint16_t fitness_period=100;
  uint32_t generation_limit=100,independent_trials=1,run_offset=0;
  bool random_initilisation=true;
  double fitness_jump=2,fitness_rise=10, fitness_factor=1;
}

//population structure to hold genotype and phenotype;
struct PopulationGenotype {
  BGenotype genotype;
  Phenotype_ID pid;
  PopulationGenotype(void) : genotype(simulation_params::n_tiles*4), pid{1,0} {};
};

void ReducedModelTable(FitnessPhenotypeTable* pt);
void FinalModelTable(FitnessPhenotypeTable* pt);

/* Main evolution runners */
void EvolvePopulation(const std::string& run_details); 
void EvolutionRunner();


void SetRuntimeConfigurations(int argc, char* argv[]);


//dynamic fitness landscape, with a fixed period and exponential rise rate
struct DynamicFitnessLandscape {
  DynamicFitnessLandscape(FitnessPhenotypeTable* pt_in,uint16_t period, uint16_t rise) : pt_iter(pt_in),period(period) {
    sharpness=rise/std::log(20);
  }
  void operator()(uint32_t generation) {
    if(generation%period==0) 
      std::rotate(pid_cyc.begin(),pid_cyc.begin()+1,pid_cyc.end());
    pt_iter->phenotype_fitnesses[pid_cyc[0]][0]=std::exp(-double(generation%period)/sharpness);
    pt_iter->phenotype_fitnesses[pid_cyc[1]][0]=1-std::exp(-double(generation%period)/sharpness);
  }

private:
  std::array<uint8_t,2> pid_cyc{10,12};
  FitnessPhenotypeTable* pt_iter;
  uint16_t period;
  double sharpness;
};
