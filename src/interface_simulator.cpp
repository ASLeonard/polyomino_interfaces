#include "interface_simulator.hpp"
#include <iostream>

//internal flag to limit fatal mutations to less fit phenotypes
bool KILL_BACK_MUTATIONS=false;

//set the file path to be where the (large) files will be written, by default the location where the 
const std::string file_base_path="";

//wrapper to run many (parallel) independent evolutions
void EvolutionRunner() {  
  const uint16_t N_runs=simulation_params::independent_trials;
#pragma omp parallel for schedule(dynamic) 
  for(uint16_t r=0;r < N_runs;++r)
    EvolvePopulation("_Run"+std::to_string(r+simulation_params::run_offset));
}

//populate fixed phenotype table with example system
void ReducedModelTable(FitnessPhenotypeTable* pt) {
  FinalModelTable(pt);
  KILL_BACK_MUTATIONS=true;
  pt->known_phenotypes[1].emplace_back(Phenotype{1,1, {1}});
  pt->known_phenotypes[2].emplace_back(Phenotype{2,1, {1, 5}});
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 5}});
  pt->known_phenotypes[8].emplace_back(Phenotype{4,4, {0, 0, 1, 0, 4, 5, 6, 0, 0, 8, 7, 2, 0, 3, 0, 0}});
  pt->known_phenotypes[16].emplace_back(Phenotype{4,4, {1, 2, 1, 2, 4, 5, 6, 3, 1, 8, 7, 2, 4, 3, 4, 3}});

  const double base_multiplier=simulation_params::fitness_jump;
  pt->phenotype_fitnesses[1].emplace_back(1);
  pt->phenotype_fitnesses[2].emplace_back(std::pow(base_multiplier,1));
  pt->phenotype_fitnesses[4][0]=std::pow(base_multiplier,1);
  pt->phenotype_fitnesses[4].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[8].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[12][0]=std::pow(base_multiplier,3);
  pt->phenotype_fitnesses[16].emplace_back(std::pow(base_multiplier,3));
}

//populate fixed phenotype table with only the 12-mer and nondeterministic polyominoes
void FinalModelTable(FitnessPhenotypeTable* pt) {
  pt->FIXED_TABLE=true;
  KILL_BACK_MUTATIONS=false;
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 3}});
  pt->known_phenotypes[12].emplace_back(Phenotype{4,4, {0, 1, 2, 0, 1, 5, 6, 2, 4, 8, 7, 3, 0, 4, 3, 0}});
  pt->known_phenotypes[10].emplace_back(Phenotype{4,3, {0, 1, 2, 0, 1, 5, 6, 2, 4, 3, 7, 3}});
  pt->phenotype_fitnesses[4].emplace_back(0);
  pt->phenotype_fitnesses[12].emplace_back(0);
  pt->phenotype_fitnesses[10].emplace_back(0);
}

//vertical columns of phenotype subset
const std::map<Phenotype_ID,uint8_t> phen_stages{{{0,0},0},{{10,0},4},{{1,0},1},{{2,0},2},{{4,0},2},{{4,1},3},{{8,0},3},{{12,0},4},{{16,0},4}};

//main evolution simulator function
void EvolvePopulation(const std::string& run_details) {

  //create file names and make text files
  std::string file_simulation_details= run_details+".txt";
    
  std::ofstream fout_strength(file_base_path+"Strengths"+file_simulation_details,std::ios::out);
  std::string fname_phenotype(file_base_path+"PhenotypeTable"+run_details+".txt");  
  std::ofstream fout_selection_history(file_base_path+"Selections"+file_simulation_details,std::ios::out);    
  std::ofstream fout_phenotype_IDs(file_base_path+"PIDs"+file_simulation_details,std::ios::out );
  
  //create initial populations and associated objects
  std::vector<double> population_fitnesses(simulation_params::population_size);
  std::vector<PopulationGenotype> evolving_population(simulation_params::population_size),reproduced_population(simulation_params::population_size);
  
  FitnessPhenotypeTable pt = FitnessPhenotypeTable();
  DynamicFitnessLandscape dfl(&pt,simulation_params::fitness_period,simulation_params::fitness_rise);

  //generate populations and phenotype tables based on model
  switch(simulation_params::model_type) {      
  case 2: FinalModelTable(&pt);
    {
      const BGenotype ref_genotype=GenerateTargetGraph({{1,{0,7}},{4,{5}}},simulation_params::n_tiles*4);
      for(auto& species : evolving_population)
        species.genotype=ref_genotype;
    }
    break;
  case 1: ReducedModelTable(&pt);
      [[fallthrough]];
  default:
    for(auto& species : evolving_population)
      InterfaceAssembly::RandomiseGenotype(species.genotype);
    break;
  }
  
  //main evolution loop
  for(uint32_t generation=0;generation<simulation_params::generation_limit;++generation) {


    //if model type is 2, update the dynamic fitness landscape
    if(simulation_params::model_type==2)
      dfl(generation);

    //main genotype loop
    uint16_t nth_genotype=0;
    for(PopulationGenotype& evolving_genotype : evolving_population) {

      //mutate the full genotype
      InterfaceAssembly::Mutation(evolving_genotype.genotype);

      //if model type is 1, swap subunits if only edge is on second subunit
      if(simulation_params::model_type==1)
        EnsureNeutralDisconnections(evolving_genotype.genotype);

      //update edges, genotype, and fitness
      const auto edges = InterfaceAssembly::GetActiveInterfaces(evolving_genotype.genotype);
      BGenotype assembly_genotype=evolving_genotype.genotype;
      Phenotype_ID prev_ev=evolving_genotype.pid;
      std::set<InteractionPair> pid_interactions;
      population_fitnesses[nth_genotype]=interface_model::PolyominoAssemblyOutcome(assembly_genotype,&pt,evolving_genotype.pid,pid_interactions);

      //if there is a fatal loss of fitness or complexity, set pid and fitness to null values so can terminate ancestor tree
      if((simulation_params::model_type==2 && assembly_genotype.size()/4 != simulation_params::n_tiles) || (KILL_BACK_MUTATIONS && prev_ev!=evolving_genotype.pid && phen_stages.at(prev_ev)>=phen_stages.at(evolving_genotype.pid))) {
        population_fitnesses[nth_genotype]=0;
        evolving_genotype.pid=NULL_pid;
        pid_interactions.clear();        
      }

      //increment genotype index
      ++nth_genotype;

      //print generation results to file
      for(auto x : pid_interactions)
	fout_strength<<+x.first<<" "<<+x.second<<" "<<+interface_model::SammingDistance(assembly_genotype[x.first],assembly_genotype[x.second])<<".";
      fout_strength<<",";
      fout_phenotype_IDs << +evolving_genotype.pid.first <<" "<<+evolving_genotype.pid.second<<" ";
    }
    

    //select next population by roulette wheel, write selections to file
    uint16_t nth_repro=0;
    for(uint16_t selected : RouletteWheelSelection(population_fitnesses)) {
      reproduced_population[nth_repro++]=evolving_population[selected];
        fout_selection_history<<+selected<<" ";
    }

    //set new population to the selected population
    evolving_population.swap(reproduced_population);

    //start new line in files for new generation
    fout_selection_history<<"\n";
    fout_phenotype_IDs<<"\n";
    fout_strength<<"\n";
    
    
  }

  //print phenotype table to file
  pt.PrintTable(fname_phenotype);  
}

//MAIN function called from program
int main(int argc, char* argv[]) {

  //requires arguments to run
  char run_option;
  if(argc<=1) {
    std::cout<<"Too few arguments"<<std::endl;
    return 0;
  }

  //set arguments and run option
  run_option=argv[1][1];
  SetRuntimeConfigurations(argc,argv);
  
  switch(run_option) {
  case 'E':
    //primary option
    EvolutionRunner();
    break;
  case '?':
    InterfaceAssembly::PrintBindingStrengths();
    break;
  case 'H':
  default:
    std::cout<<"Polyomino interface model\n**Simulation Parameters**\nN: number of tiles\nP: population size\nK: generation limit\nB: number of phenotype builds\n";
    std::cout<<"\n**Model Parameters**\nA: model type (0 is free evolution, 1 is sample system, 2 is dynamic landscape)\nM: average mutations per genotype\nT: temperature\nY: critical interaction strength\nX: minimum determinism limit\nF: nondeterminism punishment factor\nJ: fitness jump\nO: dynamic landscape period\nG: landscape rise rate\n";
    std::cout<<"\n**Run options**\nD: number of runs\nV: run offset value\n";
    break;
  }
  return 0;
}

//set parameters based on input arguments, if there are fewer than 3 and it isn't a help query, flag it
void SetRuntimeConfigurations(int argc, char* argv[]) {
  if(argc<3 && argv[1][1]!='H')
    std::cout<<"Invalid Parameters"<<std::endl;
  else {
    for(uint8_t arg=2;arg<argc;arg+=2) {
      switch(argv[arg][1]) {
        /*! model basics */
      case 'N': simulation_params::n_tiles=std::stoi(argv[arg+1]);break;
      case 'P': simulation_params::population_size=std::stoi(argv[arg+1]);break;
      case 'K': simulation_params::generation_limit=std::stoi(argv[arg+1]);break;
      case 'B': FitnessPhenotypeTable::phenotype_builds=std::stoi(argv[arg+1]);break;
      case 'X': FitnessPhenotypeTable::UND_threshold=std::stod(argv[arg+1]);break;

        /*! run configurations */
      case 'D': simulation_params::independent_trials=std::stoi(argv[arg+1]);break;
      case 'V': simulation_params::run_offset=std::stoi(argv[arg+1]);break;
      case 'R': simulation_params::random_initilisation=std::stoi(argv[arg+1])>0;break;

        /*! simulation specific */
        

      case 'M': simulation_params::mu_prob=std::stod(argv[arg+1]);break;
      case 'Y': simulation_params::binding_threshold=std::stod(argv[arg+1]);break;
      case 'T': simulation_params::temperature=std::stod(argv[arg+1]);break;
        
      case 'S': InterfaceAssembly::free_seed=std::stoi(argv[arg+1])>0;break;
      case 'A': simulation_params::model_type=std::stoi(argv[arg+1]);break;   
        
      case 'F': simulation_params::fitness_factor=std::stod(argv[arg+1]);break;
      case 'J': simulation_params::fitness_jump=std::stod(argv[arg+1]);break;
        
      case 'O': simulation_params::fitness_period=std::stod(argv[arg+1]);break;
      case 'G': simulation_params::fitness_rise=std::stod(argv[arg+1]);break;
        
      default: std::cout<<"Unknown Parameter Flag: "<<argv[arg][1]<<std::endl;
      }
    }
    
    //after parameters are set, calculate the strength table given the new values
    InterfaceAssembly::SetBindingStrengths();
  }
}


