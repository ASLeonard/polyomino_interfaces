#include "interface_simulator.hpp"
#include <iostream>

constexpr bool BINARY_WRITE_FILES=false;
bool KILL_BACK_MUTATIONS=false;
const std::string file_base_path="//scratch//asl47//Data_Runs//Bulk_Data//";
const std::map<Phenotype_ID,uint8_t> phen_stages{{{0,0},0},{{10,0},4},{{1,0},1},{{2,0},2},{{4,0},2},{{4,1},3},{{8,0},3},{{12,0},4},{{16,0},4}};

namespace simulation_params {
  uint16_t population_size=100;
  double fitness_factor=1;
}

void EvolutionRunner() {
  /*!PYTHON INFORMATION*/
  const std::string py_exec = "python3 ";
  const std::string py_loc = "~/Documents/PolyDev/polyomino_interfaces/scripts/interface_analysis.py ";
  const std::string py_mode="internal "+std::to_string(simulation_params::model_type);
  
  const std::string py_CALL=py_exec + py_loc + py_mode + " "+std::to_string(BINARY_WRITE_FILES)+" ";
  const std::string python_params=" "+std::to_string(simulation_params::binding_threshold)+" "+std::to_string(simulation_params::temperature)+" "+std::to_string(simulation_params::mu_prob)+" "+std::to_string(simulation_params::fitness_factor)+" "+std::to_string(simulation_params::population_size);

  const uint16_t N_runs=simulation_params::independent_trials;
#pragma omp parallel for schedule(dynamic) 
  for(uint16_t r=0;r < N_runs;++r) {
    EvolvePopulation("_Run"+std::to_string(r+simulation_params::run_offset));
    /*!PYTHON CALL*/
    std::system((py_CALL+std::to_string(r)+python_params).c_str());
    /*!PYTHON CALL*/

    
  }
  //python3 ~/Documents/PolyDev/polyomino_interfaces/scripts/interface_analysis.py "external" $Model $Thresh $T $Mu $Gamma $RUNS
}
void ReducedModelTable(FitnessPhenotypeTable* pt) {
  pt->FIXED_TABLE=true;
  KILL_BACK_MUTATIONS=true;
  pt->known_phenotypes[1].emplace_back(Phenotype{1,1, {1}});
  pt->known_phenotypes[2].emplace_back(Phenotype{2,1, {1, 5}});
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 3}});
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 5}});
  pt->known_phenotypes[10].emplace_back(Phenotype{4,3, {0, 1, 2, 0, 1, 5, 6, 2, 4, 3, 7, 3}});
  pt->known_phenotypes[8].emplace_back(Phenotype{4,4, {0, 0, 1, 0, 4, 5, 6, 0, 0, 8, 7, 2, 0, 3, 0, 0}});
  pt->known_phenotypes[12].emplace_back(Phenotype{4,4, {0, 1, 2, 0, 1, 5, 6, 2, 4, 8, 7, 3, 0, 4, 3, 0}});
  pt->known_phenotypes[16].emplace_back(Phenotype{4,4, {1, 2, 1, 2, 4, 5, 6, 3, 1, 8, 7, 2, 4, 3, 4, 3}});

  const double base_multiplier=simulation_params::fitness_jump;
  pt->phenotype_fitnesses[1].emplace_back(1);
  pt->phenotype_fitnesses[2].emplace_back(std::pow(base_multiplier,1));
  pt->phenotype_fitnesses[4].emplace_back(std::pow(base_multiplier,1));
  pt->phenotype_fitnesses[4].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[8].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[10].emplace_back(0);
  pt->phenotype_fitnesses[12].emplace_back(std::pow(base_multiplier,3));
  pt->phenotype_fitnesses[16].emplace_back(std::pow(base_multiplier,3));
}

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



void EvolvePopulation(std::string run_details) {
  std::string file_simulation_details=BINARY_WRITE_FILES ? run_details+".BIN" : "_Y"+std::to_string(simulation_params::binding_threshold)+"_T"+ std::to_string(simulation_params::temperature) +"_Mu"+std::to_string(simulation_params::mu_prob)+"_Gamma"+std::to_string(simulation_params::fitness_factor)+run_details+".txt";
    
  std::ofstream fout_strength(file_base_path+"Strengths"+file_simulation_details,BINARY_WRITE_FILES ? std::ios::binary :std::ios::out);
  std::ofstream fout_phenotype(file_base_path+"PhenotypeTable"+run_details+".BIN",std::ios::out);  
  std::ofstream fout_selection_history(file_base_path+"Selections"+file_simulation_details,BINARY_WRITE_FILES ? std::ios::binary :std::ios::out);    
  std::ofstream fout_phenotype_IDs(file_base_path+"PIDs"+file_simulation_details,BINARY_WRITE_FILES ? std::ios::binary :std::ios::out );
  
  
  std::vector<double> population_fitnesses(simulation_params::population_size);
  std::vector<PopulationGenotype> evolving_population(simulation_params::population_size),reproduced_population(simulation_params::population_size);
  
  FitnessPhenotypeTable pt = FitnessPhenotypeTable();
  DynamicFitnessLandscape dfl(&pt,simulation_params::fitness_period,simulation_params::fitness_rise);

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
  
  std::set<InteractionPair> pid_interactions;
  BGenotype assembly_genotype;
  Phenotype_ID prev_ev;
  std::vector<uint8_t> binary_pids,binary_strengths;
  std::vector<uint16_t> binary_selections;
  if constexpr (BINARY_WRITE_FILES) {
    binary_pids.reserve(2*simulation_params::population_size);
    binary_strengths.reserve(12*simulation_params::population_size);
    binary_selections.reserve(simulation_params::population_size);
  }
  
  
  for(uint32_t generation=0;generation<simulation_params::generation_limit;++generation) { /*! MAIN EVOLUTION LOOP */

    if(simulation_params::model_type==2) {
      dfl(generation);
      //std::cout<<"12: "<<pt.phenotype_fitnesses[12][0]<<" /10: "<<pt.phenotype_fitnesses[10][0]<<std::endl;
    }

    uint16_t nth_genotype=0;
    for(PopulationGenotype& evolving_genotype : evolving_population) { /*! GENOTYPE LOOP */
      
      InterfaceAssembly::Mutation(evolving_genotype.genotype);

     
      if(simulation_params::model_type==1)
        EnsureNeutralDisconnections(evolving_genotype.genotype);         
      const std::vector<std::pair<InteractionPair,double> > edges = InterfaceAssembly::GetActiveInterfaces(evolving_genotype.genotype);
      assembly_genotype=evolving_genotype.genotype;
      prev_ev=evolving_genotype.pid;


      population_fitnesses[nth_genotype]=interface_model::PolyominoAssemblyOutcome(assembly_genotype,&pt,evolving_genotype.pid,pid_interactions);
      
      //std::cout<<+evolving_genotype.pid.first<<","<<+evolving_genotype.pid.second<<std::endl;
      
      if((simulation_params::model_type==12 && assembly_genotype.size()/4 != simulation_params::n_tiles) || (KILL_BACK_MUTATIONS && prev_ev!=evolving_genotype.pid && phen_stages.at(prev_ev)>=phen_stages.at(evolving_genotype.pid))) {
        population_fitnesses[nth_genotype]=0;
        evolving_genotype.pid=NULL_pid;
        pid_interactions.clear();        
      }
      ++nth_genotype;



      if constexpr (BINARY_WRITE_FILES) {
        binary_pids.emplace_back(evolving_genotype.pid.first);
        binary_pids.emplace_back(evolving_genotype.pid.second);
        for(auto x : pid_interactions)
          binary_strengths.insert(binary_strengths.end(),{x.first,x.second,interface_model::SammingDistance(assembly_genotype[x.first],assembly_genotype[x.second])});
        binary_strengths.emplace_back(255);
      }
      else {
        for(auto x : pid_interactions)
          fout_strength<<+x.first<<" "<<+x.second<<" "<<+interface_model::SammingDistance(assembly_genotype[x.first],assembly_genotype[x.second])<<".";
        fout_strength<<",";
        fout_phenotype_IDs << +evolving_genotype.pid.first <<" "<<+evolving_genotype.pid.second<<" ";
      }
    } /*! END GENOTYPE LOOP */

    /*! SELECTION */
    uint16_t nth_repro=0;
    for(uint16_t selected : RouletteWheelSelection(population_fitnesses)) {
      reproduced_population[nth_repro++]=evolving_population[selected];
      if constexpr (BINARY_WRITE_FILES)
        binary_selections.emplace_back(selected);
      else
        fout_selection_history<<+selected<<" ";
    }
    evolving_population.swap(reproduced_population);

  
    if constexpr (BINARY_WRITE_FILES) {
      BinaryWriter(fout_phenotype_IDs,binary_pids);
      binary_pids.clear();
      BinaryWriter(fout_selection_history,binary_selections);
      binary_selections.clear();
      BinaryWriter(fout_strength,binary_strengths);
      binary_strengths.clear();
    }
    else {
      fout_selection_history<<"\n";
      fout_phenotype_IDs<<"\n";
      fout_strength<<"\n";
    }
    
  } /* END EVOLUTION LOOP */
  pt.PrintTable(fout_phenotype);
  
}

/********************/
/*******!MAIN!*******/
/********************/


int main(int argc, char* argv[]) {
  char run_option;
  if(argc<=1) {
    std::cout<<"Too few arguments"<<std::endl;
    return 0;
  }

  run_option=argv[1][1];
  SetRuntimeConfigurations(argc,argv);
  
  switch(run_option) {
  case 'E':
    EvolutionRunner();
    break;
  case '?':
    InterfaceAssembly::PrintBindingStrengths();
    break;
  case 'H':
  default:
    std::cout<<"Polyomino interface model\n**Simulation Parameters**\nN: number of tiles\nP: population size\nK: generation limit\nB: number of phenotype builds\n";
    std::cout<<"\n**Model Parameters**\nU: mutation probability (per interface)\nT: temperature\nI: unbound size factor\nA: misbinding rate\nM: Fitness factor\n";
    std::cout<<"\n**Run options**\nR: evolution without fitness\nE: evolution with fitness\n";
    break;
  }
  return 0;
}

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
        
        //DONE IN INIT FILE
      case 'M': simulation_params::mu_prob=std::stod(argv[arg+1]);break;
      case 'Y': simulation_params::binding_threshold=std::stod(argv[arg+1]);break;
      case 'T': simulation_params::temperature=std::stod(argv[arg+1]);break;
        
      case 'S': InterfaceAssembly::free_seed=std::stoi(argv[arg+1])>0;break;
      case 'A': simulation_params::model_type=std::stoi(argv[arg+1]);break;   
      case 'H': simulation_params::dissociation_time=std::stoi(argv[arg+1]);break;
        
      case 'F': simulation_params::fitness_factor=std::stod(argv[arg+1]);break;
      case 'J': simulation_params::fitness_jump=std::stod(argv[arg+1]);break;
        
      case 'O': simulation_params::fitness_period=std::stod(argv[arg+1]);break;
      case 'G': simulation_params::fitness_rise=std::stod(argv[arg+1]);break;
        
      default: std::cout<<"Unknown Parameter Flag: "<<argv[arg][1]<<std::endl;
      }
    }
  InterfaceAssembly::SetBindingStrengths();
  }
}


