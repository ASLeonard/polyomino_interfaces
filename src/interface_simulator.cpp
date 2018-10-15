#include "interface_simulator.hpp"
bool KILL_BACK_MUTATIONS=false;
const std::string file_base_path="//scratch//asl47//Data_Runs//Bulk_Data//";
auto interface_filler = std::bind(std::uniform_int_distribution<interface_type>(), std::ref(RNG_Engine));
const std::map<Phenotype_ID,uint8_t> phen_stages{{{0,0},0},{{1,0},1},{{2,0},2},{{4,0},2},{{4,1},3},{{8,0},3},{{12,0},4},{{16,0},4}};

void EvolutionRunner() {
  const uint16_t N_runs=simulation_params::independent_trials;
#pragma omp parallel for schedule(dynamic)
  for(uint16_t r=0;r < N_runs;++r) 
    EvolvePopulation("_Run"+std::to_string(r+simulation_params::run_offset));
}
void ReducedModelTable(interface_model::InterfacePhenotypeTable* pt) {
  model_params::FIXED_TABLE=true;
  KILL_BACK_MUTATIONS=true;
  pt->known_phenotypes[1].emplace_back(Phenotype{1,1, {1}});
  pt->known_phenotypes[2].emplace_back(Phenotype{2,1, {1, 5}});
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 3}});
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 5}});
  pt->known_phenotypes[8].emplace_back(Phenotype{4,4, {0, 0, 1, 0, 4, 5, 6, 0, 0, 8, 7, 2, 0, 3, 0, 0}});
  pt->known_phenotypes[12].emplace_back(Phenotype{4,4, {0, 1, 2, 0, 1, 5, 6, 2, 4, 8, 7, 3, 0, 4, 3, 0}});
  pt->known_phenotypes[16].emplace_back(Phenotype{4,4, {1, 2, 1, 2, 4, 5, 6, 3, 1, 8, 7, 2, 4, 3, 4, 3}});

  const double base_multiplier=simulation_params::fitness_jump;
  pt->phenotype_fitnesses[1].emplace_back(1);
  pt->phenotype_fitnesses[2].emplace_back(std::pow(base_multiplier,1));
  pt->phenotype_fitnesses[4].emplace_back(std::pow(base_multiplier,1));
  pt->phenotype_fitnesses[4].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[8].emplace_back(std::pow(base_multiplier,2));
  pt->phenotype_fitnesses[12].emplace_back(std::pow(base_multiplier,3));
  pt->phenotype_fitnesses[16].emplace_back(std::pow(base_multiplier,3));
}

void FinalModelTable(interface_model::InterfacePhenotypeTable* pt) {
  model_params::FIXED_TABLE=true;
  pt->known_phenotypes[4].emplace_back(Phenotype{2,2, {1, 2, 4, 3}});
  pt->known_phenotypes[12].emplace_back(Phenotype{4,4, {0, 1, 2, 0, 1, 5, 6, 2, 4, 8, 7, 3, 0, 4, 3, 0}});
  pt->known_phenotypes[10].emplace_back(Phenotype{4,3, {0, 1, 2, 0, 1, 5, 6, 2, 4, 3, 7, 3}});
  pt->phenotype_fitnesses[4].emplace_back(0);
  pt->phenotype_fitnesses[12].emplace_back(0);
  pt->phenotype_fitnesses[10].emplace_back(0);
}

void EvolvePopulation(std::string run_details) {
  /*! Output files */
  
  std::string file_simulation_details="Y"+std::to_string(simulation_params::binding_threshold)+"_T"+ std::to_string(simulation_params::temperature) +"_Mu"+std::to_string(simulation_params::mu_prob)+"_Gamma"+std::to_string(simulation_params::fitness_factor)+run_details+".txt";
    
  std::ofstream fout_strength(file_base_path+"Strengths_"+file_simulation_details);
  std::ofstream fout_phenotype(file_base_path+"Phenotypes_"+file_simulation_details);  
  std::ofstream fout_phenotype_history(file_base_path+"PhenotypeHistory_"+file_simulation_details);
  
    
    
  std::vector<double> population_fitnesses(simulation_params::population_size);
  std::vector<PopulationGenotype> evolving_population(simulation_params::population_size),reproduced_population(simulation_params::population_size);
  GenotypeMutator mutator(simulation_params::mu_prob/(model_params::interface_size*4*simulation_params::n_tiles));
  
  interface_model::InterfacePhenotypeTable pt = interface_model::InterfacePhenotypeTable();
  DynamicFitnessLandscape dfl(&pt,simulation_params::fitness_period,simulation_params::fitness_rise);
  switch(simulation_params::model_type) {
  case 2: FinalModelTable(&pt);
    {
      const BGenotype ref_genotype=GenerateTargetGraph({{1,{0,7}},{4,{5}}},simulation_params::n_tiles*4);
      //for(auto  aeg : GetEdgePairs(ref_genotype))
      //  std::cout<<+aeg.first.first<<","<<+aeg.first.second<<": "<<aeg.second<<std::endl;
      for(auto& species : evolving_population)
        species.genotype=ref_genotype;
    }
    break;
  case 1: ReducedModelTable(&pt);
    __attribute__ ((fallthrough));
  default:
    for(auto& species : evolving_population)
      do {
        std::generate(species.genotype.begin(),species.genotype.end(),interface_filler);
      }while(CountActiveInterfaces(species.genotype)!=0);
    break;
  }
  
  std::set<interaction_pair> pid_interactions;
  BGenotype assembly_genotype;
  Phenotype_ID prev_ev;
  
  for(uint32_t generation=0;generation<simulation_params::generation_limit;++generation) { /*! MAIN EVOLUTION LOOP */
    if(simulation_params::model_type==2)
      dfl(generation);
    //std::cout<<pt.phenotype_fitnesses[10][0]<<", "<<pt.phenotype_fitnesses[12][0]<<std::endl;
    uint16_t nth_genotype=0;
    for(PopulationGenotype& evolving_genotype : evolving_population) { /*! GENOTYPE LOOP */
      mutator(evolving_genotype.genotype);      
     
      if(simulation_params::model_type==1)
        EnsureNeutralDisconnections(evolving_genotype.genotype,mutator);         
            
      assembly_genotype=evolving_genotype.genotype;
      prev_ev=evolving_genotype.pid;
      population_fitnesses[nth_genotype]=interface_model::PolyominoAssemblyOutcome(assembly_genotype,&pt,evolving_genotype.pid,pid_interactions);
      
      if((simulation_params::model_type==2 && assembly_genotype.size()/4 != simulation_params::n_tiles) || (KILL_BACK_MUTATIONS && prev_ev!=evolving_genotype.pid && phen_stages.at(prev_ev)>=phen_stages.at(evolving_genotype.pid))) {
        population_fitnesses[nth_genotype]=0;
        evolving_genotype.pid=NULL_pid;
        pid_interactions.clear();        
      }
      ++nth_genotype;
      
      for(auto x : pid_interactions)
        fout_strength<<+x.first<<" "<<+x.second<<" "<<+interface_model::SammingDistance(assembly_genotype[x.first],assembly_genotype[x.second])<<".";
      fout_strength<<",";
      
      fout_phenotype_history << +evolving_genotype.pid.first <<" "<<+evolving_genotype.pid.second<<" ";
    } /*! END GENOTYPE LOOP */


    fout_phenotype_history<<"\n";
    fout_strength<<"\n";

    /*! SELECTION */
    uint16_t nth_repro=0;
    for(uint16_t selected : RouletteWheelSelection(population_fitnesses)) {
      reproduced_population[nth_repro++]=evolving_population[selected];
      fout_phenotype_history<<+selected<<" ";
    }
    fout_phenotype_history<<"\n";
    evolving_population.swap(reproduced_population);
  } /* END EVOLUTION LOOP */
  
  pt.PrintTable(fout_phenotype); 
}

std::vector<uint16_t> RouletteWheelSelection(std::vector<double>& fitnesses) {
  std::partial_sum(fitnesses.begin(), fitnesses.end(), fitnesses.begin());
  std::vector<uint16_t> selected_indices(simulation_params::population_size);
  std::uniform_real_distribution<double> random_interval(0,fitnesses.back());
  for(uint16_t nth_selection=0; nth_selection<simulation_params::population_size; ++nth_selection) 
    selected_indices[nth_selection]=static_cast<uint16_t>(std::lower_bound(fitnesses.begin(),fitnesses.end(),random_interval(RNG_Engine))-fitnesses.begin());
  std::sort(selected_indices.begin(),selected_indices.end());
  return selected_indices;
}

/********************/
/*******!MAIN!*******/
/********************/
int main(int argc, char* argv[]) {
  char run_option;
  if(argc<2) {
    std::cout<<"no Params"<<std::endl;
    run_option='H';
  }
  else {
    run_option=argv[1][1];
    SetRuntimeConfigurations(argc,argv);
  }
  
  switch(run_option) {
  case 'E':
    EvolutionRunner();
    break;
  case '?':
    for(auto b : binding_probabilities)
      std::cout<<b<<std::endl;
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
      case 'B': model_params::phenotype_builds=std::stoi(argv[arg+1]);break;
      case 'X': model_params::UND_threshold=std::stod(argv[arg+1]);break;

        /*! run configurations */
      case 'D': simulation_params::independent_trials=std::stoi(argv[arg+1]);break;
      case 'V': simulation_params::run_offset=std::stoi(argv[arg+1]);break;
      case 'R': simulation_params::random_initilisation=std::stoi(argv[arg+1])>0;break;

        /*! simulation specific */
      case 'M': simulation_params::mu_prob=std::stod(argv[arg+1]);break;
      case 'Y': simulation_params::binding_threshold=std::stod(argv[arg+1]);break;
      case 'T': simulation_params::temperature=std::stod(argv[arg+1]);break;
      case 'S': simulation_params::fixed_seed=std::stoi(argv[arg+1])>0;break;
      case 'A': simulation_params::model_type=std::stoi(argv[arg+1]);break;   
      case 'H': simulation_params::dissociation_time=std::stoi(argv[arg+1]);break;
        
      case 'F': simulation_params::fitness_factor=std::stod(argv[arg+1]);break;
      case 'J': simulation_params::fitness_jump=std::stod(argv[arg+1]);break;
        
      case 'O': simulation_params::fitness_period=std::stod(argv[arg+1]);break;
      case 'G': simulation_params::fitness_rise=std::stod(argv[arg+1]);break;
        
      default: std::cout<<"Unknown Parameter Flag: "<<argv[arg][1]<<std::endl;
      }
    }
    simulation_params::samming_threshold=static_cast<uint8_t>(model_params::interface_size*(1-simulation_params::binding_threshold));
    binding_probabilities=GenBindingProbsLUP();
  }
}

BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size) {
  const uint8_t total_edges=std::accumulate(edge_map.begin(),edge_map.end(),0,[](uint8_t size,const auto & p1) {return size+p1.second.size();});
  BGenotype graph(graph_size);
  
  std::uniform_int_distribution<uint8_t> delta_ser(0,simulation_params::samming_threshold),delta_ser_sym(0,simulation_params::samming_threshold/2);
  std::vector<uint8_t> bits(model_params::interface_size),bits_sym(model_params::interface_size/2);
  std::iota(bits.begin(),bits.end(),0);
  std::iota(bits_sym.begin(),bits_sym.end(),0);

  do {
    std::generate(graph.begin(),graph.end(),interface_filler); 
    for(auto edge : edge_map) {
      graph[edge.first]=interface_filler();
      for(uint8_t connector : edge.second) {
        graph[connector]=interface_model::ReverseBits(~graph[edge.first]);
        if(edge.first==connector)
          std::shuffle(bits_sym.begin(),bits_sym.end(),RNG_Engine);
        else
          std::shuffle(bits.begin(),bits.end(),RNG_Engine);
        const uint8_t delta_s = (edge.first==connector) ? delta_ser_sym(RNG_Engine) : delta_ser(RNG_Engine);
        for(uint8_t b=0; b<delta_s;++b)
          graph[connector] ^=(interface_type(1)<<bits[b]);
      }
    }
  }while(CountActiveInterfaces(graph)!=total_edges);
  return graph;
}


void EnsureNeutralDisconnections(BGenotype& genotype,GenotypeMutator& mutator) {
  BGenotype temp(genotype);
  uint8_t edges = CountActiveInterfaces(temp);
  if(edges==0)
    return; //no edges, so no need to swap anything
  StripNoncodingGenotype(temp);
  if(temp.size()==genotype.size())
    return; //not disconnected, no need to swap
  uint8_t new_edges=CountActiveInterfaces(temp);
  if(new_edges==0)//disjointed with internal edge on 2nd tile
    std::swap_ranges(genotype.begin(),genotype.begin()+4,genotype.begin()+4);
  else {
    if(new_edges!=edges) { //disjointed with internal edges on both
      do {
        std::generate(genotype.begin()+4,genotype.end(),interface_filler);
      }while(CountActiveInterfaces(genotype)!=new_edges);
      //established disjointed tile with internal tile on first, neutral 2nd tile
      do {
        temp.assign(genotype.begin()+4,genotype.end());
        mutator(temp);
      }while(CountActiveInterfaces(temp)!=0); //don't allow new internal edges on 2nd tile, but can allow external edge
      std::swap_ranges(genotype.begin()+4, genotype.end(), temp.begin());
    }
  }
}
