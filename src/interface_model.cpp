#include "interface_model.hpp"
#include <functional>

thread_local std::mt19937 RNG_Engine(std::random_device{}());

namespace {
  std::array<uint8_t,interface_size> _Seqer() {
    std::array<uint8_t,interface_size> res;
    std::iota(res.begin(),res.end(),0);
    return res;
  }
  
  thread_local std::array<uint8_t,interface_size> bits=_Seqer();
  thread_local auto interface_filler = std::bind(std::uniform_int_distribution<interface_type>(), std::ref(RNG_Engine));

  static auto initialize_PAR() {
    std::ifstream fin("configs.ini");
    char param_name;
    std::string param_value;
    while ( fin >> param_name >> param_value ) {
      switch(param_name) {
      case 'M': simulation_params::mu_prob=std::stod(param_value);break;
      case 'Y': simulation_params::binding_threshold=std::stod(param_value);break;
      case 'T': simulation_params::temperature=std::stod(param_value);break;
      }
    }
    simulation_params::samming_threshold=static_cast<uint8_t>(interface_size*(1-simulation_params::binding_threshold));
    std::array<double,interface_size+1> bps{};
    for(size_t i=0;i<=simulation_params::samming_threshold;++i)
      bps[i]=std::pow(1-double(i)/interface_size,simulation_params::temperature);
   
    return std::make_pair(bps,std::binomial_distribution<uint8_t>(interface_size,simulation_params::mu_prob/(interface_size*4*simulation_params::n_tiles)));   
  }
  
}
auto [binding_probabilities, q_dist] = initialize_PAR(); // xx, yy are double


void PrintBindingStrengths() {
  for(auto b : binding_probabilities)
    std::cout<<b<<std::endl;
}      



//std::normal_distribution<double> normal_dist(0,1);


void InterfaceAssembly::Mutation(BGenotype& genotype) {
  for(interface_type& base : genotype) {
    std::shuffle(bits.begin(), bits.end(), RNG_Engine);
    const uint8_t num_mutations=q_dist(RNG_Engine);
    for(uint8_t nth=0;nth<num_mutations;++nth) 
      base ^= (interface_type(1) << bits[nth]);
  }
}

double InterfaceAssembly::InteractionMatrix(const interface_type face_1,const interface_type face_2) {
  return binding_probabilities[interface_model::SammingDistance(face_1,face_2)];
}


namespace simulation_params
{
  uint16_t dissociation_time=0;
  uint8_t n_tiles=2,model_type=0,samming_threshold=10;
  bool fixed_seed=false;
  double temperature=0,binding_threshold=1,mu_prob=1;
}
namespace model_params
{
  uint16_t phenotype_builds=10;
  double UND_threshold=0.2;
  bool FIXED_TABLE=false;
}


namespace interface_model
{
  interface_type ReverseBits(interface_type v) {
    interface_type s(interface_size), mask= ~0;
    while ((s >>= 1) > 0) {
      mask ^= (mask << s);
      v = ((v >> s) & mask) | ((v << s) & ~mask);
    }
    return v;
  }

  uint8_t SammingDistance(interface_type face1,interface_type face2) {
    return __builtin_popcountll(face1 ^ ReverseBits(~face2));
  }

  double PolyominoAssemblyOutcome(BGenotype& binary_genome,FitnessPhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions) {
    InterfaceAssembly::StripNoncodingGenotype(binary_genome);
    size_t UB_size=static_cast<size_t>(.75*binary_genome.size()*binary_genome.size());
    const std::vector<std::pair<interaction_pair,double> > edges = InterfaceAssembly::GetEdgePairs(binary_genome);

    std::vector<int8_t> assembly_information;
    Phenotype phen;
    std::vector<Phenotype_ID> Phenotype_IDs;
    Phenotype_IDs.reserve(model_params::phenotype_builds);
    std::set<interaction_pair > interacting_indices;
    std::map<Phenotype_ID, std::set<interaction_pair> > phenotype_interactions;
    
    for(uint16_t nth=0;nth<model_params::phenotype_builds;++nth) {
      assembly_information=InterfaceAssembly::AssemblePolyomino(edges,simulation_params::fixed_seed ? 1 : 1+4*std::uniform_int_distribution<uint8_t>(0,binary_genome.size()/4-1)(RNG_Engine),UB_size,interacting_indices);
      if(assembly_information.size()>0) {
        phen=GetPhenotypeFromGrid(assembly_information);
        Phenotype_IDs.emplace_back(pt->GetPhenotypeID(phen));
        phenotype_interactions[Phenotype_IDs.back()].insert(interacting_indices.begin(),interacting_indices.end());
      }
      else
        Phenotype_IDs.emplace_back(0,0);
      interacting_indices.clear();
    }

    pt->RelabelPhenotypes(Phenotype_IDs,phenotype_interactions);
    std::map<Phenotype_ID,uint16_t> ID_counter=pt->PhenotypeFrequencies(Phenotype_IDs);
    if(!ID_counter.empty())
      pid=std::max_element(ID_counter.begin(),ID_counter.end(),[] (const auto & p1, const auto & p2) {return p1.second < p2.second;})->first;
    else
      pid=NULL_pid;

    pid_interactions=phenotype_interactions[pid];    
    if(simulation_params::model_type==1)
      return pt->SingleFitness(pid,ID_counter[pid]);
    if(simulation_params::model_type==2) {
      for(auto kv : ID_counter) {
        if(kv.first!=NULL_pid && kv.first!=pid)
          pid_interactions.merge(phenotype_interactions[kv.first]);
      }

    }
    
    return pt->GenotypeFitness(ID_counter);
  }


}//end interface_model namespace

void RandomiseGenotype(BGenotype& genotype) {
  do {
    std::generate(genotype.begin(),genotype.end(),interface_filler);
  }while(InterfaceAssembly::CountActiveInterfaces(genotype)!=0);
}

BGenotype GenerateTargetGraph(std::map<uint8_t,std::vector<uint8_t>> edge_map,uint8_t graph_size) {
  const uint8_t total_edges=std::accumulate(edge_map.begin(),edge_map.end(),0,[](uint8_t size,const auto & p1) {return size+p1.second.size();});
  BGenotype graph(graph_size);
  
  std::uniform_int_distribution<uint8_t> delta_ser(0,simulation_params::samming_threshold),delta_ser_sym(0,simulation_params::samming_threshold/2);
  std::vector<uint8_t> bits(interface_size),bits_sym(interface_size/2);
  std::iota(bits.begin(),bits.end(),0);
  std::iota(bits_sym.begin(),bits_sym.end(),0);

  do {
    RandomiseGenotype(graph); 
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
  }while(InterfaceAssembly::CountActiveInterfaces(graph)!=total_edges);
  return graph;
}

void EnsureNeutralDisconnections(BGenotype& genotype) {
  BGenotype temp_genotype(genotype);
  uint8_t edges = InterfaceAssembly::CountActiveInterfaces(temp_genotype);

  if(edges==0)
    return; //no edges, so no need to swap anything
  InterfaceAssembly::StripNoncodingGenotype(temp_genotype);
  if(temp_genotype.size()==genotype.size())
    return; //not disconnected, no need to swap
  uint8_t new_edges=InterfaceAssembly::CountActiveInterfaces(temp_genotype);
  if(new_edges==0)//disjointed with internal edge on 2nd tile
    std::swap_ranges(genotype.begin(),genotype.begin()+4,genotype.begin()+4);
  else {
    if(new_edges!=edges) { //disjointed with internal edges on both
      do {
        std::generate(genotype.begin()+4,genotype.end(),interface_filler);
      }while(InterfaceAssembly::CountActiveInterfaces(genotype)!=new_edges);
      //established disjointed tile with internal tile on first, neutral 2nd tile
      do {
        temp_genotype.assign(genotype.begin()+4,genotype.end());
	InterfaceAssembly::Mutation(temp_genotype);
      }while(InterfaceAssembly::CountActiveInterfaces(temp_genotype)!=0); //don't allow new internal edges on 2nd tile, but can allow external edge
      std::swap_ranges(genotype.begin()+4, genotype.end(), temp_genotype.begin());
    }
  }
}

