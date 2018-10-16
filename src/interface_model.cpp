#include "interface_model.hpp"

thread_local std::mt19937 RNG_Engine(std::random_device{}());

std::normal_distribution<double> normal_dist(0,1);
std::array<double,model_params::interface_size+1> binding_probabilities;

namespace simulation_params
{
  uint16_t dissociation_time=0;
  uint8_t n_tiles=2,model_type=0,samming_threshold=10;
  bool fixed_seed=true;
  double temperature=0,binding_threshold=1,fitness_factor=1;
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
    interface_type s(model_params::interface_size), mask= ~0;
    while ((s >>= 1) > 0) {
      mask ^= (mask << s);
      v = ((v >> s) & mask) | ((v << s) & ~mask);
    }
    return v;
  }
  /*
  inline uint8_t ArbitraryPopcount(interface_type face) {
    uint8_t c;
    for(c = 0; face; c++)
      face &= face - 1;
    return c;
  }
  */
  uint8_t SammingDistance(interface_type face1,interface_type face2) {
    return __builtin_popcountll(face1 ^ ReverseBits(~face2));
    //return ArbitraryPopcount(face1 ^ ReverseBits(~face2));
  }

  double PolyominoAssemblyOutcome(BGenotype& binary_genome,InterfacePhenotypeTable* pt,Phenotype_ID& pid,std::set<interaction_pair>& pid_interactions) {
    StripNoncodingGenotype(binary_genome);
    size_t UB_size=static_cast<size_t>(.75*binary_genome.size()*binary_genome.size());
    const std::vector<std::pair<interaction_pair,double> > edges = GetEdgePairs(binary_genome);

    std::vector<int8_t> assembly_information;
    Phenotype phen;
    std::vector<Phenotype_ID> Phenotype_IDs;
    Phenotype_IDs.reserve(model_params::phenotype_builds);
    std::set<interaction_pair > interacting_indices;
    std::map<Phenotype_ID, std::set<interaction_pair> > phenotype_interactions;
    
    for(uint16_t nth=0;nth<model_params::phenotype_builds;++nth) {
      assembly_information=AssemblePolyomino(edges,simulation_params::fixed_seed ? 1 : 1+4*std::uniform_int_distribution<uint8_t>(0,binary_genome.size()/4-1)(RNG_Engine),UB_size,interacting_indices);
      if(assembly_information.size()>0) {
        phen=SpatialGrid(assembly_information);
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

  void InterfacePhenotypeTable::RelabelPhenotypes(std::vector<Phenotype_ID >& pids,std::map<Phenotype_ID, std::set<interaction_pair> >& p_ints)  {
    for(auto x_iter=new_phenotype_xfer.begin();x_iter!=new_phenotype_xfer.end();x_iter+=3) {
      p_ints[std::make_pair(*x_iter,*(x_iter+2))].insert(p_ints[std::make_pair(*x_iter,*(x_iter+1))].begin(),p_ints[std::make_pair(*x_iter,*(x_iter+1))].end());
      phenotype_fitnesses[*x_iter].emplace_back(std::gamma_distribution<double>(*(x_iter)*2,.5*std::pow(*x_iter,-.5))(RNG_Engine));
    }
    PhenotypeTable::RelabelPhenotypes(pids);
  }

  std::map<Phenotype_ID,uint16_t> InterfacePhenotypeTable::PhenotypeFrequencies(std::vector<Phenotype_ID >& pids) {
    std::map<Phenotype_ID, uint16_t> ID_counter;
    for(std::vector<Phenotype_ID >::const_iterator ID_iter = pids.begin(); ID_iter!=pids.end(); ++ID_iter) {
      if(ID_iter->second < known_phenotypes[ID_iter->first].size())
	++ID_counter[std::make_pair(ID_iter->first,ID_iter->second)];
      else
	++ID_counter[NULL_pid];
    }
    return ID_counter;
  }

  double InterfacePhenotypeTable::GenotypeFitness(std::map<Phenotype_ID,uint16_t> ID_counter) {
    double fitness=0;
    for(auto kv : ID_counter)
      if(kv.second>=ceil(model_params::UND_threshold*model_params::phenotype_builds))
	fitness+=phenotype_fitnesses[kv.first.first][kv.first.second] * std::pow(static_cast<double>(kv.second)/model_params::phenotype_builds,simulation_params::fitness_factor);     
    return fitness;
  }

  double InterfacePhenotypeTable::SingleFitness(Phenotype_ID pid,uint16_t commonness)  {
    return phenotype_fitnesses[pid.first][pid.second] * std::pow(static_cast<double>(commonness)/model_params::phenotype_builds,simulation_params::fitness_factor);     
  }


  
    

}//end interface_model namespace

void StripNoncodingGenotype(BGenotype& genotype) {
  std::vector<uint8_t> coding{0},noncoding(genotype.size()/4-1);
  std::iota(noncoding.begin(), noncoding.end(), 1);
  
  for(uint8_t c_in=0;c_in<coding.size();++c_in)
    for(uint8_t nc_in=0;nc_in<noncoding.size();++nc_in) {
      for(uint8_t cface=0;cface<4;++cface)
        for(uint8_t ncface=0;ncface<4;++ncface)
          if(interface_model::SammingDistance(genotype[coding[c_in]*4+cface],genotype[noncoding[nc_in]*4+ncface])<=static_cast<uint8_t>(model_params::interface_size*(1-simulation_params::binding_threshold))){
            coding.emplace_back(noncoding[nc_in]);
            noncoding.erase(noncoding.begin()+nc_in--);
            goto newtile;
          }
    newtile: ;
    }

  for(uint8_t rm=0;rm<noncoding.size();++rm)
    genotype.erase(genotype.begin()+(noncoding[rm]-rm)*4,genotype.begin()+(1+noncoding[rm]-rm)*4);
}

uint8_t CountActiveInterfaces(const BGenotype& genotype) {
  uint8_t N_interfaces=0;
  for(uint8_t b1=0;b1<genotype.size() ;++b1)
    for(uint8_t b2=b1;b2<genotype.size();++b2)
      if(interface_model::SammingDistance(genotype[b1],genotype[b2])<=static_cast<uint8_t>(model_params::interface_size*(1-simulation_params::binding_threshold)))
        ++N_interfaces;
  return N_interfaces;
}

std::vector<std::pair<interaction_pair,double> > GetEdgePairs(const BGenotype& genotype) {
  std::vector<std::pair<interaction_pair,double> > edge_pairs;
  for(uint8_t b1=0;b1<genotype.size();++b1)
    for(uint8_t b2=b1;b2<genotype.size();++b2) {
      uint8_t SD =interface_model::SammingDistance(genotype[b1],genotype[b2]);
      if(SD<=simulation_params::samming_threshold) {
        edge_pairs.emplace_back(std::minmax(b1,b2),binding_probabilities[SD]);
      }
    }
  return edge_pairs;
}


Phenotype SpatialGrid(std::vector<int8_t>& placed_tiles) {
  std::vector<int8_t> x_locs, y_locs,tile_vals;
  x_locs.reserve(placed_tiles.size()/3);y_locs.reserve(placed_tiles.size()/3);tile_vals.reserve(placed_tiles.size()/3);
  
  for(std::vector<int8_t>::iterator check_iter = placed_tiles.begin();check_iter!=placed_tiles.end();check_iter+=3) {
    x_locs.emplace_back(*check_iter);
    y_locs.emplace_back(*(check_iter+1));
    tile_vals.emplace_back(*(check_iter+2));
  }
  std::vector<int8_t>::iterator x_left,x_right,y_top,y_bottom;
  std::tie(x_left,x_right)=std::minmax_element(x_locs.begin(),x_locs.end());
  std::tie(y_bottom,y_top)=std::minmax_element(y_locs.begin(),y_locs.end());
  uint8_t dx=*x_right-*x_left+1,dy=*y_top-*y_bottom+1;
  std::vector<uint8_t> spatial_grid(dx*dy);
  
  for(uint16_t tileIndex=0;tileIndex<x_locs.size();++tileIndex) {
    uint8_t tile_detail=0;
    switch(DETERMINISM_LEVEL) {
    case 1:
      tile_detail=tile_vals[tileIndex] > 0 ? 1 : 0;
      break;
    case 2:
      tile_detail=tile_vals[tileIndex] > 0 ? (tile_vals[tileIndex]-1)/4+1 : 0;
      break;
    case 3:
      tile_detail=tile_vals[tileIndex];
    }
    spatial_grid[(*y_top-y_locs[tileIndex])*dx + (x_locs[tileIndex]-*x_left)]=tile_detail;
  }
  return Phenotype{dx,dy,spatial_grid};
}

uint8_t PhenotypeSymmetryFactor(std::vector<uint8_t>& original_shape, uint8_t dx, uint8_t dy) {
  std::vector<uint8_t> rotated_shape(original_shape);
  std::reverse(original_shape.begin(),original_shape.end());
  if(original_shape!=rotated_shape)
    return 1;
  if(dx==dy) {
    if(original_shape==rotated_shape)
      return 4;
  }
  return 2;
}

std::array<double,model_params::interface_size+1> GenBindingProbsLUP() {
  std::array<double,model_params::interface_size+1> probs;
  for(size_t i=0;i<probs.size();++i)
    probs[i]=(i<=simulation_params::samming_threshold?1:0) * std::pow(1-double(i)/model_params::interface_size,simulation_params::temperature);
  return probs;
}

