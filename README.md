## Binary polyomino self-assembly
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7ffdd17eae624212ac2686d48687d343)](https://app.codacy.com/app/ASLeonard/polyomino_interfaces?utm_source=github.com&utm_medium=referral&utm_content=ASLeonard/polyomino_interfaces&utm_campaign=Badge_Grade_Dashboard)
![License Badge](https://img.shields.io/github/license/ASLeonard/polyomino_interfaces.svg?style=flat)

Polyomimo lattice self-assembly model with binary string interfaces.

### Core modules
Majority of polyomino core features are found in the `polyomino_core' submodule.
A direct link to the current core development is [here](https://github.com/ASLeonard/polyomino_core).

### Installation
Building the simulator requires a c++ compiler that has general support for c++17. It has been tested with g++7 and greater, and clang++6 and greater. The compiling itself is taken care of automatically through the makefile. The recommended steps are as follows
```
git clone --recurse-submodules https://github.com/ASLeonard/polyomino_interfaces
cd polyomino_interfaces
make
```
Errors at this stage probably indicate the compiler (or the CXX environment variable) is not modern enough.

#### Python requirements
The analysis and plotting has been tested with the following versions, although many older/newer versions are likely to work. These are common packages, but not always installed by default.
+ python (3.65)
+ SciPy (1.2.1)
+ Numpy (1.16.2)
+ Matplotlib (3.03)

#### Directory layout
The main folders of interest are bin/ and scripts/, although a more curious user can modify the c++ in the other folders.

+ bin/
  + exectuable
+ build/ (internal compiling only)
+ scripts/
  + utility methods
  + analysis module
  + plotting module
+ src/
  + c++ source code
+ includes/
  + c++ headers
+ polyomino_core/ (submodule)
  + includes/
    + c++ headers
  + scripts/
    + generic polyomino plotting
  
### Simulations
The code is split into two main parts. The polyomino evolutions are written in c++, while the analysis is in python3. The c++ code, in the executable **ProteinEvolution**, can be run directly, with parameters mostly detailed by calling the `ProteinEvolution -H` feature.

However, the easiest way is to use the main method in the _interface\_analysis_ file, which calls `runEvolutionSequence`. The default parameters are sufficient to generate some interesting data, and can be modified if desired.

The files are generated in the calling directory unless the default parameter for file_path is changed, and can be large (>100s of mB) for larger choices of population size and longer simulations.

### Plotting
Several visualising methods have been included to interpret the generated data, in the _interface\_plotting_ file. They mostly involve loading simulation results, called through e.g. `loadPickle(params...)`, and passing those to the plotting functions. Each method has comments explaining how to use in more detail.


