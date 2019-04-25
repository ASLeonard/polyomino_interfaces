# Binary polyomino self-assembly
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/7ffdd17eae624212ac2686d48687d343)](https://app.codacy.com/app/ASLeonard/polyomino_interfaces?utm_source=github.com&utm_medium=referral&utm_content=ASLeonard/polyomino_interfaces&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/ASLeonard/polyomino_interfaces.svg?branch=master)](https://travis-ci.org/ASLeonard/polyomino_interfaces)
![License Badge](https://img.shields.io/github/license/ASLeonard/polyomino_interfaces.svg?style=flat)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2650023.svg)](https://doi.org/10.5281/zenodo.2650023)

Code repository for the generalised polyomimo lattice self-assembly model with binary string interfaces.



### Core modules
The majority of polyomino core features are found in the `polyomino_core` submodule, designed to provide a single, consistent set of definitions in all models.
Assembly algorithms, phenotype classifications, and selection dynamics can be found within this repository. 
A direct link to the current core development is [here](https://github.com/ASLeonard/polyomino_core).

## Installation
Building the simulator requires a c++ compiler that has general support for c++17. It has been tested with g++7 and greater, and clang++6 and greater. The compiling itself is taken care of through the makefile. The recommended steps are as follows
```shell
git clone --recurse-submodules https://github.com/ASLeonard/polyomino_interfaces
cd polyomino_interfaces
make
```
At which point the simulation program is callable at `./bin/ProteinEvolution`.
Errors at this stage probably indicate the compiler (set through the CXX environment variable) is not modern enough.

Several compiler flags can be added depending on what is available on the user's system if desired, like g++ has support for multi-threading (-fopenmp) and link-time optimization (-flto), but are not required to be used.

#### Python requirements
The analysis and plotting is mostly handled in python3, and should be useable with any python version 3.6+.
Some common additional packages have been used, but are not always installed by default. Many older/newer versions are likely to work, but versions used during development were:
+ Python (3.6.5)
+ Numpy (1.16.2)
+ Matplotlib (3.03)
+ SciPy (1.2.1) (only necessary for scripts within `polyomino_core`, not used by default)

#### Testing
The installation, c++ program, and python can be tested after making, by calling `make test`. This will run an evolution simulation with default parameters, analyse the generated files, save the analysis, and then erase all the generated files and analysis. Note this erases generated files by pattern matching, so will erase any generated txt or pkl files within the root directory.

  
## Simulations
The code is split into two main parts. The polyomino evolutions are written in c++, while the analysis is in python3. The c++ code, in the executable **ProteinEvolution**, can be run directly, with parameters mostly detailed by calling `./bin/ProteinEvolution -H`. There is one main parameter that is hardcoded in the c++, the interface length, found in `includes/interface_model.hpp` as `using interface_type = ...`. Details on how to change it are described there.

A simple example has been provided below to automatically generate, analyse, and plot some data. The default parameters are sufficient to generate some interesting data, and can and should be modified to run over more simulations and different parameters. They can be found in `scripts/interface_analysis.py`, as a dictionary within the `runEvolutionSequence()` method.

The files are generated in the calling directory unless the default parameter for file_path is changed, and can be large (>100s of MB) for larger choices of population size and longer simulations.

#### Example usage
A very minimal example can be achieved from the root directory with 
```shell
python3 scripts/interface_analysis.py [optional parameter for number of simulations, e.g. 10]
python3 scripts/interface_plotting.py
```

### Plotting
Several visualising methods have been included to interpret the generated data, in the `interface_plotting` file. They mostly involve loading simulation results, called through e.g. `loadPickle(params...)`, and passing those to the plotting functions. Each method has comments explaining how to use in more detail.

After generating data through the first line of the minimal example, you can plot the data more selective by calling different plots and options via
```python
data=loadPickle(.6875,25,1,5)

calculatePathwaySucess(data)
#or
plotEvolution(data)
#or
plotPhaseSpace(data)
```
Where various method default parameters have been commented on in the files.

### Directory layout
The main folder of interest is scripts/, although a more curious user can modify the c++ in the other folders. In the main directory there is also the makefile, which can be modified with the previously mentioned compiler flags.

+ scripts/ (main directory for user)
  + utility methods
  + analysis module
  + plotting module
+ src/
  + c++ source code
+ includes/s
  + c++ headers
+ polyomino_core/ (submodule)
  + includes/
    + c++ headers
  + scripts/
    + generic polyomino plotting
+ bin/
  + executable
+ build/ (internal compiling only)


