MAKEFLAGS+="-j $(nproc)"

#Compiler and Linker
#CXX         := g++

#The Target Binary Program
PE_TARGET   := ProteinEvolution


#The Directories, Source, Includes, Objects, Binary and Resources
SRCDIR      := src
INCDIR      := includes
LIBDIR      := polyomino_core
BUILDDIR    := build
TARGETDIR   := bin
PROFDIR	    := profiling
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

#VPATH=src:polyomino/src

#Flags, Libraries and Includes
CXXFLAGS    := -std=c++17 -Wall -Wextra -pedantic -pipe -O3 -fopenmp

INC         := -I$(INCDIR) -I$(LIBDIR)/$(INCDIR)
INCDEP      := -I$(INCDIR) -I$(LIBDIR)/$(INCDIR)

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
PE_SOURCES := $(shell find $(SRCDIR) -type f -name interface_*.$(SRCEXT))
PE_OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(PE_SOURCES:.$(SRCEXT)=.$(OBJEXT)))


#Default Make
all: Pe 

#Clean only Objects
clean:
	@$(RM) -rf $(BUILDDIR)

#Pull in dependency info for *existing* .o files

-include $(PE_OBJECTS:.$(OBJEXT)=.$(DEPEXT))
-include $(CORE_OBJECTS:.$(OBJEXT)=.$(DEPEXT))

Pe: $(PE_OBJECTS) $(CORE_OBJECTS) 
	@mkdir -p $(TARGETDIR)
	$(CXX) $(CXXFLAGS) -Wl,--gc-sections -o $(TARGETDIR)/$(PE_TARGET) $^


#Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' < $(BUILDDIR)/$*.$(DEPEXT).tmp > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >> $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp


#Non-File Targets
.PHONY: all clean Pe check-and-reinit-submodules

check-and-reinit-submodules:
	@if git submodule status | egrep -q '^[-]|^[+]' ; then \
		echo "INFO: Need to reinitialize git submodules"; \
		git submodule update --init; \
	fi
