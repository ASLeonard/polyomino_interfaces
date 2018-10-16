#Compiler and Linker
CXX         := g++

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
CXXFLAGS    := -std=gnu++17 -Wall -Wextra -pedantic  -pipe -march=haswell -flto -flto-partition=none -no-pie -ffunction-sections -fdata-sections $(cmdflag)
ifndef DEBUG
CXXFLAGS += -O3 -fopenmp -Iincludes -Ipolyomino_core/includes
else
CXXFLAGS += -p -g -ggdb
endif

INC         := -I$(INCDIR)
INCDEP      := -I$(INCDIR)

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
CORE_SOURCES := $(shell find $(LIBDIR)/$(SRCDIR) -type f -name core_*.$(SRCEXT))
PE_SOURCES := $(shell find $(SRCDIR) -type f -name interface_*.$(SRCEXT))

PE_OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(PE_SOURCES:.$(SRCEXT)=.$(OBJEXT)))
CORE_OBJECTS     := $(patsubst $(LIBDIR)/$(SRCDIR)/%,$(BUILDDIR)/%,$(CORE_SOURCES:.$(SRCEXT)=.$(OBJEXT)))




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

$(BUILDDIR)/%.$(OBJEXT): $(LIBDIR)/$(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<
	@$(CXX) $(CXXFLAGS) $(INCDEP) -MM $(LIBDIR)/$(SRCDIR)/$*.$(SRCEXT) > $(BUILDDIR)/$*.$(DEPEXT)
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
