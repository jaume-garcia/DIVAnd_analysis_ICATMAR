# Makefile for the HF Radar program in Fortran 90 [cite: 1]

# Compiler [cite: 1]
FC = gfortran

# Compiler Flags [cite: 1]
FFLAGS = -O4

# Automatic NetCDF Detection [cite: 1]
NETCDF_EXISTS := $(shell command -v nf-config 2> /dev/null)

ifdef NETCDF_EXISTS
    # If nf-config is available, use it to retrieve flags [cite: 1]
    NETCDF_INC = $(shell nf-config --fflags)
    NETCDF_LIB = $(shell nf-config --flibs)
else
    # Default paths (adjust according to your system environment) [cite: 1]
    NETCDF_INC = -I/usr/include
    NETCDF_LIB = -L/usr/lib/x86_64-linux-gnu -lnetcdff -lnetcdf
    
    # For macOS with Homebrew, uncomment the following: [cite: 1]
    # NETCDF_INC = -I/usr/local/include [cite: 2]
    # NETCDF_LIB = -L/usr/local/lib -lnetcdff -lnetcdf
    
    # For systems using environment modules (e.g., module load netcdf), uncomment: [cite: 1]
    # NETCDF_INC = -I$(NETCDF_DIR)/include
    # NETCDF_LIB = -L$(NETCDF_DIR)/lib -lnetcdff -lnetcdf
endif

# Object files [cite: 1]
OBJS = hfradar_module.o hfradar_main.o

# Executable name [cite: 1]
EXEC = hfradar_program

# Default target [cite: 1]
all: $(EXEC)

# Link object files to create the executable [cite: 1]
$(EXEC): $(OBJS)
	@echo "Linking executable..."
	$(FC) $(FFLAGS) -o $(EXEC) $(OBJS) $(NETCDF_LIB)
	@echo "Compilation successful: ./$(EXEC)"

# Compile module [cite: 1]
hfradar_module.o: hfradar_module.f90
	@echo "Compiling module: HFRADAR_MODULE..."
	$(FC) $(FFLAGS) -c hfradar_module.f90 $(NETCDF_INC)

# Compile main program (module dependency included) [cite: 1]
hfradar_main.o: hfradar_main.f90 hfradar_module.o
	@echo "Compiling main program..."
	$(FC) $(FFLAGS) -c hfradar_main.f90

# Remove compiled files [cite: 2, 3]
clean:
	@echo "Cleaning compiled files..."
	rm -f *.o *.mod $(EXEC)

# Clean and recompile [cite: 1]
rebuild: clean all

# Display NetCDF configuration details [cite: 1]
netcdf-info:
	@echo "=== NetCDF Information ==="
ifdef NETCDF_EXISTS
	@echo "nf-config found at: $(NETCDF_EXISTS)"
	@echo "Include flags: $(NETCDF_INC)"
	@echo "Library flags: $(NETCDF_LIB)"
else
	@echo "nf-config NOT found - using default paths"
	@echo "Include flags: $(NETCDF_INC)"
	@echo "Library flags: $(NETCDF_LIB)"
	@echo ""
	@echo "If compilation fails, please adjust NETCDF_INC and NETCDF_LIB in the Makefile"
endif

# Verify syntax without compiling [cite: 1]
check:
	$(FC) -fsyntax-only hfradar_module.f90
	$(FC) -fsyntax-only hfradar_main.f90

# Help target [cite: 1]
help:
	@echo "Available targets:"
	@echo "  make           - Compile the program"
	@echo "  make clean     - Remove compiled files"
	@echo "  make rebuild   - Clean and recompile"
	@echo "  make netcdf-info - Display NetCDF configuration"
	@echo "  make check     - Perform syntax check"
	@echo "  make help      - Show this help menu"

# .PHONY targets [cite: 1]
.PHONY: all clean rebuild netcdf-info check help
