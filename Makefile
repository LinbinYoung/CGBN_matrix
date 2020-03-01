.PHONY: pick clean xmp-run xmp_tester kepler maxwell pascal volta 

ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo

clean:
	rm -f C_API

xmp-run: C_API
	@./C_API

xmp_tester: 
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo
	@exit 0


kepler:
	nvcc $(INC) $(LIB) -I../include -arch=sm_35 C_API.cu -o C_API -lgmp

maxwell:
	nvcc $(INC) $(LIB) -I../include -arch=sm_50 C_API.cu -o C_API -lgmp

pascal:
	nvcc $(INC) $(LIB) -I../include -arch=sm_60 C_API.cu -o C_API -lgmp

volta:
	nvcc $(INC) $(LIB) -I../include -arch=sm_70 C_API.cu -o C_API -lgmp

