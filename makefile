NVCC := $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
		$(error nvcc not found.)
endif

ifneq ($(CI),true)
  ifndef GPU_COMPUTE_CAPABILITY
    GPU_COMPUTE_CAPABILITY = $(shell __nvcc_device_query)
    GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
  endif
endif

ifeq ($(GPU_COMPUTE_CAPABILITY),)
  CFLAGS = -O3 --use_fast_math
else
  CFLAGS = -O3 --use_fast_math --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

NVCCFLAGS = -lcublas -lcublasLt -std=c++17
MPI_PATHS = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib/

%: %.cu
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $< -o $@

TARGETS = gemm

all: $(TARGETS)
all_ptx:  $(TARGETS:%=%.ptx)
all_sass: $(TARGETS:%=%.sass)

gemm: gemm.cu

%.ptx: %
	cuobjdump --dump-ptx $< > $@

%.sass: %
	cuobjdump --dump-sass $< > $@

run_all: all
	@for target in $(TARGETS); do \
		echo "\n========================================"; \
		echo "Running $$target ..."; \
		echo "========================================\n"; \
		./$$target; \
	done

clean:
	rm -f $(TARGETS) *.ptx *.sass
