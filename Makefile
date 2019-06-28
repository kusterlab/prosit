DATA ?= $(HOME)/data.hdf5
MODEL_SPECTRA ?= $(HOME)/model_sectra/
MODEL_IRT ?= $(HOME)/model_irt/
OUT_FOLDER ?= $(MODEL)
HOSTPORT ?= 5000
GPU ?= 0
DOCKER = nvidia-docker
IMAGE = prosit
DOCKERFILE = Dockerfile


build:
	$(DOCKER) build -qf $(DOCKERFILE) -t $(IMAGE) .


server: build
	$(DOCKER) run -it \
	    -v "$(MODEL_SPECTRA)":/root/model_spectra/ \
	    -v "$(MODEL_IRT)":/root/model_irt/ \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    -p $(HOSTPORT):5000 \
	    $(IMAGE) python3 -m prosit.server

jump: build
	$(DOCKER) run -it \
	    -v "$(MODEL_SPECTRA)":/root/model_spectra/ \
	    -v "$(MODEL_IRT)":/root/model_irt/ \
	    -v "$(DATA)":/root/data.hdf5 \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    $(IMAGE) bash
