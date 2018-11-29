DATA ?= $(HOME)/data.hdf5
MODEL ?= $(HOME)/model/
OUT_FOLDER ?= $(MODEL)
HOSTPORT ?= 5000
GPU ?= 0
DOCKER = nvidia-docker
IMAGE = prosit
DOCKERFILE = Dockerfile


build:
	$(DOCKER) build -qf $(DOCKERFILE) -t $(IMAGE) .


predict: build
	$(DOCKER) run -it \
	    -v "$(DATA)":/root/data.hdf5 \
	    -v "$(MODEL)":/root/model/ \
	    -v "$(OUT_FOLDER)":/root/prediction/ \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    $(IMAGE) python3 -m prosit.prediction


train: build
	$(DOCKER) run -it \
	    -v "$(DATA)":/root/data.hdf5 \
	    -v "$(MODEL)":/root/model/ \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    $(IMAGE) python3 -m prosit.training


server: build
	$(DOCKER) run -it \
	    -v "$(MODEL)":/root/model/ \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    -p $(HOSTPORT):5000 \
	    $(IMAGE) python3 -m prosit.server

jump: build
	$(DOCKER) run -it \
	    -v "$(MODEL)":/root/model/ \
	    -v "$(DATA)":/root/data.hdf5 \
	    -e CUDA_VISIBLE_DEVICES=$(GPU) \
	    $(IMAGE) bash
