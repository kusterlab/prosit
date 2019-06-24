
# Prosit

Prosit is a deep neural network to predict iRT values and MS2 spectra for given peptide sequences. 
You can use it at [proteomicsdb.org/prosit/](http://www.proteomicsdb.org/prosit/) without installation.

[![CLA assistant](https://cla-assistant.io/readme/badge/kusterlab/prosit)](https://cla-assistant.io/kusterlab/prosit)

## Hardware

Prosit requires

- a [GPU with CUDA support](https://developer.nvidia.com/cuda-gpus)


## Installation

Prosit requires

- [Docker 17.05.0-ce](https://docs.docker.com/install/)
- [nvidia-docker 2.0.3](https://github.com/NVIDIA/nvidia-docker) with CUDA 8.0 and CUDNN 6 or later installed
- [make 4.1](https://www.gnu.org/software/make/)

Prosit was tested on Ubuntu 16.04, CUDA 8.0, CUDNN 6 with Nvidia Tesla K40c and Titan Xp graphic cards with the dependencies above.

The time installation takes is dependent on your download speed (Prosit downloads a 3GB docker container). In our tests installation time is ~5 minutes.

## Model

Prosit assumes your model to be in a directory that includes:

- model.yml - a saved keras model
- config.yml - a model specifying names of inputs and outputs of the model
- weights file(s) - that follow the template `weights_{epoch}_{loss}.hdf5`

You can download a pre-trained model for HCD fragmentation prediction on https://figshare.com/projects/Prosit/35582.

## Usage

The following command will load your model from `/path/to/model/`.
In the example GPU device 0 is used for computation. The default PORT is 5000.

    make server MODEL=/path/to/model/

## Example

Please find an example input file at `example/peptidelist.csv`. After starting the server you can run:

    curl -F "peptides=@examples/peptidelist.csv" http://127.0.0.1:5000/predict/

    The example takes about 4s to run. An expected output file can be found at `examples/output_msms.txt`.

## Using Prosit on your data

You can adjust the example above to your own needs. Send any list of (Peptide, Precursor charge, Collision energy) in the format of `/example/peptidelist.csv` to a running instance of the Prosit server.

Please note: Sequences with amino acid U, O, or X are not supported. Modifications except "M(ox)" are not supported. Each C is treated as Cysteine with carbamidomethylation (fixed modification in MaxQuant).

## Pseudo-code

1. Load the model given as MODEL environment variable
2. Start a server and wait for inputs
3. On incomming request
    * transform peptide list to model input format (numpy arrays)
    * predict fragment intensity with loaded model for given peptides
    * transform prediction to msms.txt output format and return response
