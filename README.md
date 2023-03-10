# adaptative_vqa_optimizers
Implementation of adaptative optimizers for variational quantum algorithms (Qhack 2023 submission)
We reimplement an optimizer that has been designed for quantum machine learning applications from the paper [Resource frugal optimizer for quantum machine learning
](https://arxiv.org/abs/2211.04965). The optimizer is mainly defined in [refoqus.py](./refoqus.py).


We provide 3 application examples as notebooks:
* Variational Quantum State Eigensolver [vqse_example.ipynb](https://github.com/chMoussa/adaptative_vqa_optimizers/blob/main/vqse_example.ipynb)
* Quantum Autoencoder [quantoencoder_example.ipynb](https://github.com/chMoussa/adaptative_vqa_optimizers/blob/main/quantoencoder_example.ipynb)
* Variational Quantum Error Correction [vQEC.ipynb](https://github.com/chMoussa/adaptative_vqa_optimizers/blob/main/vQEC.ipynb)

A last notebook ([vqse_example-lightning-gpu-vs-cpu.ipynb](./vqse_example-lightning-gpu-vs-cpu.ipynb)) illustrates how using a NVidia GPU accelerator to simulate the quantum circuits improves the runtime of our VQSE application by nearly 40%.

Slides in a PDF format explaining our submission can be found in [the presentation_Qhack_2023.pdf file](./presentation_Qhack_2023.pdf).

## Installation
You can install the required dependencies with
```sh
python -m pip install -r requirements.txt
```
If you have any issue installing cuQuantum, please follow [https://docs.nvidia.com/cuda/cuquantum/getting_started.html#install-cuquantum-python-from-conda-forge](https://docs.nvidia.com/cuda/cuquantum/getting_started.html#install-cuquantum-python-from-conda-forge).

## About us

The `my_favourite_team` team is composed of 2 members:
- [Charles Moussa](https://www.linkedin.com/in/moussacharles/).
- [Adrien Suau](https://adrien.suau.me).
