# Spike-GAN

Tensorflow implementation of Spike-GAN, which allows generating realistic patterns of neural activity whose statistics approximate a given traing dataset ([Molano-Mazon et al. 2018 ICRL2018](https://openreview.net/forum?id=r1VVsebAZ)). 

![alt tag](figs/architecture.png)

### Prerequisites

* Python 3.5+
* Tensorflow 0.12.1.
* Numpy.
* SciPy.
* Matplotlib.

All necessary modules can be installed via [Anaconda](https://anaconda.org/). 

### Installing

Just download the repository to your computer and add the Spike-GAN folder to your path.


### Example

The example below will train Spike-GAN with the semi-convolutional architecture on a simulated dataset containing the activity of two correlated neurons whose firing rate follows a uniform distribution across 12 ms. See main_conv.py for more options on the type of simulated activity (refractory period, firing rate...).

```
python3.5 main_conv.py --is_train --architecture='conv' --dataset='uniform' --num_bins=12 --num_neurons=2 
```

### Authors
* [Manuel Molano](https://github.com/manuelmolano).
* [Arno Onken](https://github.com/asnelt).
* [Eugenio Piasini](https://github.com/epiasini).
