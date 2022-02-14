# catch
This repository contains code that will reproduce the results of the paper titled, ["Recurrent Models of Visual Attention"](https://proceedings.neurips.cc/paper/2014/file/09c6c3783b4a70054da74f2538ed47c6-Paper.pdf).

Here is the [assocated blog article](https://www.storminthecastle.com/post/catch) describing this project in more detail.

There are two different tracks one is to use a recurrent model of attention to learn to play a very simplified game of "Catch" and the other track is to learn to classify mnist digits using the same recurrent attention approach.

## Setup Environment
I recommend that you use a python version manager.  I use [conda](https://docs.conda.io/en/latest/).

Using Python 3.7 and Conda.

'''
conda create -n catch python=3.7
conda activate catch
'''

Use pip to install the required python modules.

'''
pip install numpy matplotlib opencv-python tensorboard torch torchvision Pillow
'''

## Generate Derived Datasets for MNIST
Run the following command to download, generate and preprocess 3 different MNIST derived datasets.

'''
python generate_datasets.py
'''
The original downloaded dataset will be contained in the 'mnist' directory.
The three preprocessed MNIST variants that will be used by the code will be contained in the 'prepped_mnist' directory.  Please refer to the referenced blog article above for more details on the different datasets.

## Demo Catch
I've provided a pretained model contained in the file chkpt/best_catch.pth.  You can run the catch game and render a visualization of the pretrained model using the following command.

'''
python ram_catch.py --demo
'''

## Train Catch
*Note: A CUDA-capable GPU is probably required.*
To train the model from scratch just run with no additional command line arguments.  A checkpoint for each epoch will be saved in the 'chkpt' directory.

'''
python ram_catch.py
'''

If you'd like to watch the game play while the model trains you can add the --render switch.  There will be some impact to training time.  Note that the game is trained in batches (parallel).  I only render the first game in each batch.

'''
python ram_catch.py --render
'''

## Demo MNIST
I've provided pretrained models for each of the three MNIST datasets (centered, translated, cluttered).  You can run a demo/visualization for each of the pretrained models with the following command.

'''
python ram_mnist --demo --dataset centered
'''
The specified dataset can be one of 'centered', 'translated' or 'cluttered'.

## Train MNIST
*Note: A CUDA-capable GPU is probably required.*
To train the model from scratch just run without the --demo switch as follows:

'''
python ram_mnist --dataset centered
'''
The specified dataset can be one of 'centered', 'translated' or 'cluttered'.


To find out more, please visit my blog article, [Catch](https://www.storminthecastle.com/post/catch).
