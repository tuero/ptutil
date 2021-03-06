# Pytorch Util
This is a personal util library for PyTorch, to help facilitate training and testing boiler plate code,
allowing for faster prototyping for my projects/thesis work.
A lot of the inspiration is from libraries such as [PyTorch Lightning](https://www.pytorchlightning.ai/);
however, I prefer to have more control over the feature set and how it interacts with non-API code.

I do not guarantee that this is bug free, or follows every best practice. 
It is meant as a util library which I use for majority of my projects.

## Install
```shell
git clone https://github.com/tuero/ptutil.git
cd ptutil
pip install -e .
```

## Examples
See the included [examples](./example) for details on how to use. 

## Features
- Training, validation, and testing boiler plate code
- Model code is placed into simple isolated methods
- Easy experiment config support (gin-config)
- Callbacks implemented: checkpoint saves/loads, early stoppage, gradient clipping, logging, tensorboard tracking
- Simple model creation from a config object (see examples)
- Support for RL training

## TODO
- Add additional tensorboard metric support
- Add additional metric trackers (comet, etc.)
