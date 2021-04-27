# Pytorch Util
This is a personal util library for PyTorch, to help facilitate training and testing boiler plate code.
I do not guarantee that this is bug free, or follows every best practice. It is meant as a util library
which I use for majority of my projects.

## Features
- Training, validation, and testing boiler plate code
- Model code is placed into simple isolated methods
- Easy experiment config support (gin-config)
- Callbacks implemented: checkpoint saves/loads, early stoppage, gradient clipping, logging, tensorboard tracking
- Simple model creation from a config object (see examples)

## TODO
- Add additional tensorboard metric support
- Add additional metric trackers (comet, etc.)
- Intuitive RL support (using environment + sampling from replay)
- More complex model creation
