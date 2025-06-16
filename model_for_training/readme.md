# Model for training

## How to use
In this folder, you can use train_no_vai.py to call the unet model in models_no_vai.py.
Of course, this is just a demo from Emmet's networks_vitis.py
You can train it use whatever method like GAN and whatever dataset

## Why use this model
There are two reasons:
- For hardware implementation, if you are using leakyRuLu, the only available alpha value is 0.1015625, if we are using value far from this, that will lead to a vitis-unacceptable loss as I mentioned in the email before.
- The Vitis-AI framework doesn't support relu after concat, and we need to tell the framework to quantize relu function, or it will be generated as multiple sub-graphs (CPU+DPU).
- For Emmet's original model, it contain packs from pytorchnnct, to make it simpler. I splited it to a model pure pytorch.

## Flow
- Train the model in models_no_vai.py,
- Then I will load the pth with vai version model in models.py and quantize, deploy it.

## Change model structure
- For this structure, I have verified with Alveo U50
- If you change the structure of the model please let me know, I have scripts to do a fast verification to double check if hardware is happy with it :)