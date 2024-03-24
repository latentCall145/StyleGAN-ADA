# StyleGAN-ADA
My implementation of NVIDIA's StyleGAN2-ADA (paper: https://arxiv.org/pdf/2006.06676.pdf).

![Style mixing](/figures/styleMix.png)

Features:
 - Mapping network (from StyleGAN)
 - Style-based generator with noise inputs (from StyleGAN)
 - Equalized learning rate (from StyleGAN)
 - Modulated/Demodulated Convolutions (from StyleGAN2)
 - (Lazy) R1, Path Length Regularization (from StyleGAN2)
 - Skip-connection generator and residual discriminator (from StyleGAN2)
 - (good enough) adaptive discriminator augmentation (from StyleGAN2-ADA)

While implementing StyleGAN2, I noticed that style mixing didn't work since mapping activations were quite low. This was because the variance of the weight initializations were too small for Leaky ReLU, so I slightly modified the weight initializations which fixed the non-existent style mixing.

I trained a model on the Metfaces dataset provided by NVIDIA, although my results aren't quite as good as NVIDIA's results. I trained my model on Paperspace P5000 instances (GPU VRAM - 16GB, > 16GB RAM used when training), which unfortunately don't exist for free anymore. I don't remember how long I trained my model, but you should start to see signs of progress pretty quickly if you're training the model on your own custom dataset. If the program doesn't look like it's working after about 10 minutes of training, it's likely you won't see any progress at any amount of training. I learned this the hard way.

To install the required packages, type the following in a terminal/command prompt:

Windows:

    pip install -r requirements.txt

Mac/Linux:

    pip3 install -r requirements.txt
