# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 22:24:14 2020

@author: Jorge

First 'hello word' test
"""

import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN

img = Image.open('data/input/test_images/IMG-SP001.jpg')
lr_img = np.array(img)

# model = RDN(weights='noise-cancel')
#model = RRDN(weights='gans')
# model = RDN(weights='psnr-small')
model = RDN(weights='psnr-large')

sr_img = model.predict(lr_img)


im = Image.fromarray(sr_img)
im.show()


