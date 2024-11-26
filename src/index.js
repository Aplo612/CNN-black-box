import { startConvolution } from './convolution';
import { startReLU } from './reluActivation';
import { startMaxPooling } from './maxPooling';
import { createCNNVisualizer } from './cnnVisualizer';

const CNNBlackBox = {
  startConvolution,
  startReLU,
  startMaxPooling,
  createCNNVisualizer,
};
  
window.CNNBlackBox = CNNBlackBox;
  