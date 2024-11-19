import { startConvolution } from './convolution';
import { startReLU } from './reluActivation';
import { startMaxPooling } from './maxPooling';

const CNNBlackBox = {
  startConvolution,
  startReLU,
  startMaxPooling,
};
  
window.CNNBlackBox = CNNBlackBox;
  