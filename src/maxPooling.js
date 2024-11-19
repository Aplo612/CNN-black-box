import * as tf from '@tensorflow/tfjs';

export function startMaxPooling({ container, imageSrc, filter, channel = 'grayscale' }) {
  // Variables y constantes iniciales
  const operationWidth = 64;
  const operationHeight = 64;
  const displayScale = 4; // Factor de escala para agrandar cada píxel
  const poolingSize = 2;
  const poolingStride = 2;

  const convOutputWidth = operationWidth - 2; // Asumiendo filtro de tamaño 3x3 y padding 'valid'
  const convOutputHeight = operationHeight - 2;

  const pooledWidth = Math.floor(convOutputWidth / poolingStride);
  const pooledHeight = Math.floor(convOutputHeight / poolingStride);

  const inputDisplayWidth = convOutputWidth * displayScale;
  const inputDisplayHeight = convOutputHeight * displayScale;

  const outputDisplayWidth = pooledWidth * displayScale *2;
  const outputDisplayHeight = pooledHeight * displayScale *2;

  let maxAbsValue = 0;

  const containerElement = document.getElementById(container);

  // Crear el contenedor principal
  const mainContainer = document.createElement('div');
  mainContainer.style.display = 'flex';
  mainContainer.style.alignItems = 'center';
  mainContainer.style.justifyContent = 'center';
  mainContainer.style.gap = '20px';
  containerElement.appendChild(mainContainer);

  // Crear contenedor para entrada (texto y canvas)
  const inputWrapper = document.createElement('div');
  inputWrapper.style.display = 'flex';
  inputWrapper.style.flexDirection = 'column';
  inputWrapper.style.alignItems = 'center';

  // Crear el contenedor para el canvas de entrada
  const inputCanvasContainer = document.createElement('div');
  inputCanvasContainer.style.position = 'relative';
  inputCanvasContainer.style.display = 'inline-block';

  // Crear el elemento de texto para 'Entrada ReLU'
  const inputLabel = document.createElement('div');
  inputLabel.textContent = `Entrada (${convOutputWidth}, ${convOutputHeight})`;
  inputLabel.style.textAlign = 'center';
  inputLabel.style.marginBottom = '5px';

  // Crear contenedor para salida (texto y canvas)
  const outputWrapper = document.createElement('div');
  outputWrapper.style.display = 'flex';
  outputWrapper.style.flexDirection = 'column';
  outputWrapper.style.alignItems = 'center';

  // Crear el contenedor para el canvas de salida
  const outputCanvasContainer = document.createElement('div');
  outputCanvasContainer.style.position = 'relative';
  outputCanvasContainer.style.display = 'inline-block';

  // Crear el elemento de texto para 'Salida Max Pooling'
  const outputLabel = document.createElement('div');
  outputLabel.textContent = `Salida (${pooledWidth}, ${pooledHeight})`;
  outputLabel.style.textAlign = 'center';
  outputLabel.style.marginBottom = '5px';

  // Crear los canvas para la imagen de entrada y salida
  const inputCanvas = document.createElement('canvas');
  inputCanvas.width = convOutputWidth;
  inputCanvas.height = convOutputHeight;
  inputCanvas.style.width = `${inputDisplayWidth}px`;
  inputCanvas.style.height = `${inputDisplayHeight}px`;
  inputCanvas.style.border = '1px solid black';
  inputCanvas.style.imageRendering = 'pixelated';
  inputWrapper.appendChild(inputLabel);
  inputCanvasContainer.appendChild(inputCanvas);
  inputWrapper.appendChild(inputCanvasContainer);

  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = pooledWidth;
  outputCanvas.height = pooledHeight;
  outputCanvas.style.width = `${outputDisplayWidth}px`;
  outputCanvas.style.height = `${outputDisplayHeight}px`;
  outputCanvas.style.border = '1px solid black';
  outputCanvas.style.imageRendering = 'pixelated';
  outputWrapper.appendChild(outputLabel);
  outputCanvasContainer.appendChild(outputCanvas);
  outputWrapper.appendChild(outputCanvasContainer);

  // Desactivar el suavizado de imagen
  const inputCtx = inputCanvas.getContext('2d');
  inputCtx.imageSmoothingEnabled = false;

  const outputCtx = outputCanvas.getContext('2d');
  outputCtx.imageSmoothingEnabled = false;

  // Crear los canvas de resaltado
  const inputHighlightCanvas = document.createElement('canvas');
  inputHighlightCanvas.width = inputCanvas.width;
  inputHighlightCanvas.height = inputCanvas.height;
  inputHighlightCanvas.style.width = inputCanvas.style.width;
  inputHighlightCanvas.style.height = inputCanvas.style.height;
  inputHighlightCanvas.style.position = 'absolute';
  inputHighlightCanvas.style.left = '0';
  inputHighlightCanvas.style.top = '0';
  inputHighlightCanvas.style.pointerEvents = 'none';
  inputCanvasContainer.appendChild(inputHighlightCanvas);

  const outputHighlightCanvas = document.createElement('canvas');
  outputHighlightCanvas.width = outputCanvas.width;
  outputHighlightCanvas.height = outputCanvas.height;
  outputHighlightCanvas.style.width = outputCanvas.style.width;
  outputHighlightCanvas.style.height = outputCanvas.style.height;
  outputHighlightCanvas.style.position = 'absolute';
  outputHighlightCanvas.style.left = '0';
  outputHighlightCanvas.style.top = '0';
  outputHighlightCanvas.style.pointerEvents = 'none';
  outputCanvasContainer.appendChild(outputHighlightCanvas);

  // Agregar los wrappers al contenedor principal
  mainContainer.appendChild(inputWrapper);

  // Crear el contenedor para la visualización de Max Pooling (SVG)
  const svgMatrixContainer = document.createElement('div');
  svgMatrixContainer.style.display = 'flex';
  svgMatrixContainer.style.alignItems = 'center';
  svgMatrixContainer.style.justifyContent = 'center';
  mainContainer.appendChild(svgMatrixContainer);

  mainContainer.appendChild(outputWrapper);

  // Inicializar el SVG de Max Pooling
  let poolingSVG, poolingRects, poolingTexts, resultRect, resultText;
  initializePoolingSVG();

  // Variables para almacenar los datos de los tensores
  let reluData, pooledData;

  // Cargar la imagen de entrada
  const inputImage = new Image();
  inputImage.src = imageSrc;
  inputImage.crossOrigin = 'anonymous'; // Para evitar problemas de CORS

  inputImage.onload = async () => {
    // Dibujar la imagen redimensionada en un canvas temporal
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = operationWidth;
    tempCanvas.height = operationHeight;
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });
    tempCtx.drawImage(inputImage, 0, 0, operationWidth, operationHeight);

    // Obtener los datos de la imagen
    const imageData = tempCtx.getImageData(0, 0, operationWidth, operationHeight);

    // Realizar la convolución utilizando TensorFlow.js
    const filterSize = Math.sqrt(filter.length);

    let inputTensor;

    // Crear tensor de imagen normalizado
    if (channel === 'grayscale') {
      inputTensor = tf.browser.fromPixels(tempCanvas, 1).toFloat().div(127.5).sub(1); // Normalizar a [-1, 1]
      inputTensor = inputTensor.expandDims(0); // [batch, height, width, channels]
    } else if (channel === 'R' || channel === 'G' || channel === 'B') {
      // Extraer el canal seleccionado
      const channelIndex = { 'R': 0, 'G': 1, 'B': 2 }[channel];
      const channelData = extractChannel(imageData.data, channelIndex);
      inputTensor = tf.tensor2d(channelData, [operationHeight, operationWidth], 'float32')
        .div(127.5).sub(1).expandDims(0).expandDims(-1);
    }

    // Crear tensor del filtro
    const filterTensor = tf.tensor4d(filter, [filterSize, filterSize, 1, 1]);

    // Aplicar la convolución
    let convTensor = tf.conv2d(inputTensor, filterTensor, [1, 1], 'valid'); // Usamos 'valid' para evitar padding

    // Obtener el valor máximo absoluto para normalizar
    maxAbsValue = (await tf.max(tf.abs(convTensor)).data())[0];

    // Normalizar el tensor convolucionado al rango [-1, 1]
    convTensor = convTensor.div(maxAbsValue);

    // Aplicar ReLU al tensor convolucionado
    let reluTensor = tf.relu(convTensor);

    // Aplicar Max Pooling
    let pooledTensor = tf.maxPool(reluTensor, [poolingSize, poolingSize], [poolingStride, poolingStride], 'valid');

    // Obtener los datos de los tensores
    reluData = await reluTensor.squeeze().array();
    pooledData = await pooledTensor.squeeze().array();

    // Crear ImageData para la imagen ReLU (entrada)
    const reluImageData = new ImageData(convOutputWidth, convOutputHeight);

    for (let i = 0; i < reluData.length; i++) {
      for (let j = 0; j < reluData[0].length; j++) {
        const value = reluData[i][j]; // Valor entre 0 y 1
        const t = 1 - value; // Para mapear a escala de azul

        const r = 255 * t;
        const g = 255 * t;
        const b = 255;

        const index = (i * convOutputWidth + j) * 4;
        reluImageData.data[index] = Math.round(r);
        reluImageData.data[index + 1] = Math.round(g);
        reluImageData.data[index + 2] = Math.round(b);
        reluImageData.data[index + 3] = 255; // Alpha
      }
    }

    // Crear ImageData para la imagen Max Pooling (salida)
    const pooledImageData = new ImageData(pooledWidth, pooledHeight);

    for (let i = 0; i < pooledData.length; i++) {
      for (let j = 0; j < pooledData[0].length; j++) {
        const value = pooledData[i][j]; // Valor entre 0 y 1
        const t = 1 - value; // Para mapear a escala de azul

        const r = 255 * t;
        const g = 255 * t;
        const b = 255;

        const index = (i * pooledWidth + j) * 4;
        pooledImageData.data[index] = Math.round(r);
        pooledImageData.data[index + 1] = Math.round(g);
        pooledImageData.data[index + 2] = Math.round(b);
        pooledImageData.data[index + 3] = 255; // Alpha
      }
    }

    // Dibujar la imagen ReLU en el canvas de entrada
    inputCtx.putImageData(reluImageData, 0, 0);

    // Dibujar la imagen Max Pooling en el canvas de salida
    outputCtx.putImageData(pooledImageData, 0, 0);

    const croppedDisplayWidth = 62 * displayScale;
    const croppedDisplayHeight = 62 * displayScale;

    outputHighlightCanvas.width = croppedDisplayWidth;
    outputHighlightCanvas.height = croppedDisplayHeight;
    outputHighlightCanvas.style.width = `${croppedDisplayWidth*2}px`;
    outputHighlightCanvas.style.height = `${croppedDisplayHeight*2}px`;
    inputHighlightCanvas.width = croppedDisplayWidth;
    inputHighlightCanvas.height = croppedDisplayHeight;
    
    outputHighlightCanvas.style.imageRendering = 'pixelated';
    outputHighlightCanvas.imageSmoothingEnabled = false;


    // Manejar la interactividad

    inputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMove(event, inputCanvas, inputHighlightCanvas, outputHighlightCanvas);
    });

    outputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMoveOutput(event, outputCanvas, outputHighlightCanvas, inputHighlightCanvas);
    });

  };

  inputImage.onerror = () => {
    console.error('Failed to load image. Check the image URL or CORS settings.');
  };

  // Función para manejar el movimiento del mouse en el canvas de entrada
  function handleMouseMove(event, canvas, highlightCanvas, otherHighlightCanvas) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const x = Math.floor(mouseX / displayScale);
    const y = Math.floor(mouseY / displayScale);

    const width = canvas.width;
    const height = canvas.height;

    // Verificar los límites de la imagen
    if (x >= 0 && x < width && y >= 0 && y < height) {
      // Calcular la posición en el mapa de salida
      const pooledX = Math.floor(x / poolingStride);
      const pooledY = Math.floor(y / poolingStride);

      // Resaltar la región de pooling en el canvas de entrada
      highlightRegion(highlightCanvas, x, y, poolingSize);

      // Resaltar el píxel correspondiente en el canvas de salida
      highlightSinglePixel(otherHighlightCanvas, pooledX, pooledY);

      // Obtener los valores de los píxeles en la región de pooling
      const regionValues = [];
      for (let i = 0; i < poolingSize; i++) {
        for (let j = 0; j < poolingSize; j++) {
          const xi = x - (x % poolingStride) + j;
          const yi = y - (y % poolingStride) + i;
          if (xi < width && yi < height) {
            regionValues.push(reluData[yi][xi]);
          }
        }
      }

      const maxValue = Math.max(...regionValues);

      // Actualizar el SVG con los valores
      updatePoolingSVG(regionValues, maxValue);
    }
  }

  // Función para manejar el movimiento del mouse en el canvas de salida
  function handleMouseMoveOutput(event, canvas, highlightCanvas, otherHighlightCanvas) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const x = Math.floor(mouseX / displayScale/2);
    const y = Math.floor(mouseY / displayScale/2);

    const width = canvas.width;
    const height = canvas.height;

    // Verificar los límites de la imagen
    if (x >= 0 && x < width && y >= 0 && y < height) {
      // Resaltar el píxel en el canvas de salida
      highlightSinglePixel(highlightCanvas, x, y);

      // Resaltar la región de pooling correspondiente en el canvas de entrada
      const startX = x * poolingStride;
      const startY = y * poolingStride;
      highlightRegion(otherHighlightCanvas, startX, startY, poolingSize);

      // Obtener los valores de los píxeles en la región de pooling
      const regionValues = [];
      for (let i = 0; i < poolingSize; i++) {
        for (let j = 0; j < poolingSize; j++) {
          const xi = startX + j;
          const yi = startY + i;
          if (xi < reluData[0].length && yi < reluData.length) {
            regionValues.push(reluData[yi][xi]);
          }
        }
      }

      const maxValue = pooledData[y][x];

      // Actualizar el SVG con los valores
      updatePoolingSVG(regionValues, maxValue);
    }
  }

  // Función para resaltar una región en el canvas
  function highlightRegion(highlightCanvas, x, y, size) {
    const ctx = highlightCanvas.getContext('2d');
    // Limpiar el canvas de resaltado
    ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

    ctx.save();
    ctx.scale(displayScale, displayScale);
    ctx.fillStyle = 'rgba(64, 64, 64, 0.7)'; 
    ctx.fillRect(x - (x % poolingStride)+0.125, y - (y % poolingStride)+0.125, size, size);
    ctx.restore();
  }

  // Función para resaltar un solo píxel
  function highlightSinglePixel(highlightCanvas, x, y) {
    const ctx = highlightCanvas.getContext('2d');
    // Limpiar el canvas de resaltado
    ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

    ctx.save();
    ctx.scale(displayScale, displayScale);
    ctx.fillStyle = 'rgba(64, 64, 64, 0.7)';
    ctx.fillRect(x+0.125 , y+0.125 , 1, 1);
    ctx.restore();
  }

  // Función para limpiar el canvas de resaltado
  function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  // Función para extraer un canal específico (si usas canales de color)
  function extractChannel(data, channelIndex) {
    const channelData = new Float32Array(data.length / 4);

    for (let i = 0; i < data.length; i += 4) {
      channelData[i / 4] = data[i + channelIndex];
    }
    return channelData;
  }

  // Función para inicializar el SVG de Max Pooling
  function initializePoolingSVG() {
    const cellSize = 40;
    const gap = 10;
    const poolingSize = 2; // Asumiendo un tamaño de pooling de 2
    const svgWidth = 240;
    const svgHeight = 100;
  
    poolingSVG = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    poolingSVG.setAttribute("width", svgWidth);
    poolingSVG.setAttribute("height", svgHeight);
    svgMatrixContainer.appendChild(poolingSVG);
  
    poolingRects = [];
    poolingTexts = [];
  
    // Estimaciones de anchuras para los elementos de texto
    const maxTextWidth = 30; // Ancho aproximado de "max("
    const closeParenWidth = 10; // Ancho aproximado de ")"
    const equalSignWidth = 15; // Ancho aproximado de "="
  
    // Calcular el ancho total de los rectángulos y los espacios
    const rectanglesWidth = (cellSize * poolingSize) + (gap * (poolingSize - 1)); // (40*2) + (10*1) = 90
    const resultRectWidth = cellSize; // 40
  
    // Calcular el ancho total del contenido
    const contentWidth = maxTextWidth + rectanglesWidth + closeParenWidth + equalSignWidth + resultRectWidth;
  
    // Calcular el margen izquierdo para centrar el contenido
    const leftMargin = 2;
  
    // Posición vertical centrada
    const yPosition = (svgHeight - cellSize * poolingSize - gap * (poolingSize - 1)) / 2;
  
    let currentX = leftMargin;
  
    // "max("
    const maxText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    maxText.setAttribute("x", currentX);
    maxText.setAttribute("y", svgHeight / 2);
    maxText.setAttribute("font-size", "18px");
    maxText.setAttribute("text-anchor", "start");
    maxText.setAttribute("dominant-baseline", "middle");
    maxText.textContent = "max(";
    poolingSVG.appendChild(maxText);
  
    // Actualizar currentX después de "max("
    currentX += maxTextWidth;
  
    // Crear los rectángulos y textos para la región de pooling
    for (let i = 0; i < poolingSize; i++) {
      for (let j = 0; j < poolingSize; j++) {
        const x = currentX + j * (cellSize + gap)+15;
        const y = yPosition + i * (cellSize + gap);
  
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", x);
        rect.setAttribute("y", y);
        rect.setAttribute("width", cellSize);
        rect.setAttribute("height", cellSize);
        rect.setAttribute("fill", 'white');
        rect.setAttribute("stroke", "black");
        rect.setAttribute("rx", "2");
        rect.setAttribute("ry", "2");
        poolingSVG.appendChild(rect);
        poolingRects.push(rect);
  
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x + cellSize / 2);
        text.setAttribute("y", y + cellSize / 2);
        text.setAttribute("font-size", "14px");
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dominant-baseline", "middle");
        text.textContent = '0.00';
        poolingSVG.appendChild(text);
        poolingTexts.push(text);
      }
    }
  
    // Actualizar currentX después de los rectángulos
    currentX += rectanglesWidth;
  
    // ")"
    const closeParenText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    closeParenText.setAttribute("x", currentX+23);
    closeParenText.setAttribute("y", svgHeight / 2);
    closeParenText.setAttribute("font-size", "18px");
    closeParenText.setAttribute("text-anchor", "start");
    closeParenText.setAttribute("dominant-baseline", "middle");
    closeParenText.textContent = ")";
    poolingSVG.appendChild(closeParenText);
  
    currentX += closeParenWidth;
  
    // "="
    const equalSignText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    equalSignText.setAttribute("x", currentX+23);
    equalSignText.setAttribute("y", svgHeight / 2);
    equalSignText.setAttribute("font-size", "18px");
    equalSignText.setAttribute("text-anchor", "start");
    equalSignText.setAttribute("dominant-baseline", "middle");
    equalSignText.textContent = "=";
    poolingSVG.appendChild(equalSignText);
  
    currentX += equalSignWidth;
  
    // Rectángulo del resultado
    const resultX = currentX +32;
    const resultY = (svgHeight - cellSize) / 2;
  
    resultRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    resultRect.setAttribute("x", resultX);
    resultRect.setAttribute("y", resultY);
    resultRect.setAttribute("width", cellSize);
    resultRect.setAttribute("height", cellSize);
    resultRect.setAttribute("fill", 'white');
    resultRect.setAttribute("stroke", "black");
    resultRect.setAttribute("rx", "2");
    resultRect.setAttribute("ry", "2");
    poolingSVG.appendChild(resultRect);
  
    resultText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    resultText.setAttribute("x", resultX + cellSize / 2);
    resultText.setAttribute("y", resultY + cellSize / 2);
    resultText.setAttribute("font-size", "14px");
    resultText.setAttribute("text-anchor", "middle");
    resultText.setAttribute("dominant-baseline", "middle");
    resultText.textContent = '0.00';
    poolingSVG.appendChild(resultText);
  }
  

  // Función para actualizar el SVG de Max Pooling con nuevos valores
  function updatePoolingSVG(regionValues, maxValue) {
    if (regionValues === null) {
      poolingTexts.forEach((text) => text.textContent = '');
      resultText.textContent = '';
      poolingRects.forEach((rect) => rect.setAttribute("fill", 'lightgray'));
      resultRect.setAttribute("fill", 'lightgray');
    } else {
      for (let i = 0; i < regionValues.length; i++) {
        poolingTexts[i].textContent = regionValues[i].toFixed(2);
        const value = regionValues[i];
        const t = 1 - value;
        const r = 255 * t;
        const g = 255 * t;
        const b = 255;
        poolingRects[i].setAttribute("fill", `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`);
      }
      resultText.textContent = maxValue.toFixed(2);
      const t = 1 - maxValue;
      const r = 255 * t;
      const g = 255 * t;
      const b = 255;
      resultRect.setAttribute("fill", `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`);
    }
  }
}
