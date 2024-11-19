import * as tf from '@tensorflow/tfjs';

export function startReLU({ container, imageSrc, filter, channel = 'grayscale' }) {
  // Variables y constantes iniciales
  let svgMatrix;
  let pixelTexts = [];
  let filterTexts = [];
  let resultText;
  let resultRect;
  const operationWidth = 64;
  const operationHeight = 64;
  const displayScale = 4; // Factor de escala para agrandar cada píxel
  const displayWidth = (operationWidth - 2) * displayScale; // Ajustar para imagen convolucionada
  const displayHeight = (operationHeight - 2) * displayScale;

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

  // Crear el elemento de texto para 'Entrada Convolucionada (62, 62)'
  const inputLabel = document.createElement('div');
  inputLabel.textContent = 'Entrada (62, 62)';
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

  // Crear el elemento de texto para 'Salida ReLU (62, 62)'
  const outputLabel = document.createElement('div');
  outputLabel.textContent = 'Salida ReLU (62, 62)';
  outputLabel.style.textAlign = 'center';
  outputLabel.style.marginBottom = '5px';

  // Crear los canvas para la imagen de entrada y salida
  const inputCanvas = document.createElement('canvas');
  inputCanvas.width = (operationWidth - 2);
  inputCanvas.height = (operationHeight - 2);
  inputCanvas.style.width = `${displayWidth}px`;
  inputCanvas.style.height = `${displayHeight}px`;
  inputCanvas.style.border = '1px solid black';
  inputCanvas.style.imageRendering = 'pixelated';
  inputWrapper.appendChild(inputLabel);
  inputCanvasContainer.appendChild(inputCanvas);
  inputWrapper.appendChild(inputCanvasContainer);

  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = (operationWidth - 2);
  outputCanvas.height = (operationHeight - 2);
  outputCanvas.style.width = `${displayWidth}px`;
  outputCanvas.style.height = `${displayHeight}px`;
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

  // Crear el contenedor para la visualización de la matriz (SVG)
  const svgMatrixContainer = document.createElement('div');
  svgMatrixContainer.style.display = 'flex';
  svgMatrixContainer.style.alignItems = 'center';
  svgMatrixContainer.style.justifyContent = 'center';
  mainContainer.appendChild(inputWrapper);
  mainContainer.appendChild(svgMatrixContainer);
  mainContainer.appendChild(outputWrapper);

  // Crear un elemento para mostrar los valores del píxel
  const pixelValueDisplay = document.createElement('div');
  pixelValueDisplay.style.marginTop = '10px';
  pixelValueDisplay.style.textAlign = 'center';
  //svgMatrixContainer.appendChild(pixelValueDisplay);

  // Inicializar el SVG de la matriz
  let reluSVG, xTextElement, yTextElement, xRect, yRect;
  initializeReLUSVG();
  let convData, reluData;


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

    // Obtener los datos del tensor convolucionado y del tensor ReLU
    convData = await convTensor.squeeze().data();
    reluData = await reluTensor.squeeze().data();

    // Crear ImageData para la imagen convolucionada (entrada)
    const convImageData = new ImageData(operationWidth - 2, operationHeight - 2);

    for (let i = 0; i < convData.length; i++) {
      const value = convData[i]; // Valor entre -1 y 1
      let r, g, b;

      if (value < 0) {
        const t = (value + 1); // Mapear de [-1, 0] a [0, 1]
        r = 255;
        g = 0 + 255 * t;
        b = 0 + 255 * t;
      } else {
        const t = 1 - value;
        r = 255 * t;
        g = 255 * t;
        b = 255;
      }

      const index = i * 4;
      convImageData.data[index] = Math.round(r);
      convImageData.data[index + 1] = Math.round(g);
      convImageData.data[index + 2] = Math.round(b);
      convImageData.data[index + 3] = 255; // Alpha
    }

    // Crear ImageData para la imagen ReLU (salida)
    const reluImageData = new ImageData(operationWidth - 2, operationHeight - 2);

    for (let i = 0; i < reluData.length; i++) {
      const value = reluData[i]; // Valor entre 0 y máximo normalizado
      const normalizedValue = value; // Ya está entre 0 y 1
      const t = 1 - normalizedValue; // Para mapear a escala de azul

      const r = 255 * t;
      const g = 255 * t;
      const b = 255;

      const index = i * 4;
      reluImageData.data[index] = Math.round(r);
      reluImageData.data[index + 1] = Math.round(g);
      reluImageData.data[index + 2] = Math.round(b);
      reluImageData.data[index + 3] = 255; // Alpha
    }

    // Dibujar la imagen convolucionada en el canvas de entrada
    inputCtx.putImageData(convImageData, 0, 0);

    // Dibujar la imagen ReLU en el canvas de salida
    outputCtx.putImageData(reluImageData, 0, 0);

    const croppedDisplayWidth = 62 * displayScale;
    const croppedDisplayHeight = 62 * displayScale;

    outputHighlightCanvas.width = croppedDisplayWidth;
    outputHighlightCanvas.height = croppedDisplayHeight;
    outputHighlightCanvas.style.width = `${croppedDisplayWidth}px`;
    outputHighlightCanvas.style.height = `${croppedDisplayHeight}px`;
    inputHighlightCanvas.width = croppedDisplayWidth;
    inputHighlightCanvas.height = croppedDisplayHeight;
    // Manejar la interactividad

    inputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMove(event, inputCanvas, inputHighlightCanvas, outputHighlightCanvas, 'input', convData, reluData);
    });

    outputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMove(event, outputCanvas, outputHighlightCanvas, inputHighlightCanvas, 'output', convData, reluData);
    });

    
  };

  inputImage.onerror = () => {
    console.error('Failed to load image. Check the image URL or CORS settings.');
  };

  function handleMouseMove(event, canvas, highlightCanvas, otherHighlightCanvas, canvasType, convData, reluData) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const x = Math.floor(mouseX / displayScale);
    const y = Math.floor(mouseY / displayScale);

    const width = canvas.width;
    const height = canvas.height;

    // Verificar los límites de la imagen
    if (x >= 0 && x < width && y >= 0 && y < height) {
      // Resaltar el píxel en el canvas actual
      highlightSinglePixel(highlightCanvas, x, y);

      // Resaltar el píxel correspondiente en el otro canvas
      highlightSinglePixel(otherHighlightCanvas, x, y);

      // Obtener el índice del píxel
      const index = y * width + x;

      // Obtener los valores del píxel antes y después de ReLU
      const convValue = convData[index]; // Valor antes de ReLU
      const reluValue = reluData[index]; // Valor después de ReLU

      // Mostrar estos valores en el elemento DOM
      // pixelValueDisplay.textContent = `Posición: (${x}, ${y}), Valor antes de ReLU: ${convValue.toFixed(2)}, después de ReLU: ${reluValue.toFixed(2)}`;
      updateReLUSVG(convValue, reluValue);
    }
  }

  // Función para resaltar un solo píxel
  function highlightSinglePixel(highlightCanvas, x, y) {
    const ctx = highlightCanvas.getContext('2d');
    // Limpiar el canvas de resaltado
    ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

    ctx.save();
    ctx.scale(displayScale, displayScale);
    ctx.fillStyle = 'rgba(64, 64, 64, 0.7)';
    ctx.lineWidth = 1 / displayScale;
    ctx.strokeRect(x+0.25, y+0.25, 1, 1);
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

  // Función para inicializar el SVG de ReLU
  function initializeReLUSVG() {
    // Usar las mismas dimensiones que en startConvolution
    const cellWidth = 40; // Ancho de cada celda
    const cellHeight = 40; // Altura de cada celda
    const gap = 10; // Espacio entre elementos
    const svgWidth = 240; // Ancho total del SVG
    const svgHeight = cellHeight + gap * 2; // Altura total del SVG

    // Crear el SVG de ReLU
    reluSVG = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    reluSVG.setAttribute("width", svgWidth);
    reluSVG.setAttribute("height", svgHeight);
    svgMatrixContainer.appendChild(reluSVG);

    const yPosition = gap;

    // Posiciones horizontales
    const xPosition_maxText = 5;

    // "max("
    const maxText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    maxText.setAttribute("x", xPosition_maxText);
    maxText.setAttribute("y", svgHeight / 2);
    maxText.setAttribute("font-size", "18px");
    maxText.setAttribute("text-anchor", "start");
    maxText.setAttribute("dominant-baseline", "middle");
    maxText.textContent = "max(";
    reluSVG.appendChild(maxText);

    // Rectángulo para 0.00
    const xPosition_zeroGroup = xPosition_maxText + 45; // Ajustar según el ancho de "max("
    const zeroGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    zeroGroup.setAttribute("transform", `translate(${xPosition_zeroGroup}, ${yPosition})`);
    reluSVG.appendChild(zeroGroup);

    const zeroRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    zeroRect.setAttribute("width", cellWidth);
    zeroRect.setAttribute("height", cellHeight);
    zeroRect.setAttribute("fill", 'white');
    zeroRect.setAttribute("stroke", "black");
    zeroRect.setAttribute("rx", "2");
    zeroRect.setAttribute("ry", "2");
    zeroGroup.appendChild(zeroRect);

    const zeroText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    zeroText.setAttribute("x", cellWidth / 2);
    zeroText.setAttribute("y", cellHeight / 2);
    zeroText.setAttribute("font-size", "14px");
    zeroText.setAttribute("text-anchor", "middle");
    zeroText.setAttribute("dominant-baseline", "middle");
    zeroText.textContent = "0.00";
    zeroGroup.appendChild(zeroText);

    // ","
    const xPosition_commaText = xPosition_zeroGroup + cellWidth + 5;
    const commaText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    commaText.setAttribute("x", xPosition_commaText);
    commaText.setAttribute("y", svgHeight / 2);
    commaText.setAttribute("font-size", "18px");
    commaText.setAttribute("text-anchor", "start");
    commaText.setAttribute("dominant-baseline", "middle");
    commaText.textContent = ",";
    reluSVG.appendChild(commaText);

    // Rectángulo para x
    const xPosition_xGroup = xPosition_commaText + 10;
    const xGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    xGroup.setAttribute("transform", `translate(${xPosition_xGroup}, ${yPosition})`);
    reluSVG.appendChild(xGroup);

    xRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    xRect.setAttribute("width", cellWidth);
    xRect.setAttribute("height", cellHeight);
    xRect.setAttribute("fill", 'white');
    xRect.setAttribute("stroke", "black");
    xRect.setAttribute("rx", "2");
    xRect.setAttribute("ry", "2");
    xGroup.appendChild(xRect);

    xTextElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xTextElement.setAttribute("x", cellWidth / 2);
    xTextElement.setAttribute("y", cellHeight / 2);
    xTextElement.setAttribute("font-size", "14px");
    xTextElement.setAttribute("text-anchor", "middle");
    xTextElement.setAttribute("dominant-baseline", "middle");
    xTextElement.textContent = "0.00";
    xGroup.appendChild(xTextElement);

    // ") ="
    const xPosition_closeParenEqualText = xPosition_xGroup + cellWidth + 5;
    const closeParenEqualText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    closeParenEqualText.setAttribute("x", xPosition_closeParenEqualText);
    closeParenEqualText.setAttribute("y", svgHeight / 2);
    closeParenEqualText.setAttribute("font-size", "18px");
    closeParenEqualText.setAttribute("text-anchor", "start");
    closeParenEqualText.setAttribute("dominant-baseline", "middle");
    closeParenEqualText.textContent = ") =";
    reluSVG.appendChild(closeParenEqualText);

    // Rectángulo para y
    const xPosition_yGroup = xPosition_closeParenEqualText + 35; // Ajustar según el ancho de ") ="
    const yGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    yGroup.setAttribute("transform", `translate(${xPosition_yGroup}, ${yPosition})`);
    reluSVG.appendChild(yGroup);

    yRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    yRect.setAttribute("width", cellWidth);
    yRect.setAttribute("height", cellHeight);
    yRect.setAttribute("fill", 'white');
    yRect.setAttribute("stroke", "black");
    yRect.setAttribute("rx", "2");
    yRect.setAttribute("ry", "2");
    yGroup.appendChild(yRect);

    yTextElement = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yTextElement.setAttribute("x", cellWidth / 2);
    yTextElement.setAttribute("y", cellHeight / 2);
    yTextElement.setAttribute("font-size", "14px");
    yTextElement.setAttribute("text-anchor", "middle");
    yTextElement.setAttribute("dominant-baseline", "middle");
    yTextElement.textContent = "0.00";
    yGroup.appendChild(yTextElement);
  }

  // Función para actualizar el SVG de ReLU con nuevos valores
  function updateReLUSVG(convValue, reluValue) {
    if (convValue === null || reluValue === null) {
      xTextElement.textContent = '';
      yTextElement.textContent = '';
      xRect.setAttribute("fill", 'lightgray');
      yRect.setAttribute("fill", 'lightgray');
    } else {
      xTextElement.textContent = convValue.toFixed(2);
      yTextElement.textContent = reluValue.toFixed(2);

      // Actualizar el color del rectángulo xRect basado en convValue
      let xColor;
      if (convValue < 0) {
        const t = (convValue / -1); // Mapear de [-1, 0] a [0, 1]
        const r = 255;
        const g = 255 * (1 - t);
        const b = 255 * (1 - t);
        xColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
      } else {
        const t = 1 - convValue; // Mapear de [0, 1] a [1, 0]
        const r = 255 * t;
        const g = 255 * t;
        const b = 255;
        xColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
      }
      xRect.setAttribute("fill", xColor);

      // Actualizar el color del rectángulo yRect basado en reluValue
      let yColor;
      if (reluValue === 0) {
        yColor = 'rgb(255, 255, 255)'; // Blanco
      } else {
        const t = 1 - reluValue; // Mapear de [0, 1] a [1, 0]
        const r = 255 * t;
        const g = 255 * t;
        const b = 255;
        yColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
      }
      yRect.setAttribute("fill", yColor);
    }
  }
}
