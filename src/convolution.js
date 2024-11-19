import * as tf from '@tensorflow/tfjs';

export function startConvolution({ container, imageSrc, filter, channel = 'grayscale' }) {
  let svgMatrix;
  let pixelTexts = [];
  let filterTexts = [];
  let resultText;
  let resultRect;
  const operationWidth = 64;
  const operationHeight = 64;
  const displayScale = 4; // Factor de escala para agrandar cada píxel
  const displayWidth = operationWidth * displayScale;
  const displayHeight = operationHeight * displayScale;
  
  let maxAbsValue = 0;

  const containerElement = document.getElementById(container);

  // Crear el contenedor principal
  const mainContainer = document.createElement('div');
  mainContainer.style.display = 'flex';
  mainContainer.style.alignItems = 'end';
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

  // Crear el elemento de texto para 'Entrada (64, 64)'
  const inputLabel = document.createElement('div');
  inputLabel.textContent = 'Entrada (64, 64)';
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

  // Crear el elemento de texto para 'Salida (62, 62)'
  const outputLabel = document.createElement('div');
  outputLabel.textContent = 'Salida (62, 62)';
  outputLabel.style.textAlign = 'center';
  outputLabel.style.marginBottom = '5px';

  // Crear los canvas para la imagen de entrada y salida
  const inputCanvas = document.createElement('canvas');
  inputCanvas.width = displayWidth;
  inputCanvas.height = displayHeight;
  inputCanvas.style.border = '1px solid black';
  inputWrapper.appendChild(inputLabel);
  inputCanvasContainer.appendChild(inputCanvas);
  inputWrapper.appendChild(inputCanvasContainer);

  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = displayWidth;
  outputCanvas.height = displayHeight;
  outputCanvas.style.border = '1px solid black';
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
  inputHighlightCanvas.width = displayWidth;
  inputHighlightCanvas.height = displayHeight;
  inputHighlightCanvas.style.position = 'absolute';
  inputHighlightCanvas.style.left = '0';
  inputHighlightCanvas.style.top = '0';
  inputHighlightCanvas.style.pointerEvents = 'none';
  inputCanvasContainer.appendChild(inputHighlightCanvas);

  const outputHighlightCanvas = document.createElement('canvas');
  outputHighlightCanvas.width = displayWidth;
  outputHighlightCanvas.height = displayHeight;
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

  // Inicializar el SVG de la matriz
  initializeMatrixSVG();

  function initializeMatrixSVG() {
    // Calcular dimensiones del SVG
    const cellWidth = 40; // Ancho de cada celda
    const pixelHeight = 40; // Altura del rectángulo superior para el valor del píxel
    const filterHeight = pixelHeight / 3; // Un tercio de la altura para el rectángulo inferior (valor del filtro)
    const gap = 40; // Espacio horizontal entre celdas (ajustado para mayor separación)
    const pairSpacing = 20; // Espacio adicional entre cada par de `pixel` y `filtro`
  
    const svgWidth = (cellWidth + gap) * 3;
    const svgHeight = (pixelHeight + filterHeight + pairSpacing) * 3 + 35;
  
    // Crear el SVG de la matriz 3x3
    svgMatrix = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svgMatrix.setAttribute("width", svgWidth);
    svgMatrix.setAttribute("height", svgHeight);
    svgMatrixContainer.appendChild(svgMatrix);
  
    // Crear los elementos gráficos y almacenarlos en arrays
    for (let index = 0; index < 9; index++) {
      const col = index % 3;
      const row = Math.floor(index / 3);
  
      const x = col * (cellWidth + gap) + 15;
      const y = row * (pixelHeight + filterHeight + pairSpacing);
  
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("transform", `translate(${x}, ${y})`);
      svgMatrix.appendChild(group);
  
      // Rectángulo superior para el valor del píxel
      const rectPixel = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rectPixel.setAttribute("width", cellWidth);
      rectPixel.setAttribute("height", pixelHeight);
      rectPixel.setAttribute("fill", 'rgba(0, 0, 0, 0.75)');
      rectPixel.setAttribute("stroke", "black");
      rectPixel.setAttribute("rx", "2"); 
      rectPixel.setAttribute("ry", "2"); 
      rectPixel.setAttribute("stroke-width", "1");
      group.appendChild(rectPixel);
  
      const pixelText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      pixelText.setAttribute("x", cellWidth / 2);
      pixelText.setAttribute("y", 2 + pixelHeight / 2);
      pixelText.setAttribute("font-size", "14px");
      pixelText.setAttribute("fill", "black");
      pixelText.setAttribute("text-anchor", "middle");
      pixelText.setAttribute("dominant-baseline", "middle");
      pixelText.textContent = "0.00";
      group.appendChild(pixelText);
      pixelTexts.push(pixelText); // Almacenar referencia
  
      // Agregar el símbolo "x" a la izquierda del rectángulo inferior
      const xSymbol = document.createElementNS("http://www.w3.org/2000/svg", "text");
      xSymbol.setAttribute("x", -5);
      xSymbol.setAttribute("y", pixelHeight + 2 + filterHeight / 2);
      xSymbol.setAttribute("font-size", "12px");
      xSymbol.setAttribute("fill", "black");
      xSymbol.setAttribute("text-anchor", "middle");
      xSymbol.setAttribute("dominant-baseline", "middle");
      xSymbol.textContent = "×";
      group.appendChild(xSymbol);
  
      // Rectángulo inferior para el valor del filtro
      const rectFilter = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rectFilter.setAttribute("x", 0);
      rectFilter.setAttribute("y", pixelHeight);
      rectFilter.setAttribute("width", cellWidth);
      rectFilter.setAttribute("height", filterHeight);
      rectFilter.setAttribute("fill", 'rgba(255, 255, 255, 1)');
      rectFilter.setAttribute("stroke-width", "0");
      group.appendChild(rectFilter);
  
      const filterText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      filterText.setAttribute("x", cellWidth / 2);
      filterText.setAttribute("y", pixelHeight + 2 + filterHeight / 2);
      filterText.setAttribute("font-size", "14px");
      filterText.setAttribute("fill", "black");
      filterText.setAttribute("text-anchor", "middle");
      filterText.setAttribute("dominant-baseline", "middle");
      filterText.textContent = "0";
      group.appendChild(filterText);
      filterTexts.push(filterText); // Almacenar referencia
  
      // Agregar el símbolo "+" o "="
      const plusSymbol = document.createElementNS("http://www.w3.org/2000/svg", "text");
      plusSymbol.setAttribute("x", cellWidth + 18);
      plusSymbol.setAttribute("y", 2 + pixelHeight / 2);
      plusSymbol.setAttribute("font-size", "18px");
      plusSymbol.setAttribute("fill", "black");
      plusSymbol.setAttribute("text-anchor", "middle");
      plusSymbol.setAttribute("dominant-baseline", "middle");
      plusSymbol.textContent = index < 8 ? "+" : "=";
      group.appendChild(plusSymbol);
    }
  
    // Crear los elementos para el resultado
    const resultGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    resultGroup.setAttribute("transform", `translate(${svgWidth / 2 - cellWidth / 2 - 5}, ${svgHeight - 40})`);
    svgMatrix.appendChild(resultGroup);
  
    resultRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    resultRect.setAttribute("width", cellWidth);
    resultRect.setAttribute("height", pixelHeight);
    resultRect.setAttribute("fill", 'rgba(0, 0, 0, 0.75)');
    resultRect.setAttribute("stroke", "black");
    resultRect.setAttribute("rx", "2");
    resultRect.setAttribute("ry", "2");
    resultGroup.appendChild(resultRect);
  
    resultText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    resultText.setAttribute("x", cellWidth / 2);
    resultText.setAttribute("y", pixelHeight / 2);
    resultText.setAttribute("font-size", "14px");
    resultText.setAttribute("text-anchor", "middle");
    resultText.setAttribute("dominant-baseline", "middle");
    resultText.textContent = "0.00";
    resultGroup.appendChild(resultText);
  }
  

  outputWrapper.appendChild(outputCanvasContainer);
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
    const data = imageData.data;

    
    // Crear ImageData para la imagen de entrada según el canal seleccionado
    let inputImageData;

    if (channel === 'grayscale') {
      // Convertir a escala de grises
      inputImageData = new ImageData(operationWidth, operationHeight);

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        inputImageData.data[i] = gray;
        inputImageData.data[i + 1] = gray;
        inputImageData.data[i + 2] = gray;
        inputImageData.data[i + 3] = 255;
      }
    } else if (channel === 'R' || channel === 'G' || channel === 'B') {
      // Extraer el canal seleccionado
      const channelIndex = { 'R': 0, 'G': 1, 'B': 2 }[channel];
      inputImageData = new ImageData(operationWidth, operationHeight);

      for (let i = 0; i < data.length; i += 4) {
        const value = data[i + channelIndex];
        inputImageData.data[i] = channel === 'R' ? value : 0;
        inputImageData.data[i + 1] = channel === 'G' ? value : 0;
        inputImageData.data[i + 2] = channel === 'B' ? value : 0;
        inputImageData.data[i + 3] = 255;
      }
    } else {
      // Mostrar la imagen normal
      inputImageData = imageData;
    }

    // Dibujar la imagen escalada en el canvas de entrada
    inputCtx.putImageData(inputImageData, 0, 0);

    // Escalar la imagen en el canvas de entrada
    inputCtx.save();
    inputCtx.scale(displayScale, displayScale);
    inputCtx.drawImage(inputCanvas, 0, 0);
    inputCtx.restore();


    // Realizar la convolución utilizando TensorFlow.js
    const filterSize = Math.sqrt(filter.length);
    
    function extractChannel(data, channelIndex) {
      const channelData = new Float32Array(data.length / 4);
    
      for (let i = 0; i < data.length; i += 4) {
        channelData[i / 4] = data[i + channelIndex];
      }
      return channelData;
    }

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
    let outputTensor = tf.conv2d(inputTensor, filterTensor, [1, 1], 'same');

    // Obtener el valor máximo absoluto para normalizar
    maxAbsValue = (await tf.max(tf.abs(outputTensor)).data())[0];

    // Normalizar el tensor de salida al rango [-1, 1]
    outputTensor = outputTensor.div(maxAbsValue);

    // Quitar las dimensiones extra
    outputTensor = outputTensor.squeeze();

    // Obtener los datos del tensor de salida
    const outputData = await outputTensor.data();

    // Crear un ImageData para la imagen de salida
    const outputImageData = new ImageData(operationWidth, operationHeight);

    for (let i = 0; i < outputData.length; i++) {
      const value = outputData[i]; // Valor entre -1 y 1
      let r, g, b;

      if (value < 0) {
        // Valores negativos: rojo oscuro a rojo claro
        const t = (value + 1); // Mapear de [-1, 0] a [0, 1]
        r = 255;   // De 0 a 255 (rojo oscuro a rojo claro)
        g = 0 + 255 * t;
        b = 0 + 255 * t;
      } else {
        // Valores positivos: azul oscuro a azul claro
        const t = 1- value; // Mapear de [0, 1] a [0, 1]
        r = 255 * t;
        g = 255 * t;
        b = 255;   // De 0 a 255 (azul oscuro a azul claro)
      }

      const index = i * 4;
      outputImageData.data[index] = Math.round(r);
      outputImageData.data[index + 1] = Math.round(g);
      outputImageData.data[index + 2] = Math.round(b);
      outputImageData.data[index + 3] = 255; // Alpha
    }

    const croppedWidth = operationWidth - 2; // 62
    const croppedHeight = operationHeight - 2; // 62

    const croppedImageData = new ImageData(croppedWidth, croppedHeight);

    // Copiar los datos centrales al croppedImageData
    for (let y = 1; y < operationHeight - 1; y++) {
      for (let x = 1; x < operationWidth - 1; x++) {
        const srcIndex = (y * operationWidth + x) * 4;
        const destIndex = ((y - 1) * croppedWidth + (x - 1)) * 4;
        croppedImageData.data[destIndex] = outputImageData.data[srcIndex];
        croppedImageData.data[destIndex + 1] = outputImageData.data[srcIndex + 1];
        croppedImageData.data[destIndex + 2] = outputImageData.data[srcIndex + 2];
        croppedImageData.data[destIndex + 3] = outputImageData.data[srcIndex + 3];
      }
    }

    // Ajustar el tamaño del canvas de salida
    const croppedDisplayWidth = croppedWidth * displayScale;
    const croppedDisplayHeight = croppedHeight * displayScale;

    outputCanvas.width = croppedWidth; // 62
    outputCanvas.height = croppedHeight; // 62
    outputHighlightCanvas.width = croppedDisplayWidth;
    outputHighlightCanvas.height = croppedDisplayHeight;

    outputCanvas.style.width = `${croppedDisplayWidth}px`;
    outputCanvas.style.height = `${croppedDisplayHeight}px`;
    outputHighlightCanvas.style.width = `${croppedDisplayWidth}px`;
    outputHighlightCanvas.style.height = `${croppedDisplayHeight}px`;

    outputCtx.imageSmoothingEnabled = false;
    outputCanvas.style.imageRendering = 'pixelated';
    outputHighlightCanvas.style.imageRendering = 'pixelated';

    outputCtx.putImageData(croppedImageData, 0, 0);

    // Dibujar la imagen escalada en el canvas de salida
    // outputCtx.save();
    // outputCtx.scale(displayScale, displayScale);
    // outputCtx.putImageData(croppedImageData, 0, 0);
    // outputCtx.restore();

    // Crear los rectángulos de resaltado para la matriz 3x3 en la imagen de entrada
    const highlightRects = [];
    for (let i = 0; i < 9; i++) {
      const rect = {
        x: 0,
        y: 0,
      };
      highlightRects.push(rect);
    }

    const initialX = Math.floor(operationWidth / 2);
    const initialY = Math.floor(operationHeight / 2);

    // Mapear a la posición en la imagen de entrada
    const initialXInInput = initialX + 1; // Compensar el recorte
    const initialYInInput = initialY + 1;

    const initialMatrixValues = [];
    let initialResultSum = 0;

    let index = 0;

    for (let offsetY = -1; offsetY <= 1; offsetY++) {
      for (let offsetX = -1; offsetX <= 1; offsetX++) {
        const pixelX = initialXInInput + offsetX;
        const pixelY = initialYInInput + offsetY;
        const pixelIndex = (pixelY * operationWidth + pixelX) * 4;
        const r = data[pixelIndex];
        const grayValue = r;
        const filterValue = filter[(offsetY + 1) * 3 + (offsetX + 1)];
        initialMatrixValues.push({ grayValue, filterValue });
        initialResultSum += (grayValue / 127.5 - 1) * filterValue;
    
        // Actualizar las posiciones de los rectángulos de resaltado
        highlightRects[index].x = pixelX;
        highlightRects[index].y = pixelY;
        index++;
      }
    }

    // Crear la visualización inicial de la matriz
    updateMatrixSVG(initialMatrixValues, initialResultSum, maxAbsValue);

    // Resaltar la matriz 3x3 en la imagen de entrada
    highlightPixels(inputHighlightCanvas, highlightRects);

    // Resaltar el píxel en la imagen de salida
    highlightSinglePixel(outputHighlightCanvas, initialX, initialY);

    // Manejar la interactividad del mouse en el canvas de entrada
    inputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMove(event, inputCanvas, inputHighlightCanvas, data, operationWidth, operationHeight, filter, maxAbsValue, svgMatrixContainer, 'input', highlightRects);
    });

    // Manejar la interactividad del mouse en el canvas de salida
    outputCanvas.addEventListener('mousemove', (event) => {
      handleMouseMove(event, outputCanvas, outputHighlightCanvas, data, operationWidth, operationHeight, filter, maxAbsValue, svgMatrixContainer, 'output', highlightRects);
    });
    
  };

  inputImage.onerror = () => {
    console.error('Failed to load image. Check the image URL or CORS settings.');
  };

  function handleMouseMove(event, canvas, highlightCanvas, data, width, height, filter, maxAbsValue, container, canvasType, highlightRects) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const x = Math.floor(mouseX / displayScale);
    const y = Math.floor(mouseY / displayScale);

    const croppedWidth = width - 2; // 62
    const croppedHeight = height - 2; // 62

    // Verificar los límites de la imagen
    if (x >= 0 && x < croppedWidth && y >= 0 && y < croppedHeight) {
      let xInInput = x + 1; // Compensar el recorte de un píxel
      let yInInput = y + 1;
      const matrixValues = [];
      let resultSum = 0;

      let index = 0;

      for (let offsetY = -1; offsetY <= 1; offsetY++) {
        for (let offsetX = -1; offsetX <= 1; offsetX++) {
          const pixelX = xInInput + offsetX;
          const pixelY = yInInput + offsetY;
          const pixelIndex = (pixelY * width + pixelX) * 4;
          const r = data[pixelIndex];
          const grayValue = r;
          const filterValue = filter[(offsetY + 1) * 3 + (offsetX + 1)];
          matrixValues.push({ grayValue, filterValue });
          resultSum += (grayValue / 127.5 - 1) * filterValue;
  
          // Actualizar las posiciones de los rectángulos de resaltado
          highlightRects[index].x = pixelX;
          highlightRects[index].y = pixelY;
          index++;
        }
      }

      // Actualizar la visualización de la matriz
      updateMatrixSVG(matrixValues, resultSum, maxAbsValue);

      // Resaltar los píxeles en los canvas correspondientes
      if (canvasType === 'input') {
        // Resaltar la matriz 3x3 en la imagen de entrada
        highlightPixels(highlightCanvas, highlightRects);
        // Resaltar el píxel correspondiente en la imagen de salida
        clearCanvas(outputHighlightCanvas);
        highlightSinglePixel(outputHighlightCanvas, x, y);
      } else if (canvasType === 'output') {
        // Resaltar el píxel en la imagen de salida
        highlightSinglePixel(highlightCanvas, x, y);
        // Resaltar la matriz 3x3 en la imagen de entrada
        clearCanvas(inputHighlightCanvas);
        highlightPixels(inputHighlightCanvas, highlightRects);
      }
    }
  }

  function updateMatrixSVG(matrixValues, resultSum, maxAbsValue) {
    for (let index = 0; index < 9; index++) {
      const { grayValue, filterValue } = matrixValues[index];
  
      // Actualizar el texto del píxel
      const valor1 = (grayValue / 127.5) - 1;
      pixelTexts[index].textContent = `${valor1.toFixed(2)}`;
  
      // Actualizar el color del rectángulo del píxel
      const color1 = `rgba(${grayValue}, ${grayValue}, ${grayValue}, 0.75)`;
      pixelTexts[index].previousSibling.setAttribute("fill", color1);
  
      // Actualizar el texto del filtro
      filterTexts[index].textContent = filterValue;
  
      // Actualizar el color del rectángulo del filtro
      let filterColor;
      if (filterValue < 0) {
        const intensity = Math.floor(255 * (1 + filterValue / 2)); // Escala de amarillo a blanco
        filterColor = `rgb(255, 255, ${intensity})`; // Amarillo a blanco
      } else {
        const intensity = Math.floor(255 * (1 - filterValue / 2)); // Escala de blanco a verde
        filterColor = `rgb(${intensity}, 255, ${intensity})`; // Blanco a verde
      }
      filterTexts[index].previousSibling.setAttribute("fill", filterColor);
    }
  
    // Actualizar el resultado
    const normalizedResultSum = resultSum / maxAbsValue;
    resultText.textContent = `${normalizedResultSum.toFixed(2)}`;
  
    // Actualizar el color del rectángulo del resultado
    let resultColor;
    if (normalizedResultSum < 0) {
      const t = normalizedResultSum + 1;
      const r = 255;
      const g = 255 * t;
      const b = 255 * t;
      resultColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
    } else {
      const t = 1 - normalizedResultSum;
      const r = 255 * t;
      const g = 255 * t;
      const b = 255;
      resultColor = `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
    }
    resultRect.setAttribute("fill", resultColor);
  }  

  function highlightPixels(highlightCanvas, highlightRects) {
    const ctx = highlightCanvas.getContext('2d');
    // Limpiar el canvas de resaltado
    ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

    ctx.save();
    ctx.scale(displayScale, displayScale);
    ctx.fillStyle  = 'rgba(64, 64, 64, 0.7)';
    ctx.lineWidth = 1 / displayScale;

    highlightRects.forEach(({ x, y }) => {
      ctx.strokeRect(x, y, 1, 1);
    });

    ctx.restore();
  }

  function highlightSinglePixel(highlightCanvas, x, y) {
    const ctx = highlightCanvas.getContext('2d');
    // Limpiar el canvas de resaltado
    ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);

    ctx.save();
    ctx.scale(displayScale, displayScale);
    ctx.fillStyle  = 'rgba(64, 64, 64, 0.7)';
    ctx.lineWidth = 1 / displayScale;
    ctx.strokeRect(x+0.25, y+0.25, 1, 1);
    ctx.restore();
  }

  function clearCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}