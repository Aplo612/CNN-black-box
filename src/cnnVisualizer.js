import * as d3 from 'd3';
import * as tf from '@tensorflow/tfjs';

const filters = [
  // Filtro de detección de bordes (Sobel horizontal)
  [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1,
  ],
  // Filtro de detección de bordes (Sobel vertical)
  [
    -1,  0,  1,
    -2,  0,  2,
    -1,  0,  1,
  ],
  // Filtro de desenfoque (blur)
  [
    0.11, 0.11, 0.11,
    0.11, 0.11, 0.11,
    0.11, 0.11, 0.11,
  ],
  // Filtro de realce
  [
    0, -1,  0,
    -1,  5, -1,
    0, -1,  0,
  ],
  // Filtro aleatorio 1
  [
    0.3, -0.21, 0.07,
    -0.19, 0.1, -0.01,
    -0.04, -0.02, -0.08,
  ],
  // Filtro aleatorio 2
  [
    0, -2, -2,
    2, 0, -2,
    2, 2, 2,
  ],
  // Filtro aleatorio 3
  [
    0.07, -0.19, 0.1,
    -0.02, -0.08, 0.3,
    -0.21, 0.1, -0.01,
  ],
  // Filtro aleatorio 4
  [
    -1,  2, -1,
     0,  0,  0,
     1, -2,  1,
  ],
  // Filtro aleatorio 5
  [
    -0.5,  0,  0.5,
    -1.5,  0,  1.5,
    -0.5,  0,  0.5,
  ],
  // Filtro aleatorio 6
  [
    0.2,  0.1,  0.2,
    0.1, -0.8,  0.1,
    0.2,  0.1,  0.2,
  ],
  // Filtro aleatorio 7
  [
    0.5, -0.3, -0.2,
    -0.4,  1.0, -0.4,
    -0.2, -0.3,  0.5,
  ],
  // Filtro aleatorio 8
  [
    -0.1, -0.2, -0.1,
    0.0,   0.5,  0.0,
    -0.1, -0.2, -0.1,
  ],
  // Filtro aleatorio 9
  [
    0.4,  0.2,  0.0,
    0.2, -0.5, -0.2,
    0.0, -0.2, -0.4,
  ],
  // Filtro aleatorio 10
  [
    -0.3,  0.1, -0.3,
     0.1,  0.8,  0.1,
    -0.3,  0.1, -0.3,
  ],
  // Filtro aleatorio 11
  [
    -1,  0.5,  1,
    0.5, -2,  0.5,
    1,  0.5, -1,
  ],
  // Filtro aleatorio 12
  [
    0.6, 0.1, 0.6,
    0.1, -0.9, 0.1,
    0.6, 0.1, 0.6,
  ],
  // Filtro aleatorio 13
  [
    -0.7, -0.3, -0.7,
    -0.3,  1.0, -0.3,
    -0.7, -0.3, -0.7,
  ],
  // Filtro aleatorio 14
  [
    0.1, -0.2,  0.1,
    -0.2,  0.6, -0.2,
    0.1, -0.2,  0.1,
  ],
  // Filtro aleatorio 15
  [
    -0.5, -0.2, -0.5,
    -0.2,  0.9, -0.2,
    -0.5, -0.2, -0.5,
  ],
];

export async function createCNNVisualizer({ container, imageSrc }) {    
    // Verificar si imageSrc está definido
    if (!imageSrc) {
      console.error('El parámetro imageSrc es undefined.');
      return;
    }
    // Obtener el elemento del contenedor
    const containerElement = document.getElementById(container);
    if (!containerElement) {
      console.error(`No se encontró un elemento con id '${container}'.`);
      return;
    }
    
    // Dimensiones del SVG
    const width = 1900;
    const height = 800;

    // Crear el SVG dentro del contenedor
    const svg = d3.select(containerElement)
      .append('svg')
      .attr('width', width)
      .attr('height', height);
  
    // Cargar y procesar la imagen
    let imageTensor;
    try {
      imageTensor = await loadAndProcessImage(imageSrc);
    } catch (error) {
      console.error('Error al cargar la imagen:', error);
      return;
    }

    // Definir las capas de la CNN
    const layers = [
      { type: 'input', label: 'Entrada', subtitle: '[64x64]', nodes: 3 },
      { type: 'conv', label: 'Convolución 1', subtitle: '[62x62]', nodes: 10 },
      { type: 'relu', label: 'ReLU 1', subtitle: '[62x62]', nodes: 10 },
      { type: 'conv', label: 'Convolución 2', subtitle: '[60x60]', nodes: 10 },
      { type: 'relu', label: 'ReLU 2', subtitle: '[60x60]', nodes: 10 },
      { type: 'pool', label: 'MaxPooling 1', subtitle: '[30x30]', nodes: 10 },
      { type: 'conv', label: 'Convolución 3', subtitle: '[28x28]', nodes: 10 },
      { type: 'relu', label: 'ReLU 3', subtitle: '[28x28]', nodes: 10 },
      { type: 'conv', label: 'Convolución 4', subtitle: '[26x26]', nodes: 10 },
      { type: 'relu', label: 'ReLU 4', subtitle: '[26x26]', nodes: 10 },
      { type: 'pool', label: 'MaxPooling 2', subtitle: '[13x13]', nodes: 10 },
      { type: 'output', label: 'Salida', subtitle: '', nodes: 1 },
    ];

    // Definir los grupos de capas
    const groups = [
      [0], // Grupo 0: Entrada
      [1, 2], // Grupo 1: Convolución 1, ReLU 1
      [3, 4, 5], // Grupo 2: Convolución 2, ReLU 2, MaxPooling 1
      [6, 7], // Grupo 3: Convolución 3, ReLU 3
      [8, 9, 10], // Grupo 4: Convolución 4, ReLU 4, MaxPooling 2
      [11], // Grupo 5: Salida
    ];

    // Procesar la imagen a través de las capas y obtener las salidas
    // const layerOutputs = await getLayerOutputs(imageTensor, layers);
  
    // Crear los nodos y enlaces para el gráfico
    createGraph(svg, layers, groups, width, height, imageTensor, imageSrc);
}

async function loadAndProcessImage(imageSrc) {
  const img = new Image();
  img.crossOrigin = 'Anonymous';
  img.src = imageSrc;

  await new Promise((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = (err) => reject(err);
  });

  // Convertir la imagen en un tensor
  let tensor = tf.browser.fromPixels(img);
  
  // Redimensionar la imagen a 64x64
  tensor = tf.image.resizeBilinear(tensor, [64, 64]);

  // Normalizar los valores de los píxeles a [0, 1]
  tensor = tensor.toFloat().div(tf.scalar(255));

  return tensor;
}


function createGraph(svg, layers, groups, width, height, imageTensor, imagenc) {
  // Definir espaciados
  const withinGroupSpacing = 1; // Unidad de espaciado dentro del grupo
  const betweenGroupSpacing = 3; // Unidad de espaciado entre grupos
  const margin = 80; // Margen en píxeles para los bordes izquierdo y derecho

  // Calcular las posiciones horizontales
  let totalUnits = 0;
  const xPositions = [];

  const { rChannel, gChannel, bChannel } = getRGBChannels(imageTensor);

  const grayscaleTensor = imageTensor.mean(2).expandDims(2);
  let previousLayerOutputs = {};
  previousLayerOutputs[0] = grayscaleTensor;
  for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
    const group = groups[groupIndex];
    for (let i = 0; i < group.length; i++) {
      const layerIndex = group[i];
      xPositions[layerIndex] = totalUnits;
      if (i < group.length - 1) {
        // Añadimos espaciado dentro del grupo
        totalUnits += withinGroupSpacing;
      }
    }
    if (groupIndex < groups.length - 1) {
      // Añadimos espaciado entre grupos
      totalUnits += betweenGroupSpacing;
    }
  }

  // Ajustar el ancho disponible restando los márgenes
  const availableWidth = width - margin * 2;

  // Escalar las posiciones X al ancho disponible
  const scaleX = availableWidth / totalUnits;

  const layerPositions = layers.map((layer, i) => ({
    x: margin + xPositions[i] * scaleX,
    layer,
    layerIndex: i,
  }));

  // Generar los nodos con sus posiciones
  const nodes = [];
  const links = [];

  // Mapa para acceder rápidamente a los nodos de cada capa
  const layerNodesMap = {};

  layerPositions.forEach(({ x, layer, layerIndex }) => {
    const numNodes = layer.nodes;

    // Calcular el espaciado vertical para distribuir los nodos en la altura total
    const nodeSpacing = height / (numNodes + 1); // Espacio vertical entre nodos
    const startY = nodeSpacing; // Iniciar desde nodeSpacing para dejar margen superior

    const layerNodes = [];

    for (let i = 0; i < numNodes; i++) {
      const node = {
        id: `layer${layerIndex}_node${i}`,
        layerIndex,
        indexInLayer: i,
        x,
        y: startY + i * nodeSpacing,
        label: layer.label,
        type: layer.type,
        imageTensor: null,
      };
      nodes.push(node);
      layerNodes.push(node);
    }
    // Procesar la capa
    if (layer.type === 'input') {
      // Asignar los canales R, G y B a los nodos de entrada
      const channels = [rChannel, gChannel, bChannel];

      for (let i = 0; i < layerNodes.length; i++) {
        if (i < channels.length) {
          layerNodes[i].imageTensor = channels[i];
        }
      }

      // No asignamos nada a previousLayerOutputs aquí

    } else if (layer.type === 'conv' && (layerIndex === 1)) {
      const inputTensor = previousLayerOutputs[layerIndex - 1];
      const layerFilters = filters.slice(0, numNodes); // Obtener los filtros necesarios
      const convolutedTensors = [];
      
      for (let i = 0; i < numNodes; i++) {
        
        const filter = filters[i % filters.length]; // Reutilizar filtros si es necesario
        const convoluted = applyFilter(inputTensor, filter);
        convolutedTensors.push(convoluted);
        // Asignar la imagen al nodo
        layerNodes[i].imageTensor = convoluted;
      }
      //console.log(inputTensor);
      // Guardar la salida de la capa para usarla en la siguiente
      previousLayerOutputs[layerIndex] = tf.stack(convolutedTensors, -1);
      // Liberar tensor de entrada si ya no es necesario
      if (previousLayerOutputs[layerIndex - 1] && layerIndex - 1 !== -1) {
        previousLayerOutputs[layerIndex - 1].dispose();
        previousLayerOutputs[layerIndex - 1] = null;
      }

    } else if (layer.type === 'relu') {
      const inputTensor = previousLayerOutputs[layerIndex - 1];
       //console.log(inputTensor);
      // Aplicar la función ReLU
      const activatedTensor = inputTensor.relu();
    
      // Asignar las salidas a los nodos correspondientes
      const channels = activatedTensor.unstack(-1);
    
      for (let i = 0; i < layerNodes.length; i++) {
        if (i < channels.length) {
          layerNodes[i].imageTensor = channels[i];
        }
      }
    
      // Actualizar previousLayerOutputs
      previousLayerOutputs[layerIndex] = activatedTensor;
    
      // Liberar el tensor de entrada si ya no es necesario
      if (previousLayerOutputs[layerIndex - 1]) {
        previousLayerOutputs[layerIndex - 1].dispose();
        previousLayerOutputs[layerIndex - 1] = null;
      }
    }else if (layer.type === 'conv' && (layerIndex === 3 || layerIndex === 6 || layerIndex === 8)) { // Suponiendo que layerIndex 3 es la segunda convolución
      const inputTensor = previousLayerOutputs[layerIndex - 1]; // Salida de ReLU 1
      const numInputChannels = inputTensor.shape[2]; // Número de canales de entrada (10)
      const numFilters = numNodes; // Número de filtros es igual al número de nodos en esta capa
      const convolutedTensors = [];
    
      for (let i = 0; i < numFilters; i++) {
        // Obtener el filtro correspondiente
        const filter = filters[(i+3) % filters.length]; // Reutilizar filtros si es necesario
    
        // Crear un filtro que tenga en cuenta los canales de entrada
        const filterArray = [];
        for (let c = 0; c < numInputChannels; c++) {
          filterArray.push(...filter);
        }

        const filterTensor = tf.tensor4d(filterArray, [3, 3, 10, 1]);
        // Expandir el tensor de entrada si es necesario
        const inputTensor4D = inputTensor.expandDims(0); // [height, width, channels] -> [1, height, width, channels]
    
        // Aplicar la convolución
        const convoluted = inputTensor4D.conv2d(filterTensor, [1, 1], 'same').squeeze();
    
        // Asignar la imagen al nodo
        layerNodes[i].imageTensor = convoluted;
        convolutedTensors.push(convoluted);
    
        // Liberar tensores
        filterTensor.dispose();
        inputTensor4D.dispose();
      }
    
      // Combinar los tensores de salida para formar la salida de la capa
      const outputTensor = tf.stack(convolutedTensors, -1); // Forma [height, width, numFilters]
    
      previousLayerOutputs[layerIndex] = outputTensor;
    
      // Liberar tensor de entrada si ya no es necesario
      if (previousLayerOutputs[layerIndex - 1]) {
        previousLayerOutputs[layerIndex - 1].dispose();
        previousLayerOutputs[layerIndex - 1] = null;
      }
    }else if (layer.type === 'pool') {
      const inputTensor = previousLayerOutputs[layerIndex - 1];
      
      // Expandir dimensiones si es necesario
      let inputTensor4D = inputTensor;
      if (inputTensor.rank === 3) {
        inputTensor4D = inputTensor.expandDims(0); // [height, width, channels] -> [1, height, width, channels]
      }
      
      // Aplicar MaxPooling
      const pooledTensor = inputTensor4D.maxPool([2, 2], [2, 2], 'same').squeeze();
      
      // Asignar las salidas a los nodos correspondientes
      const channels = pooledTensor.unstack(-1);
      
      for (let i = 0; i < layerNodes.length; i++) {
        if (i < channels.length) {
          layerNodes[i].imageTensor = channels[i];
        }
      }
      
      // Actualizar previousLayerOutputs
      previousLayerOutputs[layerIndex] = pooledTensor;
      
      // Liberar tensores
      if (inputTensor4D !== inputTensor) {
        inputTensor4D.dispose();
      }
      if (previousLayerOutputs[layerIndex - 1]) {
        previousLayerOutputs[layerIndex - 1].dispose();
        previousLayerOutputs[layerIndex - 1] = null;
      }
      
      // Liberar los canales si ya no se necesitan (después de crear los patrones)
      // channels.forEach(channel => channel.dispose());
    }else {
      // Procesar otras capas según sea necesario
    }

    layerNodesMap[layerIndex] = layerNodes;

    // Crear enlaces con la capa anterior
    if (layerIndex > 0) {
      const prevLayerNodes = layerNodesMap[layerIndex - 1];
      const prevLayerType = layers[layerIndex - 1].type;
      const currentLayerType = layer.type;

      // Determinar el tipo de conexión
      const connectionType = getConnectionType(prevLayerType, currentLayerType);

      if (connectionType === 'one-to-one') {
        // Aseguramos que ambas capas tengan el mismo número de nodos
        const minNodes = Math.min(prevLayerNodes.length, layerNodes.length);
        for (let i = 0; i < minNodes; i++) {
          links.push({
            source: prevLayerNodes[i],
            target: layerNodes[i],
            connectionType,
          });
        }
      } else if (connectionType === 'all-to-all') {
        layerNodes.forEach(targetNode => {
          prevLayerNodes.forEach(sourceNode => {
            links.push({
              source: sourceNode,
              target: targetNode,
              connectionType,
            });
          });
        });
      }
    }
  });

  const layerLabels = svg.selectAll('.layer-label')
    .data(layerPositions)
    .enter()
    .append('g')
    .attr('class', 'layer-label')
    .attr('transform', d => `translate(${d.x}, 20)`); // Posicionar el grupo en x y en y=20

  // Agregar el título
  layerLabels.append('text')
    .attr('text-anchor', 'middle')
    .text(d => d.layer.label)
    .style('font-size', '12px')
    .style('font-weight', 'bold');

  // Agregar el subtítulo
  layerLabels.append('text')
    .attr('text-anchor', 'middle')
    .attr('dy', '1.2em') // Desplazamiento vertical para colocar el subtítulo debajo del título
    .text(d => d.layer.subtitle)
    .style('font-size', '10px') // Tamaño de fuente más pequeño para el subtítulo
    .style('font-weight', 'normal');

  // Dibujar los enlaces
  const linkGenerator = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  let defs = svg.select('defs');
  if (defs.empty()) {
    defs = svg.append('defs');
  }

  const nodeSize = 40; // Tamaño duplicado de los nodos (cuadrados)

  // Función para crear un patrón a partir de un tensor de imagen
  function createPatternFromTensor(node) {
    const tensor = node.imageTensor;
    if (!tensor) return;

    const normalizedTensor = tensor.sub(tensor.min()).div(tensor.max().sub(tensor.min())).clipByValue(0, 1);

    // Convertir el tensor a imagen y crear un data URL
    tf.browser.toPixels(normalizedTensor.squeeze()).then((pixels) => {
      // Crear un canvas para dibujar la imagen
      const canvas = document.createElement('canvas');
      canvas.width = normalizedTensor.shape[1];
      canvas.height = normalizedTensor.shape[0];
      const ctx = canvas.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      const imageData = ctx.createImageData(canvas.width, canvas.height);
      imageData.data.set(pixels);
      ctx.putImageData(imageData, 0, 0);

      // Crear un data URL de la imagen
      const dataURL = canvas.toDataURL();

      // Crear el patrón
      defs.append('pattern')
        .attr('id', `pattern-${node.id}`)
        .attr('patternUnits', 'objectBoundingBox')
        .attr('width', 1)
        .attr('height', 1)
        .append('image')
        .attr('xlink:href', dataURL)
        .attr('width', nodeSize)
        .attr('height', nodeSize)
        .attr('preserveAspectRatio', 'none');
    });
  }

  // Crear patrones para los nodos con imageTensor
  nodes.forEach(node => {
    if (node.imageTensor) {
      createPatternFromTensor(node);
    }
  });

  // Liberar los tensores de los canales RGB
  [rChannel, gChannel, bChannel].forEach(channel => channel.dispose());

  // Liberar la imagen en escala de grises
  grayscaleTensor.dispose();

  svg.selectAll('path')
    .data(links)
    .enter()
    .append('path')
    .attr('d', d => linkGenerator({
      source: { x: d.source.x + nodeSize / 2, y: d.source.y },
      target: { x: d.target.x - nodeSize / 2, y: d.target.y },
    }))
    .attr('stroke', '#e0e0e0')
    .attr('stroke-width', 1)
    .attr('fill', 'none');

  // Nodos que no son de salida
  const nonOutputNodes = nodes.filter(d => d.type !== 'output');

  // Nodo de salida
  const outputNode = nodes.find(d => d.type === 'output');

  svg.selectAll('rect')
    .data(nonOutputNodes)
    .enter()
    .append('rect')
    .attr('x', d => d.x - nodeSize / 2)
    .attr('y', d => d.y - nodeSize / 2)
    .attr('width', nodeSize)
    .attr('height', nodeSize)
    .attr('fill', d => d.imageTensor ? `url(#pattern-${d.id})` : '#69b3a2')
    // .attr('stroke', '#333')
    // .attr('stroke-width', 1)
    .on('mouseover', handleMouseOver)
    .on('mouseout', handleMouseOut)
    .on('click', function(event, d) {
      if (d.type === 'conv' || d.type === 'relu' || d.type === 'pool') {
        handleClick(event, d);
      }
    });

  if (outputNode) {
    svg.append('text')
      .attr('class', 'output-label')
      .attr('id', 'output-text')
      .attr('x', outputNode.x + 15)
      .attr('y', outputNode.y)
      .attr('dy', '.35em') // Alinear verticalmente
      .attr('text-anchor', 'middle')
      .text('Predicción')
      .style('font-size', '14px')
      .style('font-weight', 'normal')
      .style('fill', '#444444')
      .on('mouseover', () => highlightNodes(outputNode))
      .on('mouseout', () => restoreNodes(outputNode));
  }
  // Agregar etiquetas de capa
  svg.selectAll('.layer-label')
    .data(layerPositions)
    .enter()
    .append('text')
    .attr('class', 'layer-label')
    .attr('x', d => d.x)
    .attr('y', 20)
    .attr('text-anchor', 'middle')
    .text(d => d.layer.label)
    .style('font-size', '12px')
    .style('font-weight', 'bold');

  
  // Funciones para manejar eventos de mouse
  function handleMouseOver(event, d) {
    // Destacar las conexiones que tienen como fuente el nodo actual
    svg.selectAll('path')
      .filter(link => link.target === d)
      .transition()
      .duration(200)
      .attr('stroke', '#aaa') // Color para destacar
      .attr('stroke-width', 2);

    // Obtener los nodos conectados
    const connectedNodeIds = links
      .filter(link => link.target === d)
      .map(link => link.source.id);

    // Destacar el nodo actual y los nodos conectados
    svg.selectAll('rect')
      .filter(node => node.id === d.id || connectedNodeIds.includes(node.id))
      .transition()
      .duration(200)
      .attr('stroke', '#aaa')
      .attr('stroke-width', 2);
    
  }

  function handleMouseOut(event, d) {
    // Restaurar las conexiones
    svg.selectAll('path')
      .filter(link => link.target === d)
      .transition()
      .duration(200)
      .attr('stroke', '#e0e0e0') // Color original
      .attr('stroke-width', 1);

    // Obtener los nodos conectados
    const connectedNodeIds = links
      .filter(link => link.target === d)
      .map(link => link.source.id);

    // Restaurar los nodos
    svg.selectAll('rect')
      .filter(node => node.id === d.id || connectedNodeIds.includes(node.id))
      .transition()
      .duration(200)
      .attr('stroke', 'none')
      .attr('stroke-width', 0);
  }

  // Función para destacar nodos conectados y las conexiones
  function highlightNodes(node) {
    svg.selectAll('path')
      .filter(link => link.target === node)
      .transition()
      .duration(200)
      .attr('stroke', '#aaa') // Color para destacar
      .attr('stroke-width', 2);

    // Obtener los nodos conectados
    const connectedNodeIds = links
      .filter(link => link.target === node)
      .map(link => link.source.id);

    // Destacar el nodo actual y los nodos conectados
    svg.selectAll('rect')
      .filter(n => n.id === node.id || connectedNodeIds.includes(n.id))
      .transition()
      .duration(200)
      .attr('stroke', '#aaa')
      .attr('stroke-width', 2);

    // Destacar el texto de predicción
    svg.select('#output-text')
      .transition()
      .duration(200)
      .style('font-weight', 'bold');
  }

  // Función para restaurar nodos conectados y las conexiones
  function restoreNodes(node) {
    svg.selectAll('path')
      .filter(link => link.target === node)
      .transition()
      .duration(200)
      .attr('stroke', '#e0e0e0') // Color original
      .attr('stroke-width', 1);

    // Obtener los nodos conectados
    const connectedNodeIds = links
      .filter(link => link.target === node)
      .map(link => link.source.id);

    // Restaurar el nodo actual y los nodos conectados
    svg.selectAll('rect')
      .filter(n => n.id === node.id || connectedNodeIds.includes(n.id))
      .transition()
      .duration(200)
      .attr('stroke', 'none')
      .attr('stroke-width', 0);

    // Restaurar el texto de predicción
    svg.select('#output-text')
      .transition()
      .duration(200)
      .style('font-weight', 'normal');
  }


  function handleClick(event, d) {
    // Aquí puedes crear y mostrar el modal
    showModal(d);
  }
  
  function showModal(nodeData) {
    // Obtener el elemento del modal
    const modal = document.getElementById('modal');
    const modalBody = document.getElementById('modal-body');
    const closeModalBtn = document.getElementById('close-modal');
  
    // Agregar contenido al modal
    if(nodeData.type === 'conv') {
      modalBody.innerHTML = `
        <h2>Capa: ${nodeData.label}</h2>
        <div id="convolution-container"></div>
      `;

      // Mostrar el modal
      modal.style.display = 'block';
    
      // Agregar evento para cerrar el modal
      closeModalBtn.onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
    
      // Cerrar el modal cuando se hace clic en la superposición
      modal.querySelector('.modal-overlay').onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
      
      CNNBlackBox.startConvolution({
        container: 'convolution-container',
        imageSrc: imagenc,
        filter: [0.3, -0.21, 0.07, -0.19, 0.1, -0.01, -0.04, -0.02, -0.08]
      });
    } else if (nodeData.type === 'relu') { 
      modalBody.innerHTML = `
        <h2>Capa: ${nodeData.label}</h2>
        <div id="relu-container"></div>
      `;

      // Mostrar el modal
      modal.style.display = 'block';
    
      // Agregar evento para cerrar el modal
      closeModalBtn.onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
    
      // Cerrar el modal cuando se hace clic en la superposición
      modal.querySelector('.modal-overlay').onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
      
      CNNBlackBox.startReLU({
        container: 'relu-container',
        imageSrc: imagenc,
        filter: [0.3, -0.21, 0.07, -0.19, 0.1, -0.01, -0.04, -0.02, -0.08]
      });
    } else if (nodeData.type === 'pool') { 
      modalBody.innerHTML = `
        <h2>Capa: ${nodeData.label}</h2>
        <div id="max-pooling-container"></div>
      `;

      // Mostrar el modal
      modal.style.display = 'block';
    
      // Agregar evento para cerrar el modal
      closeModalBtn.onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
    
      // Cerrar el modal cuando se hace clic en la superposición
      modal.querySelector('.modal-overlay').onclick = function() {
        modal.style.display = 'none';
        modalBody.innerHTML = '';
      };
      
      CNNBlackBox.startMaxPooling({
        container: 'max-pooling-container',
        imageSrc: imagenc,
        filter: [0.3, -0.21, 0.07, -0.19, 0.1, -0.01, -0.04, -0.02, -0.08]
      });
    }
  
    
  }

}

function applyFilter(tensor, filter) {
  let inputTensor = tensor; 
  if (tensor.rankType === 2) {
    // Si el tensor tiene forma [height, width], añade canales
    inputTensor = tensor.expandDims(-1); // [height, width] -> [height, width, 1]
  }
  if (tensor.rankType === 3) {
    // Añade la dimensión de batch
    inputTensor = inputTensor.expandDims(0); // [height, width, channels] -> [1, height, width, channels]
  }

  // Expande las dimensiones del filtro para que tenga forma [filterHeight, filterWidth, inChannels, outChannels]
  const filterTensor = tf.tensor4d(filter, [3, 3, 1, 1]);

  // Aplica la convolución
  const convoluted = inputTensor.conv2d(filterTensor, [1, 1], 'same').squeeze();

  filterTensor.dispose();

  return convoluted;
}


// Función para determinar el tipo de conexión entre capas
function getConnectionType(prevLayerType, currentLayerType) {
  if (
    (prevLayerType === 'conv' && currentLayerType === 'relu') ||
    (prevLayerType === 'relu' && currentLayerType === 'pool')
  ) {
    return 'one-to-one';
  } else {
    return 'all-to-all';
  }
}

function getRGBChannels(tensor) {
  // tensor tiene forma [64, 64, 3]
  const [height, width, channels] = tensor.shape;

  // Separar los canales R, G y B
  const rChannel = tensor.slice([0, 0, 0], [height, width, 1]);
  const gChannel = tensor.slice([0, 0, 1], [height, width, 1]);
  const bChannel = tensor.slice([0, 0, 2], [height, width, 1]);

  return { rChannel, gChannel, bChannel };
}
