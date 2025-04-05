import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Node, Edge } from "reactflow";

interface NetworkVisualization3DProps {
  isActive: boolean;
  isTraining: boolean;
  trainingProgress?: {
    currentEpoch: number;
    totalEpochs: number;
    accuracy: number;
    loss: number;
    valAccuracy: number;
    valLoss: number;
  };
  nodes: Node[];
  edges: Edge[];
}

const NetworkVisualization3D: React.FC<NetworkVisualization3DProps> = ({
  isActive,
  isTraining,
  trainingProgress,
  nodes,
  edges,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive) return;

    console.log("3D Visualization initializing with React Three.js");
    console.log("Nodes:", nodes.length, "Edges:", edges.length);

    // Load Three.js and set up the visualization
    const loadThreeJS = async () => {
      try {
        if (!containerRef.current) {
          console.error("Container not found");
          return;
        }

        const container = containerRef.current;

        // Initialize scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f5ff);

        // Initialize camera
        const camera = new THREE.PerspectiveCamera(
          60,
          container.clientWidth / container.clientHeight,
          0.1,
          1000
        );
        camera.position.set(300, 150, 500);
        camera.lookAt(0, 0, 0);

        // Initialize renderer
        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          alpha: true,
        });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Initialize controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.1;
        controls.rotateSpeed = 0.7;

        // Add lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(200, 400, 300);
        scene.add(directionalLight);

        // Material definitions for different layer types
        const layerMaterials = {
          input: new THREE.MeshPhongMaterial({ color: 0x4285f4 }),
          convolution: new THREE.MeshPhongMaterial({ color: 0xf59e0b }),
          maxpooling: new THREE.MeshPhongMaterial({ color: 0x10b981 }),
          globalaveragepool: new THREE.MeshPhongMaterial({ color: 0x6366f1 }),
          flatten: new THREE.MeshPhongMaterial({ color: 0xf97316 }),
          dense: new THREE.MeshPhongMaterial({ color: 0xec4899 }),
          dropout: new THREE.MeshPhongMaterial({
            color: 0x8b5cf6,
            transparent: true,
            opacity: 0.8,
          }),
          batchnormalization: new THREE.MeshPhongMaterial({ color: 0x14b8a6 }),
          resnetblock: new THREE.MeshPhongMaterial({ color: 0xf43f5e }),
          output: new THREE.MeshPhongMaterial({ color: 0xef4444 }),
          connection: new THREE.LineBasicMaterial({
            color: 0xcbd5e1,
            opacity: 0.6,
            transparent: true,
          }),
        };

        // Layout algorithm parameters
        const LAYER_DEPTH = 100; // Z-axis spacing between layers
        const LAYER_HEIGHT = 80; // Y-axis baseline for layers
        const NODE_SPACING = 25; // Spacing between nodes in complex layers
        const NODE_RADIUS = 10; // Base radius for nodes

        // Create a mapping of node IDs to their positions in 3D space
        const nodePositions = new Map<
          string,
          { x: number; y: number; z: number }
        >();
        const layerObjects = new Map<string, THREE.Object3D>();

        // Extract node and edge data from your model
        const nodesList = [...nodes];
        const edgesList = [...edges];

        // Sort nodes by their visual layer in the network
        // This is a simplified topological sort
        const layers: Node[][] = [];
        const visited = new Set<string>();

        // First, identify input nodes (no incoming edges)
        const incomingEdges = new Map<string, string[]>();
        edgesList.forEach((edge) => {
          if (!incomingEdges.has(edge.target)) {
            incomingEdges.set(edge.target, []);
          }
          incomingEdges.get(edge.target)!.push(edge.source);
        });

        // Find all input nodes (no incoming edges)
        const inputNodes = nodesList.filter(
          (node) =>
            !incomingEdges.has(node.id) ||
            incomingEdges.get(node.id)!.length === 0
        );

        // Add input nodes to first layer
        layers.push(inputNodes);
        inputNodes.forEach((node) => visited.add(node.id));

        // Build subsequent layers
        while (visited.size < nodesList.length) {
          const currentLayer: Node[] = [];

          nodesList.forEach((node) => {
            if (visited.has(node.id)) return;

            // Check if all prerequisites are visited
            const prerequisites = incomingEdges.get(node.id) || [];
            const allPrerequisitesVisited = prerequisites.every((preId) =>
              visited.has(preId)
            );

            if (allPrerequisitesVisited) {
              currentLayer.push(node);
              visited.add(node.id);
            }
          });

          if (currentLayer.length === 0 && visited.size < nodesList.length) {
            // Handle potential cycles by adding remaining nodes
            const remainingNodes = nodesList.filter(
              (node) => !visited.has(node.id)
            );
            layers.push(remainingNodes);
            remainingNodes.forEach((node) => visited.add(node.id));
            break;
          }

          layers.push(currentLayer);
        }

        // Create layer geometries based on layer type
        layers.forEach((layerNodes, layerIndex) => {
          const layerZ = -layerIndex * LAYER_DEPTH;

          // Position nodes in this layer
          layerNodes.forEach((node, nodeIndex) => {
            let nodeGeometry: THREE.BufferGeometry;
            let nodeMesh: THREE.Object3D;
            const nodeType = (node.type as string) || "default";
            const material =
              layerMaterials[nodeType as keyof typeof layerMaterials] ||
              layerMaterials.dense;

            // Calculate position for node in layer
            let nodeX =
              (nodeIndex - (layerNodes.length - 1) / 2) * NODE_SPACING * 2;
            let nodeY = LAYER_HEIGHT;
            let nodeZ = layerZ;

            // Create different geometries based on layer type
            switch (nodeType) {
              case "input":
                nodeGeometry = new THREE.SphereGeometry(
                  NODE_RADIUS * 1.2,
                  16,
                  16
                );
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              case "convolution":
                // For conv layers, use cube with size based on filters
                const filters = node.data?.filters || 32;
                const size = Math.max(10, Math.log2(filters) * 3);
                nodeGeometry = new THREE.BoxGeometry(size, size, size);
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              case "dense":
                // For dense layers, use sphere with size based on neurons
                const neurons = node.data?.neurons || 64;
                const sphereSize = Math.max(8, Math.log2(neurons) * 2);
                nodeGeometry = new THREE.SphereGeometry(sphereSize, 16, 16);
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              case "maxpooling":
              case "globalaveragepool":
                // For pooling layers, use a flattened cube
                nodeGeometry = new THREE.BoxGeometry(
                  NODE_RADIUS * 2,
                  NODE_RADIUS * 0.5,
                  NODE_RADIUS * 2
                );
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              case "flatten":
                // For flatten layers, use a plane
                nodeGeometry = new THREE.PlaneGeometry(
                  NODE_RADIUS * 2,
                  NODE_RADIUS * 2
                );
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                nodeMesh.rotation.x = -Math.PI / 2; // Rotate to be horizontal
                break;

              case "output":
                // For output layer, use a star-like geometry
                nodeGeometry = new THREE.SphereGeometry(
                  NODE_RADIUS * 1.5,
                  16,
                  16
                );
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              case "resnetblock":
                // For ResNet blocks, use a special block representation
                nodeGeometry = new THREE.BoxGeometry(
                  NODE_RADIUS * 2,
                  NODE_RADIUS * 2,
                  NODE_RADIUS * 2
                );
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;

              default:
                // Default to a sphere for other layer types
                nodeGeometry = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
                nodeMesh = new THREE.Mesh(nodeGeometry, material);
                break;
            }

            // Position the mesh
            nodeMesh.position.set(nodeX, nodeY, nodeZ);
            scene.add(nodeMesh);

            // Store position for edge creation
            nodePositions.set(node.id, { x: nodeX, y: nodeY, z: nodeZ });
            layerObjects.set(node.id, nodeMesh);
          });
        });

        // Create edges between nodes
        edgesList.forEach((edge) => {
          const sourcePos = nodePositions.get(edge.source);
          const targetPos = nodePositions.get(edge.target);

          if (!sourcePos || !targetPos) return;

          // Create line for the edge
          const points = [
            new THREE.Vector3(sourcePos.x, sourcePos.y, sourcePos.z),
            new THREE.Vector3(targetPos.x, targetPos.y, targetPos.z),
          ];

          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const line = new THREE.Line(geometry, layerMaterials.connection);
          scene.add(line);
        });

        // Add a grid to help with orientation
        const gridHelper = new THREE.GridHelper(500, 20, 0x444444, 0x222222);
        gridHelper.position.y = -50;
        scene.add(gridHelper);

        // Animation loop
        const animate = () => {
          requestAnimationFrame(animate);
          controls.update();
          renderer.render(scene, camera);
        };
        animate();

        // Handle window resize
        const handleResize = () => {
          if (!container) return;
          camera.aspect = container.clientWidth / container.clientHeight;
          camera.updateProjectionMatrix();
          renderer.setSize(container.clientWidth, container.clientHeight);
        };
        window.addEventListener("resize", handleResize);

        // Cleanup function
        return () => {
          console.log("Cleaning up Three.js visualization");
          window.removeEventListener("resize", handleResize);
          container.removeChild(renderer.domElement);

          // Dispose of geometries and materials
          scene.traverse((object: THREE.Object3D) => {
            if (object instanceof THREE.Mesh) {
              object.geometry.dispose();
              if (
                object.material &&
                typeof object.material.dispose === "function"
              ) {
                object.material.dispose();
              }
            }
          });
        };
      } catch (error) {
        console.error("Error setting up Three.js visualization:", error);
      }
    };

    // Initialize Three.js
    const cleanupPromise = loadThreeJS();

    // Return cleanup function
    return () => {
      console.log("3D Visualization cleanup initiated");
      cleanupPromise
        .then((cleanup) => {
          if (cleanup && typeof cleanup === "function") {
            cleanup();
          }
        })
        .catch((err) => {
          console.error("Cleanup error:", err);
        });
    };
  }, [isActive, nodes, edges]);

  return (
    <div
      ref={containerRef}
      id="network-3d-container"
      className="network-3d-container"
      style={{
        width: "100%",
        height: "100%",
        display: isActive ? "block" : "none",
      }}
    />
  );
};

export default NetworkVisualization3D;
