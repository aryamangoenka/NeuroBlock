import React, { useRef, useEffect, useState } from "react";

interface DrawingCanvasProps {
  onDrawingComplete: (imageData: Blob) => void;
  width?: number;
  height?: number;
  lineWidth?: number;
  clearCanvas?: boolean;
}

const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  onDrawingComplete,
  width = 280,
  height = 280,
  lineWidth = 20,
  clearCanvas = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [context, setContext] = useState<CanvasRenderingContext2D | null>(null);
  const [hasDrawing, setHasDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set up canvas
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "black";
    ctx.lineWidth = lineWidth;

    // Set white background
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, width, height);

    setContext(ctx);
  }, [width, height, lineWidth]);

  useEffect(() => {
    if (clearCanvas && context) {
      context.fillStyle = "white";
      context.fillRect(0, 0, width, height);
      setHasDrawing(false);
    }
  }, [clearCanvas, context, width, height]);

  const startDrawing = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas || !context) return;

    const rect = canvas.getBoundingClientRect();
    const x =
      "touches" in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y =
      "touches" in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    context.beginPath();
    context.moveTo(x, y);
  };

  const draw = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    if (!isDrawing || !context) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x =
      "touches" in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y =
      "touches" in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;

    context.lineTo(x, y);
    context.stroke();
    setHasDrawing(true);
  };

  const endDrawing = () => {
    setIsDrawing(false);
    if (!context) return;
    context.closePath();

    if (hasDrawing) {
      const canvas = canvasRef.current;
      if (!canvas) return;

      // Convert to grayscale and resize to 28x28
      const tempCanvas = document.createElement("canvas");
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) return;

      // First resize to 28x28
      tempCanvas.width = 28;
      tempCanvas.height = 28;
      tempCtx.fillStyle = "white";
      tempCtx.fillRect(0, 0, 28, 28);
      tempCtx.drawImage(canvas, 0, 0, 28, 28);

      // Convert to grayscale
      const imageData = tempCtx.getImageData(0, 0, 28, 28);
      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const avg = 255 - (data[i] + data[i + 1] + data[i + 2]) / 3;
        data[i] = avg; // R
        data[i + 1] = avg; // G
        data[i + 2] = avg; // B
      }
      tempCtx.putImageData(imageData, 0, 0);

      // Convert to blob and send
      tempCanvas.toBlob((blob) => {
        if (blob) {
          onDrawingComplete(blob);
        }
      }, "image/png");
    }
  };

  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling while drawing
    startDrawing(e);
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault(); // Prevent scrolling while drawing
    draw(e);
  };

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        border: "2px solid #e2e8f0",
        borderRadius: "8px",
        background: "white",
        cursor: "crosshair",
        touchAction: "none", // Prevent scrolling on touch devices
      }}
      onMouseDown={startDrawing}
      onMouseMove={draw}
      onMouseUp={endDrawing}
      onMouseOut={endDrawing}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={endDrawing}
    />
  );
};

export default DrawingCanvas;
