import React from "react";
import "../styles/components/TutorialLayerCard.scss";

interface LayerCardProps {
  name: string;
  color: string;
  type?: string;
  description?: string;
  properties?: { name: string; value: string }[];
}

const TutorialLayerCard: React.FC<LayerCardProps> = ({
  name,
  color,
  description,
  properties = [],
}) => {
  return (
    <div className="tutorial-layer-card">
      <div
        className="layer-visual"
        style={{
          backgroundColor: color,
          borderColor: adjustColorBrightness(color, -15),
        }}
      >
        <div className="layer-connector left">
          <div className="connector-dot"></div>
        </div>
        <div className="layer-content">
          <h4>{name}</h4>
          {properties.map((prop, index) => (
            <p key={index} className="layer-property">
              <span className="property-name">{prop.name}:</span> {prop.value}
            </p>
          ))}
        </div>
        <div className="layer-connector right">
          <div className="connector-dot"></div>
        </div>
      </div>
      {description && <p className="layer-description">{description}</p>}
    </div>
  );
};

// Helper function to adjust color brightness without using darken/lighten
const adjustColorBrightness = (hex: string, percent: number): string => {
  // Convert hex to RGB
  let r = parseInt(hex.substring(1, 3), 16);
  let g = parseInt(hex.substring(3, 5), 16);
  let b = parseInt(hex.substring(5, 7), 16);

  // Convert to HSL
  r /= 255;
  g /= 255;
  b /= 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h,
    s,
    l = (max + min) / 2;

  if (max === min) {
    h = s = 0; // achromatic
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
      default:
        h = 0;
    }
    h /= 6;
  }

  // Adjust lightness
  l = Math.max(0, Math.min(1, l + percent / 100));

  // Convert back to RGB
  let r1, g1, b1;
  if (s === 0) {
    r1 = g1 = b1 = l; // achromatic
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r1 = hue2rgb(p, q, h + 1 / 3);
    g1 = hue2rgb(p, q, h);
    b1 = hue2rgb(p, q, h - 1 / 3);
  }

  // Convert to HEX
  const toHex = (x: number) => {
    const hex = Math.round(x * 255).toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  return `#${toHex(r1)}${toHex(g1)}${toHex(b1)}`;
};

export default TutorialLayerCard;
