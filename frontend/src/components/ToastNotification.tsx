import React from "react";
import "../styles/components/ToastNotification.scss";

export type ToastType = "success" | "error" | "info" | "warning";

interface ToastNotificationProps {
  message: string;
  type?: ToastType;
  onClose: () => void;
}

const ToastNotification: React.FC<ToastNotificationProps> = ({
  message,
  type = "info",
  onClose,
}) => {
  React.useEffect(() => {
    const timer = setTimeout(onClose, 3500);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`toast-notification toast-${type}`}>
      <span className="toast-message">{message}</span>
      <button className="toast-close" onClick={onClose}>
        <i className="fas fa-times"></i>
      </button>
    </div>
  );
};

export default ToastNotification;
