import React from "react";

interface WorkflowStepProps {
  number: number;
  title: string;
  children: React.ReactNode;
}

const WorkflowStep: React.FC<WorkflowStepProps> = ({
  number,
  title,
  children,
}) => {
  return (
    <div className="workflow-step">
      <div className="step-number">{number}</div>
      <div className="step-content">
        <h4>{title}</h4>
        {children}
      </div>
    </div>
  );
};

interface WorkflowProps {
  children: React.ReactNode;
  title?: string;
}

export const Workflow: React.FC<WorkflowProps> = ({ children, title }) => {
  return (
    <div className="tutorial-workflow">
      {title && <h3>{title}</h3>}
      {children}
    </div>
  );
};

export default WorkflowStep;
