/**
 * Utility for tracking and explaining preprocessing steps in the ingestion pipeline
 */

import { logger } from './logger';

/**
 * Preprocessing step types
 */
export enum PreprocessingStep {
  SCHEMA_ENRICHMENT = 'schema_enrichment',
  UUID_GENERATION = 'uuid_generation',
  TIMESTAMP_ADDITION = 'timestamp_addition',
  SERVICE_IDENTIFICATION = 'service_identification',
  SCHEMA_VERSION_ADDITION = 'schema_version_addition',
  VALIDATION_ENRICHMENT = 'validation_enrichment'
}

/**
 * Preprocessing explanation entry
 */
export interface PreprocessingExplanation {
  step: PreprocessingStep;
  timestamp: string;
  description: string;
  details?: Record<string, any>;
}

/**
 * Tracks preprocessing steps for explainability
 */
export class PreprocessingExplainer {
  private static instance: PreprocessingExplainer;
  private steps: PreprocessingExplanation[];
  private enabled: boolean;

  private constructor() {
    // In a real implementation, this would come from config
    this.enabled = true;
    this.steps = [];
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): PreprocessingExplainer {
    if (!PreprocessingExplainer.instance) {
      PreprocessingExplainer.instance = new PreprocessingExplainer();
    }
    return PreprocessingExplainer.instance;
  }

  /**
   * Record a preprocessing step
   */
  public recordStep(
    step: PreprocessingStep,
    description: string,
    details?: Record<string, any>
  ): void {
    if (!this.enabled) return;

    const explanation: PreprocessingExplanation = {
      step,
      timestamp: new Date().toISOString(),
      description,
      details
    };

    this.steps.push(explanation);

    logger.debug(`Preprocessing step: ${step}`, {
      step,
      description,
      details,
      timestamp: explanation.timestamp
    });
  }

  /**
   * Get all recorded preprocessing steps
   */
  public getSteps(): PreprocessingExplanation[] {
    return [...this.steps];
  }

  /**
   * Get steps by type
   */
  public getStepsByType(step: PreprocessingStep): PreprocessingExplanation[] {
    return this.steps.filter(s => s.step === step);
  }

  /**
   * Clear all recorded steps
   */
  public clear(): void {
    this.steps = [];
  }

  /**
   * Get preprocessing summary
   */
  public getSummary(): Record<string, number> {
    const summary: Record<string, number> = {};
    this.steps.forEach(step => {
      summary[step.step] = (summary[step.step] || 0) + 1;
    });
    return summary;
  }
}