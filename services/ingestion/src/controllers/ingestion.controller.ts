import { Request, Response } from 'express';
import { randomUUID } from 'crypto';
import { PipelineEventSchema } from '../schemas/event.schema';
import { KafkaProducerService } from '../producers/kafka.producer';
import { logger } from '../utils/logger';
import { config } from '../config';
import { DemoTelemetryService, ValidationIssue } from '../services/demoTelemetry.service';
import { ValidationExplainer, EnhancedValidationIssue } from '../utils/validation-explainer';
import { PreprocessingExplainer, PreprocessingStep } from '../utils/preprocessing-explainer';

export class IngestionController {
  constructor(
    private kafkaProducer: KafkaProducerService,
    private telemetry: DemoTelemetryService
  ) {}

  /**
   * POST /api/v1/ingest
   * Receives a pipeline data event, validates it, enriches it, and sends to Kafka
   */
  ingestEvent = async (req: Request, res: Response): Promise<void> => {
    const startTime = Date.now();

    try {
      // Record preprocessing steps
      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.SCHEMA_ENRICHMENT,
        'Starting event validation and enrichment',
        { sourceIp: req.ip, userAgent: req.get('User-Agent') }
      );

      // 1. Validate the incoming payload
      const validationResult = PipelineEventSchema.safeParse(req.body);

      if (!validationResult.success) {
        // Enhanced validation error handling with explainability
        const enhancedErrors: EnhancedValidationIssue[] =
          ValidationExplainer.explainErrors(validationResult.error);

        // Record preprocessing step for validation
        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.VALIDATION_ENRICHMENT,
          'Validation failed - recording errors',
          { errorCount: enhancedErrors.length }
        );

        // Log detailed validation errors for debugging
        logger.warn('Event validation failed', {
          errors: enhancedErrors,
          sourceIp: req.ip,
          userAgent: req.get('User-Agent'),
          requestId: req.headers['x-request-id'] || 'unknown'
        });

        // Also log formatted version for easier console reading
        if (logger.isLevelEnabled('debug')) {
          logger.debug(
            `Validation error details:\n${ValidationExplainer.formatForLog(enhancedErrors)}`,
            { sourceIp: req.ip }
          );
        }

        // Convert to legacy format for backward compatibility with telemetry
        const legacyErrors: ValidationIssue[] = enhancedErrors.map(error => ({
          field: error.field,
          message: error.message
        }));

        // Send to dead letter queue for analysis
        await this.kafkaProducer.sendToDeadLetter(
          req.body,
          JSON.stringify(enhancedErrors), // Store enhanced errors in DLQ
          req.ip
        );
        await this.telemetry.recordRejectedEvent(req.body, legacyErrors, req.ip);

        res.status(400).json({
          status: 'error',
          message: 'Validation failed',
          // Include enhanced errors for clients that can use them
          errors: enhancedErrors,
          // Also include legacy format for backward compatibility
          legacyErrors,
          // Provide a user-friendly summary
          summary: ValidationExplainer.createUserSummary(enhancedErrors)
        });
        return;
      }

      // Record preprocessing step for successful validation
      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.VALIDATION_ENRICHMENT,
        'Validation passed - enriching event',
        { recordCount: validationResult.data.data.recordCount }
      );

      // 2. Enrich the event
      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.UUID_GENERATION,
        'Generating UUID for event'
      );

      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.TIMESTAMP_ADDITION,
        'Adding received timestamp'
      );

      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.SERVICE_IDENTIFICATION,
        'Adding ingestion service identification'
      );

      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.SCHEMA_VERSION_ADDITION,
        'Adding schema version'
      );

      const enrichedEvent = {
        ...validationResult.data,
        eventId: randomUUID(),
        receivedAt: new Date().toISOString(),
        ingestionService: config.service.name,
        validationStatus: 'valid' as const,
        // Add schema version for future evolution tracking
        schemaVersion: '1.0'
      };

      // 3. Send to Kafka
      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.SCHEMA_ENRICHMENT,
        'Preparing event for Kafka transmission'
      );

      await this.kafkaProducer.sendEvent(enrichedEvent);
      await this.telemetry.recordAcceptedEvent(enrichedEvent);

      const processingTime = Date.now() - startTime;

      logger.info('Event ingested successfully', {
        eventId: enrichedEvent.eventId,
        sourceId: enrichedEvent.sourceId,
        recordCount: enrichedEvent.data.recordCount,
        processingTimeMs: processingTime,
        schemaVersion: enrichedEvent.schemaVersion
      });

      // 4. Respond
      res.status(202).json({
        status: 'accepted',
        eventId: enrichedEvent.eventId,
        receivedAt: enrichedEvent.receivedAt,
        processingTimeMs: processingTime,
        schemaVersion: enrichedEvent.schemaVersion
      });
    } catch (error) {
      logger.error('Ingestion failed', { error });

      res.status(500).json({
        status: 'error',
        message: 'Internal server error during ingestion'
      });
    }
  };

  /**
   * POST /api/v1/ingest/batch
   * Receives multiple events in a single request
   */
  ingestBatch = async (req: Request, res: Response): Promise<void> => {
    const startTime = Date.now();

    try {
      // Record preprocessing step for batch start
      this.telemetry.recordPreprocessingStep(
        PreprocessingStep.SCHEMA_ENRICHMENT,
        'Starting batch ingestion processing',
        { batchSize: req.body.events.length, sourceIp: req.ip }
      );

      const events = req.body.events;

      if (!Array.isArray(events) || events.length === 0) {
        res.status(400).json({
          status: 'error',
          message: 'Request body must contain a non-empty "events" array'
        });
        return;
      }

      if (events.length > 100) {
        res.status(400).json({
          status: 'error',
          message: 'Batch size cannot exceed 100 events'
        });
        return;
      }

      const results = {
        accepted: [] as string[],
        rejected: [] as { index: number; errors: EnhancedValidationIssue[] }[]
      };

      // Process each event in the batch
      for (let i = 0; i < events.length; i++) {
        // Record preprocessing step for each batch item
        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.SCHEMA_ENRICHMENT,
          `Processing batch item ${i}`,
          { batchIndex: i }
        );

        const validationResult = PipelineEventSchema.safeParse(events[i]);

        if (!validationResult.success) {
          // Enhanced validation error handling for batch items
          const enhancedErrors: EnhancedValidationIssue[] =
            ValidationExplainer.explainErrors(validationResult.error);

          results.rejected.push({
            index: i,
            errors: enhancedErrors
          });

          // Record preprocessing step for validation failure
          this.telemetry.recordPreprocessingStep(
            PreprocessingStep.VALIDATION_ENRICHMENT,
            `Batch item ${i} validation failed`,
            { errorCount: enhancedErrors.length, batchIndex: i }
          );

          // Log validation error for batch item
          logger.warn(`Batch item ${i} validation failed`, {
            errors: enhancedErrors,
            batchIndex: i,
            sourceIp: req.ip
          });

          await this.kafkaProducer.sendToDeadLetter(
            events[i],
            JSON.stringify({
              batchIndex: i,
              validationErrors: enhancedErrors
            }),
            req.ip
          );
          await this.telemetry.recordRejectedEvent(
            events[i],
            enhancedErrors.map(e => ({ field: e.field, message: e.message })),
            req.ip
          );
          continue;
        }

        // Record preprocessing steps for successful batch item
        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.VALIDATION_ENRICHMENT,
          `Batch item ${i} validation passed`,
          { batchIndex: i }
        );

        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.UUID_GENERATION,
          `Generating UUID for batch item ${i}`,
          { batchIndex: i }
        );

        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.TIMESTAMP_ADDITION,
          `Adding timestamp for batch item ${i}`,
          { batchIndex: i }
        );

        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.SERVICE_IDENTIFICATION,
          `Adding service ID for batch item ${i}`,
          { batchIndex: i }
        );

        this.telemetry.recordPreprocessingStep(
          PreprocessingStep.SCHEMA_VERSION_ADDITION,
          `Adding schema version for batch item ${i}`,
          { batchIndex: i }
        );

        const enrichedEvent = {
          ...validationResult.data,
          eventId: randomUUID(),
          receivedAt: new Date().toISOString(),
          ingestionService: config.service.name,
          validationStatus: 'valid' as const,
          schemaVersion: '1.0'
        };

        await this.kafkaProducer.sendEvent(enrichedEvent);
        await this.telemetry.recordAcceptedEvent(enrichedEvent);
        results.accepted.push(enrichedEvent.eventId);
      }

      const processingTime = Date.now() - startTime;

      logger.info('Batch ingestion completed', {
        total: events.length,
        accepted: results.accepted.length,
        rejected: results.rejected.length,
        processingTimeMs: processingTime
      });

      res.status(202).json({
        status: 'completed',
        summary: {
          total: events.length,
          accepted: results.accepted.length,
          rejected: results.rejected.length
        },
        acceptedEventIds: results.accepted,
        // Include detailed rejection information
        rejections: results.rejected,
        processingTimeMs: processingTime
      });
    } catch (error) {
      logger.error('Batch ingestion failed', { error });

      res.status(500).json({
        status: 'error',
        message: 'Internal server error during batch ingestion'
      });
    }
  };

  /**
   * GET /health
   */
  healthCheck = async (_req: Request, res: Response): Promise<void> => {
    res.status(200).json({
      status: 'healthy',
      service: config.service.name,
      kafka: this.kafkaProducer.getStatus() ? 'connected' : 'disconnected',
      uptime: process.uptime(),
      timestamp: new Date().toISOString()
    });
  };
}
