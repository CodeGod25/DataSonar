import { Router } from 'express';
import { IngestionController } from '../controllers/ingestion.controller';
import { KafkaProducerService } from '../producers/kafka.producer';
import { DemoTelemetryService } from '../services/demoTelemetry.service';
import { createMetricsRouter } from './metrics.routes';
import { createDlqRouter } from './dlq.routes';

export function createIngestionRoutes(
  kafkaProducer: KafkaProducerService,
  telemetry: DemoTelemetryService
): Router {
  const router = Router();
  const controller = new IngestionController(kafkaProducer, telemetry);

  router.post('/ingest', controller.ingestEvent);
  router.post('/ingest/batch', controller.ingestBatch);

  // Mount metrics router
  router.use('/metrics', createMetricsRouter(kafkaProducer));

  // Mount DLQ inspection router (in production, this should be protected/disabled)
  router.use('/dlq', createDlqRouter(kafkaProducer));

  return router;
}