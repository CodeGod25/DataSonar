import { Router } from 'express';
import { KafkaProducerService } from '../producers/kafka.producer';

/**
 * Create DLQ inspection router (for debugging purposes)
 * In production, this should be secured or disabled
 */
export function createDlqRouter(kafkaProducer: KafkaProducerService): Router {
  const router = Router();

  // Note: In a real implementation, we would need a Kafka consumer to read from DLQ
  // For now, this is a placeholder showing where DLQ inspection would go
  router.get('/', (_req, res) => {
    res.json({
      status: 'info',
      message: 'DLQ inspection endpoint',
      note: 'In a full implementation, this would provide access to dead letter queue contents for debugging',
      suggestion: 'Use Kafka CLI tools or UI to inspect the datasonar.dead-letter topic directly'
    });
  });

  return router;
}