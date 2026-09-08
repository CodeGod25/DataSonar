import { Router } from 'express';
import { KafkaProducerService } from '../producers/kafka.producer';

/**
 * Create metrics router for Prometheus scraping
 */
export function createMetricsRouter(kafkaProducer: KafkaProducerService): Router {
  const router = Router();

  /**
   * GET /metrics
   * Prometheus-compatible metrics endpoint
   */
  router.get('/', (_req, res) => {
    try {
      const metrics = kafkaProducer.getPrometheusMetrics();

      res.set('Content-Type', 'text/plain');
      res.send(metrics);
    } catch (error) {
      res.status(500).send('# Error generating metrics\n');
    }
  });

  return router;
}