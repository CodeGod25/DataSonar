import { Kafka, Producer, Partitioners, CompressionTypes } from 'kafkajs';
import { config } from '../config';
import { logger } from '../utils/logger';
import { EnrichedEvent } from '../schemas/event.schema';
import { KafkaMetrics } from '../utils/kafka-metrics';

export class KafkaProducerService {
  private kafka: Kafka;
  private producer: Producer;
  private isConnected: boolean = false;
  private metrics = KafkaMetrics.getInstance();

  constructor() {
    this.kafka = new Kafka({
      clientId: config.kafka.clientId,
      brokers: config.kafka.brokers,
      retry: {
        initialRetryTime: 100,
        retries: 8,
      },
    });

    this.producer = this.kafka.producer({
      createPartitioner: Partitioners.DefaultPartitioner,
      allowAutoTopicCreation: true,
      transactionalId: undefined,
    });
  }

  async connect(): Promise<void> {
    try {
      await this.producer.connect();
      this.isConnected = true;
      logger.info('Kafka producer connected successfully', {
        brokers: config.kafka.brokers,
      });
    } catch (error) {
      logger.error('Failed to connect Kafka producer', { error });
      throw error;
    }
  }

  async sendEvent(event: EnrichedEvent): Promise<void> {
    if (!this.isConnected) {
      throw new Error('Kafka producer is not connected');
    }

    const startTime = Date.now();
    const eventString = JSON.stringify(event);
    const bytes = Buffer.byteLength(eventString, 'utf8');

    try {
      await this.producer.send({
        topic: config.kafka.topics.rawEvents,
        compression: CompressionTypes.GZIP,
        messages: [
          {
            key: event.sourceId,
            value: eventString,
            headers: {
              'event-type': event.eventType,
              'source-id': event.sourceId,
              'event-id': event.eventId,
              'received-at': event.receivedAt,
              'ingestion-service': config.service.name,
              'schema-version': event.schemaVersion || '1.0'
            },
          },
        ],
      });

      const latencyMs = Date.now() - startTime;
      this.metrics.recordSuccess(bytes, latencyMs);

      logger.debug('Event sent to Kafka', {
        eventId: event.eventId,
        topic: config.kafka.topics.rawEvents,
        sourceId: event.sourceId,
        bytes,
        latencyMs
      });
    } catch (error) {
      const latencyMs = Date.now() - startTime;
      this.metrics.recordFailure(error);

      logger.error('Failed to send event to Kafka', {
        eventId: event.eventId,
        error,
        latencyMs
      });
      throw error;
    }
  }

  async sendToDeadLetter(
    rawPayload: unknown,
    error: string,
    sourceIp?: string
  ): Promise<void> {
    if (!this.isConnected) {
      logger.warn('Cannot send to DLQ — producer not connected');
      return;
    }

    const startTime = Date.now();
    const dlqPayload = {
      originalPayload: rawPayload,
      error,
      failedAt: new Date().toISOString(),
      sourceIp,
      ingestionService: config.service.name
    };

    const payloadString = JSON.stringify(dlqPayload);
    const bytes = Buffer.byteLength(payloadString, 'utf8');

    try {
      await this.producer.send({
        topic: config.kafka.topics.deadLetter,
        messages: [
          {
            value: payloadString,
            headers: {
              'dlq-reason': 'validation_error',
              'failed-at': new Date().toISOString(),
              'ingestion-service': config.service.name
            },
          },
        ],
      });

      const latencyMs = Date.now() - startTime;
      this.metrics.recordSuccess(bytes, latencyMs);

      logger.warn('Event sent to dead letter queue', { error });
    } catch (dlqError) {
      const latencyMs = Date.now() - startTime;
      this.metrics.recordFailure(dlqError);

      logger.error('Failed to send to dead letter queue', { dlqError });
    }
  }

  async disconnect(): Promise<void> {
    await this.producer.disconnect();
    this.isConnected = false;
    logger.info('Kafka producer disconnected');
  }

  getStatus(): boolean {
    return this.isConnected;
  }

  /**
   * Get Kafka producer metrics
   */
  getMetrics() {
    return this.metrics.getMetrics();
  }

  /**
   * Get metrics in Prometheus format
   */
  getPrometheusMetrics(): string {
    return this.metrics.getPrometheusMetrics();
  }
}