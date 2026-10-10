/**
 * Kafka Producer Metrics Utility
 * Tracks ingestion and Kafka production metrics for observability
 */

import { config } from '../config';
import { logger } from '../utils/logger';

/**
 * Metrics storage for Kafka producer
 */
export interface KafkaProducerMetrics {
  messagesSent: number;
  messagesFailed: number;
  bytesSent: number;
  lastSuccessTimestamp: number | null;
  lastFailureTimestamp: number | null;
  averageLatencyMs: number;
  latencySum: number;
  latencyCount: number;
}

/**
 * Kafka metrics collector
 */
export class KafkaMetrics {
  private static instance: KafkaMetrics;
  private metrics: KafkaProducerMetrics;
  private enabled: boolean;

  private constructor() {
    this.enabled = config.metrics?.enabled ?? true;
    this.metrics = {
      messagesSent: 0,
      messagesFailed: 0,
      bytesSent: 0,
      lastSuccessTimestamp: null,
      lastFailureTimestamp: null,
      averageLatencyMs: 0,
      latencySum: 0,
      latencyCount: 0
    };
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): KafkaMetrics {
    if (!KafkaMetrics.instance) {
      KafkaMetrics.instance = new KafkaMetrics();
    }
    return KafkaMetrics.instance;
  }

  /**
   * Record a successful message send
   */
  public recordSuccess(bytes: number, latencyMs: number): void {
    if (!this.enabled) return;

    this.metrics.messagesSent++;
    this.metrics.bytesSent += bytes;
    this.metrics.lastSuccessTimestamp = Date.now();

    // Update latency metrics
    this.metrics.latencySum += latencyMs;
    this.metrics.latencyCount++;
    this.metrics.averageLatencyMs =
      this.metrics.latencyCount > 0
        ? this.metrics.latencySum / this.metrics.latencyCount
        : 0;

    logger.debug('Kafka message sent successfully', {
      bytes,
      latencyMs,
      totalSent: this.metrics.messagesSent
    });
  }

  /**
   * Record a failed message send
   */
  public recordFailure(error: any): void {
    if (!this.enabled) return;

    this.metrics.messagesFailed++;
    this.metrics.lastFailureTimestamp = Date.now();

    logger.warn('Kafka message failed to send', {
      error: error.message || error,
      totalFailed: this.metrics.messagesFailed
    });
  }

  /**
   * Get current metrics snapshot
   */
  public getMetrics(): KafkaProducerMetrics {
    return { ...this.metrics };
  }

  /**
   * Get metrics in Prometheus format
   */
  public getPrometheusMetrics(): string {
    if (!this.enabled) return '';

    const m = this.metrics;
    return `# HELP datasonar_ingestion_kafka_messages_sent Total number of messages sent to Kafka
# TYPE datasonar_ingestion_kafka_messages_sent counter
datasonar_ingestion_kafka_messages_sent ${m.messagesSent}
# HELP datasonar_ingestion_kafka_messages_failed Total number of messages failed to send to Kafka
# TYPE datasonar_ingestion_kafka_messages_failed counter
datasonar_ingestion_kafka_messages_failed ${m.messagesFailed}
# HELP datasonar_ingestion_kafka_bytes_sent Total bytes sent to Kafka
# TYPE datasonar_ingestion_kafka_bytes_sent counter
datasonar_ingestion_kafka_bytes_sent ${m.bytesSent}
# HELP datasonar_ingestion_kafka_average_latency_ms Average latency for Kafka sends in milliseconds
# TYPE datasonar_ingestion_kafka_average_latency_ms gauge
datasonar_ingestion_kafka_average_latency_ms ${m.averageLatencyMs}
# HELP datasonar_ingestion_kafka_last_success_timestamp Timestamp of last successful Kafka send
# TYPE datasonar_ingestion_kafka_last_success_timestamp gauge
datasonar_ingestion_kafka_last_success_timestamp ${m.lastSuccessTimestamp ?? 0}
# HELP datasonar_ingestion_kafka_last_failure_timestamp Timestamp of last failed Kafka send
# TYPE datasonar_ingestion_kafka_last_failure_timestamp gauge
datasonar_ingestion_kafka_last_failure_timestamp ${m.lastFailureTimestamp ?? 0}
`;
  }

  /**
   * Reset metrics (useful for testing)
   */
  public reset(): void {
    if (!this.enabled) return;

    this.metrics = {
      messagesSent: 0,
      messagesFailed: 0,
      bytesSent: 0,
      lastSuccessTimestamp: null,
      lastFailureTimestamp: null,
      averageLatencyMs: 0,
      latencySum: 0,
      latencyCount: 0
    };
  }
}