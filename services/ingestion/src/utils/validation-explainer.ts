/**
 * Utility for providing detailed, actionable explanations of Zod validation errors
 * Enhances the default Zod error messages with context and remediation suggestions
 */

import { ZodError, ZodIssue } from 'zod';

/**
 * Enhanced error information with actionable guidance
 */
export interface EnhancedValidationIssue {
  field: string;
  message: string;
  received: any;
  expected: string | string[];
  validation: string;
  suggestion?: string;
}

/**
 * Explains a Zod validation error with enhanced context and suggestions
 */
export class ValidationExplainer {
  /**
   * Transforms Zod errors into enhanced, actionable explanations
   */
  static explainErrors(error: ZodError): EnhancedValidationIssue[] {
    return error.issues.map(issue => this.explainSingleIssue(issue));
  }

  /**
   * Explains a single Zod issue with detailed context
   */
  private static explainSingleIssue(issue: ZodIssue): EnhancedValidationIssue {
    const { path, message, code } = issue;
    const field = path.length > 0 ? path.join('.') : '<root>';

    // Determine expected value/type based on error code
    let expected: string | string[] = 'unknown';
    let received: any = (issue as any).received;
    let validationRule = message;
    let suggestion: string | undefined;

    switch (code as string) {
      case 'invalid_type':
        expected = (issue as any).expected ?? 'unknown';
        validationRule = `Expected type "${expected}"`;
        // Handle expected being either string or string[]
        const expectedStr = Array.isArray(expected)
          ? expected.join(' | ')
          : String(expected);
        suggestion = this.getTypeSuggestion(received, expectedStr);
        break;

      case 'invalid_literal':
        expected = (issue as any).expected ?? 'unknown';
        validationRule = `Expected literal value "${expected}"`;
        suggestion = `Check if the value should be "${expected}"`;
        break;

      case 'unrecognized_keys':
        expected = (issue as any).keys ?? [];
        received = undefined;
        validationRule = `Unrecognized key(s): ${Array.isArray(expected) ? expected.join(', ') : expected}`;
        suggestion = `Remove the unrecognized field(s) or check for typos`;
        break;

      case 'invalid_union':
        received = undefined;
        validationRule = 'Value does not match any of the allowed union types';
        suggestion = 'Check the allowed values for this field';
        break;

      case 'invalid_union_discriminator':
        received = undefined;
        validationRule = 'Union discriminator is missing or invalid';
        suggestion = 'Ensure the discriminator field is present and has a valid value';
        break;

      case 'too_big':
        received = undefined;
        validationRule = `Value exceeds maximum limit`;
        suggestion = `Reduce the value to be less than or equal to the limit`;
        break;

      case 'too_small':
        received = undefined;
        validationRule = `Value is below minimum limit`;
        suggestion = `Increase the value to be greater than or equal to the limit`;
        break;

      case 'invalid_string':
        // For string validation, try to get more specific error info
        if (typeof message === 'string') {
          if (message.includes('email')) {
            validationRule = 'Invalid email format';
            suggestion = 'Provide a valid email address (e.g., user@example.com)';
          } else if (message.includes('url')) {
            validationRule = 'Invalid URL format';
            suggestion = 'Provide a valid URL (e.g., https://example.com)';
          } else if (message.includes('includes')) {
            // Extract the required substring from error message if possible
            const match = message.match(/must include "([^"]+)"/);
            if (match) {
              validationRule = `Must include "${match[1]}"`;
              suggestion = `Ensure the value contains "${match[1]}"`;
            } else {
              validationRule = message;
            }
          } else if (message.includes('startsWith')) {
            // Extract the required prefix from error message if possible
            const match = message.match(/must start with "([^"]+)"/);
            if (match) {
              validationRule = `Must start with "${match[1]}"`;
              suggestion = `Ensure the value starts with "${match[1]}"`;
            } else {
              validationRule = message;
            }
          } else if (message.includes('endsWith')) {
            // Extract the required suffix from error message if possible
            const match = message.match(/must end with "([^"]+)"/);
            if (match) {
              validationRule = `Must end with "${match[1]}"`;
              suggestion = `Ensure the value ends with "${match[1]}"`;
            } else {
              validationRule = message;
            }
          } else {
            validationRule = message;
          }
        }
        break;

      case 'invalid_date':
        received = (issue as any).received;
        validationRule = 'Invalid date format';
        suggestion = 'Provide a valid date in ISO 8601 format (e.g., 2023-01-15T10:30:00Z)';
        break;

      case 'invalid_arguments':
        received = (issue as any).received;
        validationRule = 'Invalid function arguments';
        break;

      case 'invalid_return_type':
        received = (issue as any).received;
        validationRule = 'Invalid function return type';
        break;

      case 'invalid_params':
        received = (issue as any).received;
        validationRule = 'Invalid parameters';
        break;

      case 'custom':
        received = (issue as any).received;
        validationRule = message;
        // Try to extract suggestion from custom message if it follows a pattern
        suggestion = this.extractSuggestionFromMessage(message);
        break;

      case 'invalid':
        received = (issue as any).received;
        validationRule = message;
        break;

      default:
        received = (issue as any).received;
        validationRule = message;
        break;
    }

    return {
      field,
      message,
      received,
      expected,
      validation: validationRule,
      suggestion
    };
  }

  /**
   * Provides type-specific suggestions based on received and expected types
   */
  private static getTypeSuggestion(received: any, expected: string): string | undefined {
    const receivedType = typeof received;

    // Handle null values
    if (received === null) {
      return 'Value cannot be null; provide a valid value or make the field optional if appropriate';
    }

    // Handle undefined values
    if (receivedType === 'undefined') {
      return 'Value is required; provide a valid value';
    }

    // Type-specific suggestions
    switch (expected) {
      case 'string':
        if (receivedType === 'number') {
          return 'Convert number to string (e.g., String(${received}))';
        } else if (receivedType === 'boolean') {
          return 'Convert boolean to string (e.g., ${received ? "true" : "false"})';
        }
        break;

      case 'number':
        if (receivedType === 'string') {
          const num = Number(received);
          return !isNaN(num)
            ? `Convert string to number (e.g., Number("${received}") = ${num})`
            : `String "${received}" cannot be converted to a number`;
        } else if (receivedType === 'boolean') {
          return `Convert boolean to number (${received} ? 1 : 0)`;
        }
        break;

      case 'boolean':
        if (receivedType === 'string') {
          const lower = received.toLowerCase();
          if (lower === 'true' || lower === 'false') {
            return `String "${received}" converts to boolean ${lower === 'true'}`;
          }
          return 'Use "true" or "false" for boolean values';
        } else if (receivedType === 'number') {
          return `${received} converts to boolean ${received !== 0}`;
        }
        break;

      case 'array':
        if (Array.isArray(received)) {
          return 'Value is already an array';
        }
        return 'Provide an array value (e.g., ["item1", "item2"])';
    }

    return undefined;
  }

  /**
   * Attempts to extract a suggestion from custom error messages
   */
  private static extractSuggestionFromMessage(message: string): string | undefined {
    // Common patterns in custom error messages that might contain suggestions
    const suggestionPatterns = [
      /(?:please|try|consider|suggest)(?:ing)?[:\s]+(.+)/i,
      /(?:use|provide|ensure)[:\s]+(.+)/i,
      /(?:should|must|need to)[:\s]+(.+)/i
    ];

    for (const pattern of suggestionPatterns) {
      const match = message.match(pattern);
      if (match && match[1]) {
        return match[1].trim();
      }
    }

    return undefined;
  }

  /**
   * Formats enhanced errors for console/logging output
   */
  static formatForLog(errors: EnhancedValidationIssue[]): string {
    return errors
      .map(error => {
        let output = `❌ Field "${error.field}": ${error.message}`;
        output += `\n   📝 Validation: ${error.validation}`;
        output += `\n   📥 Received: ${JSON.stringify(error.received)}`;
        output += `\n   📤 Expected: ${Array.isArray(error.expected)
          ? error.expected.join(' | ')
          : error.expected}`;
        if (error.suggestion) {
          output += `\n   💡 Suggestion: ${error.suggestion}`;
        }
        return output;
      })
      .join('\n\n');
  }

  /**
   * Creates a user-friendly summary of validation errors
   */
  static createUserSummary(errors: EnhancedValidationIssue[]): string {
    if (errors.length === 0) return 'No validation errors';

    const errorCount = errors.length;
    const fieldErrors = new Map<string, number>();

    errors.forEach(error => {
      fieldErrors.set(error.field, (fieldErrors.get(error.field) || 0) + 1);
    });

    let summary = `❌ Validation failed with ${errorCount} error${errorCount !== 1 ? 's' : ''}`;

    if (fieldErrors.size > 0) {
      summary += `\n📊 Errors by field:`;
      for (const [field, count] of fieldErrors.entries()) {
        summary += `\n   • ${field}: ${count} error${count !== 1 ? 's' : ''}`;
      }
    }

    // Add top 3 suggestions
    const suggestions = errors
      .filter(e => e.suggestion)
      .slice(0, 3);

    if (suggestions.length > 0) {
      summary += `\n💡 Top suggestions:`;
      suggestions.forEach((s, index) => {
        summary += `\n   ${index + 1}. ${s.suggestion}`;
      });
    }

    return summary;
  }
}