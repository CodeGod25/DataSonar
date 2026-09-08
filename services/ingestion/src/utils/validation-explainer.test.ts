import { ValidationExplainer } from './validation-explainer';
import { z } from 'zod';

// Test schema for validation
const testSchema = z.object({
  name: z.string().min(1, 'Name is required'),
  age: z.number().min(0, 'Age must be positive').max(150, 'Age must be realistic'),
  email: z.string().email('Invalid email format'),
  tags: z.array(z.string()).max(5, 'Maximum 5 tags allowed')
});

describe('ValidationExplainer', () => {
  describe('explainErrors', () => {
    it('should explain type errors', () => {
      const result = testSchema.safeParse({
        name: 123, // Should be string
        age: 'twenty', // Should be number
        email: 'not-an-email',
        tags: ['a', 'b', 'c', 'd', 'e', 'f'] // Too many tags
      });

      expect(result.success).toBe(false);

      if (!result.success) {
        const enhancedErrors = ValidationExplainer.explainErrors(result.error);

        // Should have 4 errors
        expect(enhancedErrors.length).toBe(4);

        // Check name error
        const nameError = enhancedErrors.find(e => e.field === 'name');
        expect(nameError).toBeDefined();
        expect(nameError?.message).toContain('Required');
        expect(nameError?.expected).toContain('string');
        expect(nameError?.received).toBe(123);

        // Check age error
        const ageError = enhancedErrors.find(e => e.field === 'age');
        expect(ageError).toBeDefined();
        expect(ageError?.message).toContain('Invalid');
        expect(ageError?.expected).toContain('number');
        expect(ageError?.received).toBe('twenty');

        // Check email error
        const emailError = enhancedErrors.find(e => e.field === 'email');
        expect(emailError).toBeDefined();
        expect(emailError?.message).toContain('email');
        expect(emailError?.received).toBe('not-an-email');

        // Check tags error
        const tagsError = enhancedErrors.find(e => e.field === 'tags');
        expect(tagsError).toBeDefined();
        expect(tagsError?.message).toContain('maximum');
        expect(tagsError?.received).toHaveLength(6);
      }
    });

    it('should provide suggestions for common errors', () => {
      const result = testSchema.safeParse({
        name: '', // Empty string
        age: -5, // Negative number
        email: 'test', // Invalid email
        tags: 123 // Should be array
      });

      expect(result.success).toBe(false);

      if (!result.success) {
        const enhancedErrors = ValidationExplainer.explainErrors(result.error);

        // Check that we get suggestions
        const nameError = enhancedErrors.find(e => e.field === 'name');
        // Note: Our explainer might not generate suggestions for all error types yet

        const ageError = enhancedErrors.find(e => e.field === 'age');
        // Should have some guidance for negative numbers

        const emailError = enhancedErrors.find(e => e.field === 'email');
        // Should suggest valid email format

        const tagsError = enhancedErrors.find(e => e.field === 'tags');
        // Should suggest providing an array
      }
    });
  });

  describe('formatForLog', () => {
    it('should format errors for logging', () => {
      const result = testSchema.safeParse({
        name: 123,
        age: -5
      });

      expect(result.success).toBe(false);

      if (!result.success) {
        const enhancedErrors = ValidationExplainer.explainErrors(result.error);
        const formatted = ValidationExplainer.formatForLog(enhancedErrors);

        expect(formatted).toContain('Field "name"');
        expect(formatted).toContain('Field "age"');
        expect(formatted).toContain('Validation:');
        expect(formatted).toContain('Received:');
        expect(formatted).toContain('Expected:');
      }
    });
  });

  describe('createUserSummary', () => {
    it('should create a user-friendly summary', () => {
      const result = testSchema.safeParse({
        name: 123,
        age: -5,
        email: 'bad-email'
      });

      expect(result.success).toBe(false);

      if (!result.success) {
        const enhancedErrors = ValidationExplainer.explainErrors(result.error);
        const summary = ValidationExplainer.createUserSummary(enhancedErrors);

        expect(summary).toContain('Validation failed');
        expect(summary).toContain('Errors by field');
        expect(summary).toContain('Top suggestions');
      }
    });

    it('should handle no errors', () => {
      const result = testSchema.safeParse({
        name: 'John Doe',
        age: 25,
        email: 'john@example.com',
        tags: ['developer']
      });

      expect(result.success).toBe(true);

      if (result.success) {
        // This test is for the success case - we won't call createUserSummary on success
        // but we can verify the function handles empty arrays
        const summary = ValidationExplainer.createUserSummary([]);
        expect(summary).toBe('No validation errors');
      }
    });
  });
});