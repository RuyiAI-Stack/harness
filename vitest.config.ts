import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'unit',
          include: ['packages/*/tests/unit/**/*.spec.ts', 'tests/unit/**/*.spec.ts'],
          environment: 'node',
        },
      },
      {
        test: {
          name: 'integration',
          include: ['packages/*/tests/integration/**/*.spec.ts', 'tests/integration/**/*.spec.ts'],
          environment: 'node',
          hookTimeout: 120_000,
          testTimeout: 120_000,
        },
      },
    ],
  },
})
