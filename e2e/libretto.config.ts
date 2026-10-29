import { defineConfig } from 'libretto'

export default defineConfig({
  baseURL: 'http://127.0.0.1:8090',
  specs: 'specs',
  agent: { backend: 'claude', model: 'claude-sonnet-5-5', maxTurns: 40, maxBudgetUsd: 2 },
})
