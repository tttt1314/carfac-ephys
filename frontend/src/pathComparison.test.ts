import { describe, expect, it } from 'vitest';
import { makePathComparisonRequest, summarizePathComparison } from './pathComparison';
import type { ComparedPath, PathId } from './types';

function metricsOnly(id: PathId, fit: number, test: number): Pick<ComparedPath, 'id' | 'metrics'> {
  return { id, metrics: { fit: { count: 2, mae_db: fit, rmse_db: fit }, test: { count: 2, mae_db: test, rmse_db: test }, all: { count: 4, mae_db: 0, rmse_db: 0 } } };
}

describe('four-path comparison protocol helpers', () => {
  it('sends only the fixed point count and shared middle-ear setting', () => {
    expect(makePathComparisonRequest('none')).toEqual({ point_count: 48, middle_ear: 'none' });
    expect(makePathComparisonRequest('Verhulst_2018')).not.toHaveProperty('health');
    expect(makePathComparisonRequest('Verhulst_2018')).not.toHaveProperty('source_coupling');
  });
  it('ranks held-out errors rather than rewarding a lower fit error', () => {
    const summary = summarizePathComparison([metricsOnly('direct_sum', 3, 2), metricsOnly('full', .1, 4), metricsOnly('delay_only', 5, 1)]);
    expect(summary.bestPath).toBe('delay_only');
    expect(summary.difference).toBe(2);
    expect(summary.comparison).toContain('高 2.00 dB');
  });
  it('reports improvement, ties and unavailable comparisons without pass/fail claims', () => {
    expect(summarizePathComparison([metricsOnly('direct_sum', 2, 5), metricsOnly('full', 1, 3)]).comparison).toContain('低 2.00 dB');
    expect(summarizePathComparison([metricsOnly('direct_sum', 2, 3), metricsOnly('full', 1, 3)]).comparison).toContain('相同');
    expect(summarizePathComparison([])).toMatchObject({ bestPath: null, difference: null });
  });
});
