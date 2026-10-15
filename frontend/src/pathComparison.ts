import type { ComparedPath, FormSettings, PathId } from './types';

export const PATH_META: Record<PathId, { label: string; english: string; color: string; description: string }> = {
  direct_sum: { label: '直接加總', english: 'Direct sum', color: '#7c8997', description: '不加 channel 傳播延遲，也不加 CF 路徑濾波。' },
  delay_only: { label: '只有延遲', english: 'Delay only', color: '#8c6faf', description: '保留逐級傳播延遲，移除 CF 路徑濾波。' },
  filter_only: { label: '只有 CF 濾波', english: 'Filter only', color: '#c58b2a', description: '保留 CF 路徑濾波，移除逐級傳播延遲。' },
  full: { label: '完整路徑', english: 'Full path', color: '#087f8c', description: '同時使用逐級傳播延遲與 CF 路徑濾波。' },
};

export function makePathComparisonRequest(middleEar: FormSettings['middleEar']) {
  return { point_count: 48, middle_ear: middleEar };
}

export function summarizePathComparison(paths: Pick<ComparedPath, 'id' | 'metrics'>[]) {
  const comparable = paths.filter(path => path.metrics.test.count > 0 && Number.isFinite(path.metrics.test.rmse_db));
  const best = comparable.reduce<Pick<ComparedPath, 'id' | 'metrics'> | null>((current, path) => !current || path.metrics.test.rmse_db < current.metrics.test.rmse_db ? path : current, null);
  const direct = comparable.find(path => path.id === 'direct_sum');
  const full = comparable.find(path => path.id === 'full');
  const difference = direct && full ? full.metrics.test.rmse_db - direct.metrics.test.rmse_db : null;
  let comparison = '目前沒有足夠的保留頻帶資料比較完整路徑與直接加總。';
  if (difference !== null) {
    comparison = Math.abs(difference) < 1e-8 ? '完整路徑與直接加總在保留頻帶的 test RMSE 相同。'
      : `完整路徑在保留頻帶的 test RMSE 比直接加總${difference < 0 ? '低' : '高'} ${Math.abs(difference).toFixed(2)} dB。`;
  }
  return { bestPath: best?.id ?? null, difference, comparison };
}
