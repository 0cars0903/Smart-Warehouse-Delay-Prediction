"""Step 1: 데이터 로드 + 피처 엔지니어링 → pickle 저장"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

t0 = time.time()
print('[Step 1] 데이터 로드 + 피처 엔지니어링 시작...')

from run_v4_postprocess_IF import load_data, get_feat_cols
import pickle

train, test = load_data()
feat_cols = get_feat_cols(train)

cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', '_v4_IF_cache.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump({'train': train, 'test': test, 'feat_cols': feat_cols}, f)

print(f'  피처 수: {len(feat_cols)}')
print(f'  train: {train.shape}, test: {test.shape}')
print(f'  캐시 저장: {cache_path}')
print(f'  소요: {time.time()-t0:.1f}초')
