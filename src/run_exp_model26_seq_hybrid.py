"""
모델실험26: Hybrid Sequence-Tabular Stacking (v3.0 Phase 1)
=============================================================
v3 전략 E: 시퀀스 모델(1D-CNN + BiLSTM)을 임베딩 생성기로 사용,
기존 GBDT 5모델 스태킹에 추가 base learner로 합류.

핵심 변경:
  - 시퀀스 입력: 원본 연속형 18종 (lag/rolling 제외 — MLP 실패 교훈)
  - 1D-CNN (k=3+5, f=32): 파라미터 ~30K, 위치 불변 필터
  - BiLSTM (h=16): 파라미터 ~20K, 양방향 시계열 포착
  - 출력: per-timestep 예측값 (OOF) → 메타 학습기의 추가 입력
  - 과적합 제어: Dropout 0.3, L2=1e-4, early stopping, fold별 정규화

Phase 1 목표:
  - CNN/LSTM 단독 OOF MAE 확인 (예상: 9.0~9.5)
  - 기존 LGBM OOF와의 상관 확인 → 0.95 미만이면 Phase 2 진행
  - Phase 2: 7모델(5 GBDT + CNN + LSTM) 하이브리드 스태킹

실행: python src/run_exp_model26_seq_hybrid.py
예상 시간: ~10분 (MPS), ~30분 (CPU)
출력: docs/model26_ckpt/
의존성: pip install torch (PyTorch)
GPU: Apple Silicon MPS 자동 감지 (CUDA > MPS > CPU 우선순위)
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import gc
import warnings
import json

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PyTorch imports
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import build_features

_BASE    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE, '..', 'data')
CKPT_DIR = os.path.join(_BASE, '..', 'docs', 'model26_ckpt')
M23_CKPT = os.path.join(_BASE, '..', 'docs', 'model23_ckpt')
M21_CKPT = os.path.join(_BASE, '..', 'docs', 'model21_ckpt')
N_SPLITS = 5
RANDOM_STATE = 42
# Apple Silicon MPS > CUDA > CPU 우선순위
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

os.makedirs(CKPT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 시퀀스 모델 입력 피처 (18종 — 원본 연속형만, lag/rolling 제외)
# ─────────────────────────────────────────────
SEQ_FEATURES = [
    # NaN=0% 그룹 (자기상관 0.91~0.98)
    'robot_utilization', 'robot_idle', 'robot_active', 'robot_charging',
    # NaN ~12% 그룹 (자기상관 0.86~0.99, 보간 필요)
    'order_inflow_15m', 'congestion_score', 'low_battery_ratio',
    'battery_mean', 'battery_std', 'charge_queue_length', 'max_zone_density',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time',
    'blocked_path_15m', 'sku_concentration', 'urgent_order_ratio', 'pack_utilization',
]

# ─────────────────────────────────────────────
# 시나리오 집계 피처 (시퀀스 모델에 정적 컨텍스트로 제공)
# ─────────────────────────────────────────────
SC_AGG_COLS = SEQ_FEATURES  # 동일 18종


def add_scenario_agg_features(df):
    """시나리오 집계 피처 broadcast (mean/std/max/min/diff × 18종 = 90피처)"""
    df = df.copy()
    for col in SC_AGG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]
        df[f'sc_{col}_mean'] = grp.transform('mean')
        df[f'sc_{col}_std']  = grp.transform('std').fillna(0)
        df[f'sc_{col}_max']  = grp.transform('max')
        df[f'sc_{col}_min']  = grp.transform('min')
        df[f'sc_{col}_diff'] = df[col] - df[f'sc_{col}_mean']
    return df


# ─────────────────────────────────────────────
# 데이터 전처리
# ─────────────────────────────────────────────
def prepare_sequence_data(train, test):
    """
    2D DataFrame → 3D 시퀀스 배열 변환
    Returns: X_train_3d (n_sc, 25, n_feat), X_test_3d, y_train_3d, scenario_ids_train
    """
    # 타임슬롯 인덱스 추가
    train = train.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    test  = test.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    train['ts_idx'] = train.groupby('scenario_id').cumcount()
    test['ts_idx']  = test.groupby('scenario_id').cumcount()

    # NaN 처리: 시나리오 내 선형 보간 → 시나리오 평균 → 전역 0
    for col in SEQ_FEATURES:
        if col not in train.columns:
            continue
        train[col] = train.groupby('scenario_id')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
        train[col] = train.groupby('scenario_id')[col].transform(
            lambda x: x.fillna(x.mean()))
        train[col] = train[col].fillna(0)

        test[col] = test.groupby('scenario_id')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
        test[col] = test.groupby('scenario_id')[col].transform(
            lambda x: x.fillna(x.mean()))
        test[col] = test[col].fillna(0)

    # 피처 선택 (실제 존재하는 것만)
    feat_cols = [c for c in SEQ_FEATURES if c in train.columns]
    n_feat = len(feat_cols)
    print(f'시퀀스 피처: {n_feat}종')

    # 시나리오별 3D 배열 구성
    train_scenarios = train.groupby('scenario_id')
    test_scenarios  = test.groupby('scenario_id')

    sc_ids_train = sorted(train['scenario_id'].unique())
    sc_ids_test  = sorted(test['scenario_id'].unique())

    X_train_3d = np.zeros((len(sc_ids_train), 25, n_feat), dtype=np.float32)
    y_train_3d = np.zeros((len(sc_ids_train), 25), dtype=np.float32)
    X_test_3d  = np.zeros((len(sc_ids_test), 25, n_feat), dtype=np.float32)

    for i, sc_id in enumerate(sc_ids_train):
        grp = train_scenarios.get_group(sc_id).sort_values('ts_idx')
        X_train_3d[i] = grp[feat_cols].values[:25]
        y_train_3d[i] = grp['avg_delay_minutes_next_30m'].values[:25]

    for i, sc_id in enumerate(sc_ids_test):
        grp = test_scenarios.get_group(sc_id).sort_values('ts_idx')
        X_test_3d[i] = grp[feat_cols].values[:25]

    return X_train_3d, X_test_3d, y_train_3d, sc_ids_train, sc_ids_test, feat_cols


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ─────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────
class CNN1D(nn.Module):
    """1D-CNN: 병렬 커널 (k=3, k=5) → concat → per-timestep 출력"""
    def __init__(self, n_feat, hidden=32, dropout=0.3):
        super().__init__()
        self.conv3 = nn.Conv1d(n_feat, hidden, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(n_feat, hidden, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (batch, seq_len=25, n_feat) → (batch, n_feat, seq_len) for Conv1d
        x_t = x.transpose(1, 2)
        h3 = torch.relu(self.conv3(x_t))   # (batch, hidden, 25)
        h5 = torch.relu(self.conv5(x_t))   # (batch, hidden, 25)
        h = torch.cat([h3, h5], dim=1)     # (batch, hidden*2, 25)
        h = self.bn(h)
        h = self.drop(h)
        h = h.transpose(1, 2)              # (batch, 25, hidden*2)
        out = self.fc(h).squeeze(-1)        # (batch, 25)
        return out


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM: h=16 양방향 → per-timestep 출력"""
    def __init__(self, n_feat, hidden=16, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat, hidden_size=hidden,
            batch_first=True, bidirectional=True, dropout=0)
        self.bn = nn.BatchNorm1d(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        # x: (batch, 25, n_feat)
        h, _ = self.lstm(x)                 # (batch, 25, hidden*2)
        h = h.transpose(1, 2)              # (batch, hidden*2, 25)
        h = self.bn(h)
        h = h.transpose(1, 2)              # (batch, 25, hidden*2)
        h = self.drop(h)
        out = self.fc(h).squeeze(-1)        # (batch, 25)
        return out


# ─────────────────────────────────────────────
# 학습 함수
# ─────────────────────────────────────────────
def train_seq_model(model_class, model_name, X_train_3d, X_test_3d, y_train_3d,
                     sc_ids_train, groups_arr, n_feat,
                     hidden=32, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                     epochs=100, patience=15, batch_size=128):
    """
    GroupKFold 5-fold로 시퀀스 모델 학습, OOF + test 예측 생성
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    n_train = len(X_train_3d)
    n_test  = len(X_test_3d)

    # OOF: (n_scenarios, 25) 형태 → 나중에 flatten해서 사용
    oof_pred = np.zeros((n_train, 25), dtype=np.float32)
    test_pred = np.zeros((n_test, 25), dtype=np.float32)

    # groups: 시나리오 인덱스 (0~9999)
    dummy_y = np.zeros(n_train)

    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(np.arange(n_train), dummy_y, groups_arr)):
        print(f'\n  [{model_name}] Fold {fold+1}/{N_SPLITS} (train={len(tr_idx)}, val={len(va_idx)})')

        # Fold별 정규화 (train fold 기준)
        scaler = StandardScaler()
        X_tr = X_train_3d[tr_idx].copy()
        X_va = X_train_3d[va_idx].copy()
        X_te = X_test_3d.copy()

        # reshape → 2D로 fit → 3D 복원
        orig_shape_tr = X_tr.shape
        orig_shape_va = X_va.shape
        orig_shape_te = X_te.shape

        X_tr_2d = X_tr.reshape(-1, n_feat)
        scaler.fit(X_tr_2d)

        X_tr = scaler.transform(X_tr_2d).reshape(orig_shape_tr).astype(np.float32)
        X_va = scaler.transform(X_va.reshape(-1, n_feat)).reshape(orig_shape_va).astype(np.float32)
        X_te = scaler.transform(X_te.reshape(-1, n_feat)).reshape(orig_shape_te).astype(np.float32)

        y_tr = y_train_3d[tr_idx]
        y_va = y_train_3d[va_idx]

        # log1p 타겟 (음수 방지 + 분포 안정화)
        y_tr_log = np.log1p(np.clip(y_tr, 0, None))
        y_va_log = np.log1p(np.clip(y_va, 0, None))

        # DataLoader (MPS: pin_memory 불가, CUDA: pin_memory 활용)
        pin = (DEVICE == 'cuda')
        tr_ds = SeqDataset(X_tr, y_tr_log)
        va_ds = SeqDataset(X_va, y_va_log)
        te_ds = SeqDataset(X_te)

        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           drop_last=False, pin_memory=pin, num_workers=0)
        va_dl = DataLoader(va_ds, batch_size=batch_size * 2, shuffle=False,
                           pin_memory=pin, num_workers=0)
        te_dl = DataLoader(te_ds, batch_size=batch_size * 2, shuffle=False,
                           pin_memory=pin, num_workers=0)

        # 모델 생성
        if model_class == CNN1D:
            model = CNN1D(n_feat=n_feat, hidden=hidden, dropout=dropout).to(DEVICE)
        else:
            model = BiLSTMModel(n_feat=n_feat, hidden=hidden, dropout=dropout).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.L1Loss()  # MAE in log1p space

        # 학습 루프
        best_val_mae = float('inf')
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batch = 0
            for X_b, y_b in tr_dl:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(X_b)
                loss = criterion(pred, y_b)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batch += 1

            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_b, y_b in va_dl:
                    X_b = X_b.to(DEVICE)
                    pred = model(X_b)
                    val_preds.append(pred.cpu().numpy())

            val_preds = np.concatenate(val_preds, axis=0)
            # MAE in raw space
            val_raw = np.expm1(val_preds)
            val_mae = np.abs(val_raw - y_va).mean()

            scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f'    Early stop at epoch {epoch+1}, best MAE={best_val_mae:.4f}')
                break

            if (epoch + 1) % 20 == 0:
                print(f'    Epoch {epoch+1}: train_loss={train_loss/n_batch:.4f}, val_MAE={val_mae:.4f}, best={best_val_mae:.4f}')

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Best 모델로 OOF + test 예측
        model.load_state_dict(best_state)
        model.eval()

        # OOF
        with torch.no_grad():
            va_preds = []
            for X_b, _ in va_dl:
                va_preds.append(model(X_b.to(DEVICE)).cpu().numpy())
            va_preds = np.concatenate(va_preds, axis=0)
            oof_pred[va_idx] = va_preds  # log1p space

        # Test
        with torch.no_grad():
            te_preds = []
            for X_b in te_dl:
                if isinstance(X_b, (list, tuple)):
                    X_b = X_b[0]
                te_preds.append(model(X_b.to(DEVICE)).cpu().numpy())
            te_preds = np.concatenate(te_preds, axis=0)
            test_pred += te_preds / N_SPLITS  # log1p space, 5-fold 평균

        fold_mae = best_val_mae
        fold_maes.append(fold_mae)
        print(f'    Fold {fold+1} best MAE: {fold_mae:.4f}')

        del model, optimizer, scheduler
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        elif DEVICE == 'mps':
            torch.mps.empty_cache()

    # 전체 OOF MAE
    oof_raw = np.expm1(oof_pred)
    overall_mae = np.abs(oof_raw - y_train_3d).mean()
    print(f'\n  [{model_name}] Overall OOF MAE: {overall_mae:.4f}')
    print(f'  [{model_name}] Fold MAEs: {[f"{m:.4f}" for m in fold_maes]}')

    # oof_pred, test_pred는 (n_scenarios, 25) 형태, log1p space
    return oof_pred, test_pred, overall_mae, fold_maes


# ─────────────────────────────────────────────
# 상관 분석
# ─────────────────────────────────────────────
def analyze_correlations(oof_cnn, oof_lstm, oof_lgbm, oof_tw, oof_cb, oof_et, oof_rf, y_raw_flat):
    """시퀀스 모델 OOF와 기존 GBDT OOF의 상관 분석"""
    print('\n' + '=' * 60)
    print('상관 분석 (Pearson)')
    print('=' * 60)

    names = ['CNN', 'LSTM', 'LGBM', 'TW', 'CB', 'ET', 'RF']
    oofs  = [oof_cnn, oof_lstm, oof_lgbm, oof_tw, oof_cb, oof_et, oof_rf]

    # 상관 행렬
    print(f'\n{"":8s}', end='')
    for n in names:
        print(f'{n:>8s}', end='')
    print()

    corr_matrix = {}
    for i, (n1, o1) in enumerate(zip(names, oofs)):
        print(f'{n1:8s}', end='')
        for j, (n2, o2) in enumerate(zip(names, oofs)):
            c = np.corrcoef(o1, o2)[0, 1]
            corr_matrix[(n1, n2)] = c
            print(f'{c:8.4f}', end='')
        print()

    # 핵심 지표: 시퀀스 모델 vs GBDT 상관
    print(f'\n핵심 상관:')
    for seq_name in ['CNN', 'LSTM']:
        for gbdt_name in ['LGBM', 'TW', 'CB', 'ET', 'RF']:
            c = corr_matrix[(seq_name, gbdt_name)]
            status = '✅ 다양성 유효' if c < 0.95 else '❌ 다양성 부족'
            print(f'  {seq_name}-{gbdt_name}: {c:.4f}  {status}')

    cnn_lgbm = corr_matrix[('CNN', 'LGBM')]
    lstm_lgbm = corr_matrix[('LSTM', 'LGBM')]
    cnn_lstm = corr_matrix[('CNN', 'LSTM')]

    print(f'\n판정:')
    if cnn_lgbm < 0.95 or lstm_lgbm < 0.95:
        print(f'  ✅ Phase 2 진행 가능 (CNN-LGBM={cnn_lgbm:.4f}, LSTM-LGBM={lstm_lgbm:.4f})')
    else:
        print(f'  ❌ 다양성 부족 — Phase 2 진행 불가')
        print(f'     전략 A(시나리오 형상 피처) 복귀 권장')

    return corr_matrix


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    t0 = time.time()
    print('=' * 60)
    print('모델실험26: Hybrid Sequence-Tabular Stacking (v3.0 Phase 1)')
    print(f'기준: model23 CV 8.5038 / Public 9.9522')
    print(f'목표: 시퀀스 모델 OOF 생성 + GBDT와의 상관 < 0.95 확인')
    print(f'Device: {DEVICE}')
    if DEVICE == 'mps':
        print(f'  Apple Silicon MPS 가속 활성화')
        # MPS float32 강제 — float64 미지원 방지
        torch.set_default_dtype(torch.float32)
    print('=' * 60)

    # 데이터 로드
    print('\n[1/5] 데이터 로드...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    # 시퀀스 데이터 준비 (3D)
    print('\n[2/5] 시퀀스 데이터 준비...')
    X_train_3d, X_test_3d, y_train_3d, sc_ids_train, sc_ids_test, feat_cols = \
        prepare_sequence_data(train, test)
    n_feat = len(feat_cols)
    print(f'  X_train: {X_train_3d.shape}, X_test: {X_test_3d.shape}')
    print(f'  y_train: {y_train_3d.shape}')
    print(f'  피처: {feat_cols}')

    # groups: 시나리오별 고유 ID → GroupKFold용
    # 시나리오 인덱스(0~9999)를 group으로 사용
    groups_arr = np.arange(len(sc_ids_train))

    # ── 1D-CNN 학습 ──
    print('\n' + '=' * 60)
    print('[3/5] 1D-CNN 학습')
    print('=' * 60)

    n_params_cnn = sum(p.numel() for p in CNN1D(n_feat).parameters())
    print(f'  CNN 파라미터: {n_params_cnn:,} ({n_params_cnn/len(sc_ids_train):.1f}× train)')

    oof_cnn, test_cnn, mae_cnn, folds_cnn = train_seq_model(
        CNN1D, 'CNN', X_train_3d, X_test_3d, y_train_3d,
        sc_ids_train, groups_arr, n_feat,
        hidden=32, dropout=0.3, lr=1e-3, weight_decay=1e-4,
        epochs=100, patience=15, batch_size=128,
    )

    # ── BiLSTM 학습 ──
    print('\n' + '=' * 60)
    print('[4/5] BiLSTM 학습')
    print('=' * 60)

    n_params_lstm = sum(p.numel() for p in BiLSTMModel(n_feat).parameters())
    print(f'  LSTM 파라미터: {n_params_lstm:,} ({n_params_lstm/len(sc_ids_train):.1f}× train)')

    oof_lstm, test_lstm, mae_lstm, folds_lstm = train_seq_model(
        BiLSTMModel, 'LSTM', X_train_3d, X_test_3d, y_train_3d,
        sc_ids_train, groups_arr, n_feat,
        hidden=16, dropout=0.3, lr=1e-3, weight_decay=1e-4,
        epochs=100, patience=15, batch_size=128,
    )

    # ── 상관 분석 ──
    print('\n' + '=' * 60)
    print('[5/5] 상관 분석')
    print('=' * 60)

    # 기존 GBDT OOF 로드 (model23 또는 model21)
    ckpt_dir = M23_CKPT if os.path.exists(os.path.join(M23_CKPT, 'lgbm_oof.npy')) else M21_CKPT
    print(f'  GBDT OOF 소스: {os.path.basename(ckpt_dir)}')

    # GBDT OOF는 (250000,) flat → (10000, 25) 변환 필요 없음
    # 시퀀스 모델 OOF를 (10000, 25) → (250000,) flat으로 변환
    oof_cnn_flat  = np.expm1(oof_cnn).flatten()   # raw space
    oof_lstm_flat = np.expm1(oof_lstm).flatten()   # raw space

    # GBDT OOF 로드 (log1p space → raw space)
    oof_lgbm = np.expm1(np.load(os.path.join(ckpt_dir, 'lgbm_oof.npy')))
    oof_tw   = np.load(os.path.join(ckpt_dir, 'tw18_oof.npy'))  # 이미 raw
    oof_cb   = np.expm1(np.load(os.path.join(ckpt_dir, 'cb_oof.npy')))
    oof_et   = np.expm1(np.load(os.path.join(ckpt_dir, 'et_oof.npy')))
    oof_rf   = np.expm1(np.load(os.path.join(ckpt_dir, 'rf_oof.npy')))

    # 시퀀스 OOF 순서 맞추기:
    # GBDT OOF는 원본 train 순서, 시퀀스 OOF는 scenario_id 정렬 순서
    # → 원본 순서로 복원
    train_sorted = train.sort_values(['scenario_id', 'ID']).reset_index(drop=True)
    orig_order = train.sort_values(['scenario_id', 'ID']).index

    # 실제로 시퀀스 데이터는 scenario_id 정렬 후 ts_idx 순서이므로,
    # train_sorted의 순서와 일치. 원본 train 순서로 복원
    train['_orig_idx'] = range(len(train))
    train_s = train.sort_values(['scenario_id', 'ID'])
    reorder_idx = train_s['_orig_idx'].values

    oof_cnn_reordered  = np.zeros(len(train), dtype=np.float32)
    oof_lstm_reordered = np.zeros(len(train), dtype=np.float32)
    oof_cnn_reordered[reorder_idx]  = oof_cnn_flat
    oof_lstm_reordered[reorder_idx] = oof_lstm_flat

    y_raw_flat = train['avg_delay_minutes_next_30m'].values

    corr_matrix = analyze_correlations(
        oof_cnn_reordered, oof_lstm_reordered,
        oof_lgbm, oof_tw, oof_cb, oof_et, oof_rf,
        y_raw_flat
    )

    # ── 체크포인트 저장 ──
    # OOF: log1p space, (n_scenarios, 25) 형태 저장
    np.save(os.path.join(CKPT_DIR, 'cnn_oof_3d.npy'), oof_cnn)
    np.save(os.path.join(CKPT_DIR, 'cnn_test_3d.npy'), test_cnn)
    np.save(os.path.join(CKPT_DIR, 'lstm_oof_3d.npy'), oof_lstm)
    np.save(os.path.join(CKPT_DIR, 'lstm_test_3d.npy'), test_lstm)

    # flat OOF (원본 순서, raw space) — 메타 학습기용
    np.save(os.path.join(CKPT_DIR, 'cnn_oof_flat.npy'), oof_cnn_reordered)
    np.save(os.path.join(CKPT_DIR, 'lstm_oof_flat.npy'), oof_lstm_reordered)

    # test도 flat 변환 (scenario_id 정렬 → 원본 ID 순서)
    test_cnn_flat  = np.expm1(test_cnn).flatten()
    test_lstm_flat = np.expm1(test_lstm).flatten()

    test['_orig_idx'] = range(len(test))
    test_s = test.sort_values(['scenario_id', 'ID'])
    test_reorder = test_s['_orig_idx'].values

    test_cnn_reordered = np.zeros(len(test), dtype=np.float32)
    test_lstm_reordered = np.zeros(len(test), dtype=np.float32)
    test_cnn_reordered[test_reorder] = test_cnn_flat
    test_lstm_reordered[test_reorder] = test_lstm_flat

    np.save(os.path.join(CKPT_DIR, 'cnn_test_flat.npy'), test_cnn_reordered)
    np.save(os.path.join(CKPT_DIR, 'lstm_test_flat.npy'), test_lstm_reordered)

    # 결과 JSON 저장
    results = {
        'cnn_mae': float(mae_cnn),
        'lstm_mae': float(mae_lstm),
        'cnn_fold_maes': [float(m) for m in folds_cnn],
        'lstm_fold_maes': [float(m) for m in folds_lstm],
        'cnn_lgbm_corr': float(corr_matrix[('CNN', 'LGBM')]),
        'lstm_lgbm_corr': float(corr_matrix[('LSTM', 'LGBM')]),
        'cnn_lstm_corr': float(corr_matrix[('CNN', 'LSTM')]),
        'cnn_tw_corr': float(corr_matrix[('CNN', 'TW')]),
        'lstm_tw_corr': float(corr_matrix[('LSTM', 'TW')]),
        'phase2_gate': bool(
            corr_matrix[('CNN', 'LGBM')] < 0.95 or
            corr_matrix[('LSTM', 'LGBM')] < 0.95
        ),
        'n_feat': n_feat,
        'device': DEVICE,
    }

    with open(os.path.join(CKPT_DIR, 'model26_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── 최종 보고 ──
    elapsed = time.time() - t0
    print('\n' + '=' * 60)
    print('최종 결과')
    print('=' * 60)
    print(f'  CNN  OOF MAE: {mae_cnn:.4f}')
    print(f'  LSTM OOF MAE: {mae_lstm:.4f}')
    print(f'  CNN-LGBM 상관:  {corr_matrix[("CNN", "LGBM")]:.4f}')
    print(f'  LSTM-LGBM 상관: {corr_matrix[("LSTM", "LGBM")]:.4f}')
    print(f'  CNN-LSTM 상관:  {corr_matrix[("CNN", "LSTM")]:.4f}')
    print(f'  Phase 2 Gate:  {"✅ PASS" if results["phase2_gate"] else "❌ FAIL"}')
    print(f'  소요 시간: {elapsed/60:.1f}분')
    print(f'  체크포인트: {CKPT_DIR}')

    if results['phase2_gate']:
        print(f'\n  → Phase 2 진행: 7모델 하이브리드 스태킹 (5 GBDT + CNN + LSTM)')
        print(f'     다음 실행: run_exp_model27_hybrid_stacking.py')
    else:
        print(f'\n  → 전략 A 복귀: 시나리오 형상 피처 (slope/skew/변곡점)')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()
