"""
CSV 블렌드 실험 — model45c_q7_q85 × blend_m34bd_b60
================================================================
두 최고 예측의 가중 평균으로 다양한 블렌드 비율을 시험한다.

  A = model45c_q7_q85_cv8.4735.csv  (Public 9.8048, 현 최고)
  B = blend_m34bd_b60.csv           (Public 9.8053)

블렌드 비율: A×0.3+B×0.7 ~ A×0.7+B×0.3 (10% 단위)
추가: A×0.5+B×0.5 균등 블렌드

실행: python src/blend_q85_m34bd.py
소요 시간: < 30초 (연산만)
"""

import numpy as np
import pandas as pd
import os

_BASE   = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.join(_BASE, '..', 'submissions')

FILE_A = os.path.join(SUB_DIR, 'model45c_q7_q85_cv8.4735.csv')
FILE_B = os.path.join(SUB_DIR, 'blend_m34bd_b60.csv')


def blend_and_save(df_a, df_b, col, w_a, label):
    df_out = df_a.copy()
    df_out[col] = np.clip(w_a * df_a[col].values + (1 - w_a) * df_b[col].values, 0, None)
    path = os.path.join(SUB_DIR, f'blend_{label}.csv')
    df_out.to_csv(path, index=False)
    print(f"  w_A={w_a:.1f} | mean={df_out[col].mean():.3f} | "
          f"std={df_out[col].std():.3f} | max={df_out[col].max():.2f} → {os.path.basename(path)}")
    return path


def main():
    if not os.path.exists(FILE_A):
        print(f"파일 없음: {FILE_A}")
        return
    if not os.path.exists(FILE_B):
        print(f"파일 없음: {FILE_B}")
        return

    df_a = pd.read_csv(FILE_A)
    df_b = pd.read_csv(FILE_B)
    col  = [c for c in df_a.columns if c != 'ID'][0]

    print(f"A (q85, 9.8048): mean={df_a[col].mean():.3f} | std={df_a[col].std():.3f} | max={df_a[col].max():.2f}")
    print(f"B (m34bd, 9.8053): mean={df_b[col].mean():.3f} | std={df_b[col].std():.3f} | max={df_b[col].max():.2f}")
    print(f"A-B 상관: {np.corrcoef(df_a[col].values, df_b[col].values)[0,1]:.4f}")
    print()

    # 단순 통계 차이 확인
    diff = df_a[col].values - df_b[col].values
    print(f"A-B 차이: mean={diff.mean():.4f} | std={diff.std():.4f} | "
          f"max={diff.max():.2f} | min={diff.min():.2f}")
    print(f"A>B 비율: {(diff > 0).mean():.3f} | A<B 비율: {(diff < 0).mean():.3f}")
    print()

    print("블렌드 파일 생성:")
    weights_a = [0.3, 0.4, 0.5, 0.6, 0.7]
    for w_a in weights_a:
        w_b = 1 - w_a
        label = f'q85_m34bd_a{int(w_a*10)}b{int(w_b*10)}'
        blend_and_save(df_a, df_b, col, w_a, label)

    print(f"\n총 {len(weights_a)}개 블렌드 파일 생성 완료")
    print(f"기대 Public: 두 파일이 상관 >0.99이면 블렌드 효과 미미,")
    print(f"             상관 <0.98이면 0.5~1.0점 개선 가능성 있음")


if __name__ == '__main__':
    main()
