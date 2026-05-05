#!/usr/bin/env bash
# =============================================================================
# git_push.sh — Smart-Warehouse-Delay-Prediction 포트폴리오 GitHub 업로드 자동화
# 실행: bash git_push.sh
# =============================================================================

set -euo pipefail

# ── 색상 정의 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅  $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️   $*${NC}"; }
err()  { echo -e "${RED}❌  $*${NC}"; }
info() { echo -e "${CYAN}ℹ️   $*${NC}"; }
sep()  { echo -e "${BOLD}──────────────────────────────────────────${NC}"; }

# ── 프로젝트 루트로 이동 ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "작업 디렉터리: $SCRIPT_DIR"
sep

# =============================================================================
# Step 0. 상태 확인
# =============================================================================
echo -e "${BOLD}[Step 0] 현재 브랜치 & 원격 저장소 확인${NC}"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
REMOTE=$(git remote get-url origin 2>/dev/null || echo "(없음)")
info "브랜치 : $BRANCH"
info "원격   : $REMOTE"
sep

# =============================================================================
# Step 1. 내부 파일 추적 해제 (git rm --cached)
# =============================================================================
echo -e "${BOLD}[Step 1] 내부 작업 파일 추적 해제${NC}"

INTERNAL_FILES=(
    "CLAUDE.md"
    "docs/COWORK_GUIDELINES.md"
    "docs/github_upload_checklist.md"
)

for f in "${INTERNAL_FILES[@]}"; do
    if git ls-files --error-unmatch "$f" &>/dev/null 2>&1; then
        git rm --cached "$f"
        ok "$f — 추적 해제 완료"
    else
        info "$f — 이미 추적 해제됨 (건너뜀)"
    fi
done
sep

# =============================================================================
# Step 2. 항상 업데이트할 파일 추가
# =============================================================================
echo -e "${BOLD}[Step 2] 핵심 문서 & .gitignore 추가${NC}"

ALWAYS_FILES=(
    ".gitignore"
    "README.md"
    "docs/v6_strategy.md"
)

for f in "${ALWAYS_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        git add "$f"
        ok "$f"
    else
        warn "$f — 파일 없음, 건너뜀"
    fi
done
sep

# =============================================================================
# Step 3. 핵심 src 파일 추가 (⭐⭐⭐)
# =============================================================================
echo -e "${BOLD}[Step 3] 핵심 실험 스크립트 추가 (⭐⭐⭐)${NC}"

CORE_FILES=(
    "src/run_exp_model27_hybrid_stacking.py"
    "src/analysis_model28A_axis3.py"
    "src/run_exp_v3_model28A_layout_robust.py"
    "src/run_exp_v3_model29A_ratio_expand.py"
    "src/run_exp_v3_model30_combined.py"
    "src/run_model33_asymmetric.py"
    "src/run_model34_loss_opt.py"
    "src/blend_m33_m34.py"
    "src/run_model41_traj_fe.py"
)

CORE_MISSING=0
for f in "${CORE_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        git add "$f"
        ok "$f"
    else
        err "$f — 파일 없음!"
        CORE_MISSING=$((CORE_MISSING + 1))
    fi
done

if [[ $CORE_MISSING -gt 0 ]]; then
    warn "핵심 파일 ${CORE_MISSING}개 누락. 계속 진행하려면 엔터, 중단은 Ctrl+C"
    read -r
fi
sep

# =============================================================================
# Step 4. 분석/탐색 src 파일 추가 (⭐⭐)
# =============================================================================
echo -e "${BOLD}[Step 4] 분석/탐색 스크립트 추가 (⭐⭐)${NC}"

ANALYSIS_FILES=(
    "src/eda_tail_driver.py"
    "src/eda_loss_ablation.py"
    "src/run_exp_v3_model29B_optuna_retune.py"
    "src/run_v4_extreme_2stage.py"
)

for f in "${ANALYSIS_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        git add "$f"
        ok "$f"
    else
        warn "$f — 파일 없음, 건너뜀"
    fi
done
sep

# =============================================================================
# Step 5. (선택) v5 ablation 기록 파일 추가 (⭐)
# =============================================================================
echo -e "${BOLD}[Step 5] v5 ablation 기록 파일 추가 (선택)${NC}"

OPTIONAL_FILES=(
    "src/run_model37_feat_select.py"
    "src/run_model38_pseudo_label.py"
    "src/run_model39_multiseed.py"
    "src/run_model40_scenario_pp.py"
)

OPTIONAL_EXIST=0
for f in "${OPTIONAL_FILES[@]}"; do
    [[ -f "$f" ]] && OPTIONAL_EXIST=$((OPTIONAL_EXIST + 1))
done

if [[ $OPTIONAL_EXIST -gt 0 ]]; then
    echo -e "존재하는 v5 ablation 파일 ${OPTIONAL_EXIST}개를 업로드하겠습니까? [y/N] "
    read -r OPT_ANSWER
    if [[ "$OPT_ANSWER" =~ ^[Yy]$ ]]; then
        for f in "${OPTIONAL_FILES[@]}"; do
            if [[ -f "$f" ]]; then
                git add "$f"
                ok "$f"
            fi
        done
    else
        info "v5 ablation 파일 건너뜀"
    fi
else
    info "v5 ablation 파일 없음, 건너뜀"
fi
sep

# =============================================================================
# Step 6. 변경 내역 요약
# =============================================================================
echo -e "${BOLD}[Step 6] Staged 변경 내역 확인${NC}"
git diff --cached --stat || true
sep

# =============================================================================
# Step 7. 커밋
# =============================================================================
echo -e "${BOLD}[Step 7] 커밋${NC}"

STAGED=$(git diff --cached --name-only | wc -l | tr -d ' ')
if [[ "$STAGED" -eq 0 ]]; then
    warn "Staged 파일이 없습니다. 커밋할 내용이 없어 종료합니다."
    exit 0
fi

COMMIT_MSG="feat: portfolio update — final results (Public 9.8073), v2~v6 experiment scripts, updated README"
git commit -m "$COMMIT_MSG"
ok "커밋 완료: \"$COMMIT_MSG\""
sep

# =============================================================================
# Step 8. Push
# =============================================================================
echo -e "${BOLD}[Step 8] GitHub Push${NC}"
echo -e "원격 저장소 ${CYAN}$REMOTE${NC} ($BRANCH 브랜치) 로 push하겠습니까? [y/N] "
read -r PUSH_ANSWER

if [[ "$PUSH_ANSWER" =~ ^[Yy]$ ]]; then
    git push origin "$BRANCH"
    ok "Push 완료! 🚀"
    info "확인: https://github.com/0cars0903/Smart-Warehouse-Delay-Prediction"
else
    warn "Push 취소됨. 나중에 직접 실행: git push origin $BRANCH"
fi

sep
echo -e "${BOLD}${GREEN}git_push.sh 완료!${NC}"
