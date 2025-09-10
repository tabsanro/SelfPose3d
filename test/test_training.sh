#!/bin/bash
set -e  # 에러 발생 시 즉시 종료

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 아이콘 정의
ROCKET="🚀"
GEAR="⚙️"
CHECKMARK="✅"
CLOCK="⏰"
FIRE="🔥"
STAR="⭐"

# 설정 파일 배열
CONFIGS=(
    "test/test_configs/2D_heatmap.yaml"
    "test/test_configs/3D_rootnet.yaml"
    "test/test_configs/3D_posenet.yaml"
)

# 함수 정의
print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    ${WHITE}${FIRE} SelfPose3D 학습 테스트 ${FIRE}${CYAN}                    ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_config_info() {
    local config=$1
    local current=$2
    local total=$3
    local config_name=$(basename "$config" .yaml)
    
    echo -e "${PURPLE}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${PURPLE}│ ${WHITE}${GEAR} 설정 파일: ${YELLOW}${config_name}${WHITE} (${current}/${total})${PURPLE}                    │${NC}"
    echo -e "${PURPLE}│ ${WHITE}${CLOCK} 시작 시간: ${CYAN}$(date '+%Y-%m-%d %H:%M:%S')${PURPLE}                      │${NC}"
    echo -e "${PURPLE}└─────────────────────────────────────────────────────────┘${NC}"
}

print_progress_bar() {
    local current=$1
    local total=$2
    local progress=$((current * 50 / total))
    local bar=""
    
    for ((i=0; i<progress; i++)); do
        bar+="█"
    done
    for ((i=progress; i<50; i++)); do
        bar+="░"
    done
    
    echo -e "${WHITE}진행률: [${GREEN}${bar}${WHITE}] ${current}/${total} (${GREEN}$((current * 100 / total))%${WHITE})${NC}"
}

print_success() {
    local config=$1
    local duration=$2
    local config_name=$(basename "$config" .yaml)
    
    echo -e "${GREEN}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${GREEN}│ ${CHECKMARK} ${WHITE}완료: ${YELLOW}${config_name}${WHITE} - 소요시간: ${CYAN}${duration}초${GREEN}              │${NC}"
    echo -e "${GREEN}└─────────────────────────────────────────────────────────┘${NC}"
    echo ""
}

print_final_summary() {
    local total_time=$1
    local total_configs=${#CONFIGS[@]}
    
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    ${WHITE}${STAR} 테스트 완료 요약 ${STAR}${CYAN}                      ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║ ${WHITE}총 설정 파일: ${GREEN}${total_configs}개${CYAN}                                        ║${NC}"
    echo -e "${CYAN}║ ${WHITE}총 소요 시간: ${GREEN}${total_time}초${CYAN}                                      ║${NC}"
    echo -e "${CYAN}║ ${WHITE}상태: ${GREEN}모든 테스트 성공! ${CHECKMARK}${CYAN}                             ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
}

# 메인 실행
print_header

# 전체 시작 시간
TOTAL_START_TIME=$(date +%s)

# 각 설정 파일에 대해 학습 실행
for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[i]}"
    CURRENT=$((i + 1))
    TOTAL=${#CONFIGS[@]}
    
    # 현재 설정 정보 출력
    print_config_info "$CFG" "$CURRENT" "$TOTAL"
    
    # 진행률 표시
    print_progress_bar "$CURRENT" "$TOTAL"
    echo ""
    
    # 개별 시작 시간
    START_TIME=$(date +%s)
    
    # 학습 실행
    echo -e "${WHITE}${ROCKET} 학습 시작...${NC}"
    python test/test_tools/train_3d.py --cfg "$CFG"
    
    # 개별 종료 시간 및 소요시간 계산
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 성공 메시지 출력
    print_success "$CFG" "$DURATION"
done

# 전체 종료 시간 및 소요시간 계산
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

# 최종 요약 출력
print_final_summary "$TOTAL_DURATION"