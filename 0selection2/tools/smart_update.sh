#!/usr/bin/env bash
# ARIS Smart Skill Update
# Intelligently compares local skills with upstream, detects personal
# customizations, and recommends safe update strategy per skill.
#
# Usage:
#   bash tools/smart_update.sh [--apply]
#   --apply: actually perform the updates (default: dry-run analysis only)

set -euo pipefail

UPSTREAM_DIR="$(cd "$(dirname "$0")/.." && pwd)/skills"
LOCAL_DIR="${HOME}/.claude/skills"
APPLY=false

if [[ "${1:-}" == "--apply" ]]; then
    APPLY=true
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Personal info patterns (paths, IPs, API keys, usernames, server configs)
PERSONAL_PATTERNS=(
    'ssh '
    'SJTUServer'
    'rfyang'
    'yangruofeng'
    'api_key'
    'API_KEY'
    'sk-'
    'token'
    '@sjtu'
    '@gmail'
    '/home/'
    '/Users/'
    'CUDA_VISIBLE'
    'wandb_project'
    'server_ip'
    'gpu_server'
    'screen -'
    'conda activate'
    '192\.168\.'
    '10\.\d+\.'
    '122\.'
)

echo -e "${BLUE}━━━ ARIS Smart Skill Update ━━━${NC}"
echo -e "Upstream: ${UPSTREAM_DIR}"
echo -e "Local:    ${LOCAL_DIR}"
echo ""

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo -e "${RED}Local skills directory not found: ${LOCAL_DIR}${NC}"
    exit 1
fi

# Counters
NEW=0
IDENTICAL=0
SAFE_UPDATE=0
NEEDS_MERGE=0
LOCAL_ONLY=0

# Results arrays
declare -a NEW_SKILLS=()
declare -a IDENTICAL_SKILLS=()
declare -a SAFE_SKILLS=()
declare -a MERGE_SKILLS=()
declare -a LOCAL_SKILLS=()

# Check each upstream skill
for skill_dir in "$UPSTREAM_DIR"/*/; do
    skill_name=$(basename "$skill_dir")
    [[ "$skill_name" == "skills-codex" ]] && continue  # skip codex mirror
    [[ "$skill_name" == "shared-references" ]] && continue  # handled separately

    local_skill="$LOCAL_DIR/$skill_name"
    upstream_file="$skill_dir/SKILL.md"

    if [[ ! -f "$upstream_file" ]]; then
        continue
    fi

    if [[ ! -d "$local_skill" ]]; then
        # New skill — doesn't exist locally
        NEW=$((NEW + 1))
        NEW_SKILLS+=("$skill_name")
        continue
    fi

    local_file="$local_skill/SKILL.md"
    if [[ ! -f "$local_file" ]]; then
        NEW=$((NEW + 1))
        NEW_SKILLS+=("$skill_name")
        continue
    fi

    # Compare
    if diff -q "$upstream_file" "$local_file" > /dev/null 2>&1; then
        # Identical
        IDENTICAL=$((IDENTICAL + 1))
        IDENTICAL_SKILLS+=("$skill_name")
        continue
    fi

    # Different — check if local has personal info
    has_personal=false
    for pattern in "${PERSONAL_PATTERNS[@]}"; do
        # Check if the LOCAL version has lines matching personal patterns
        # that the UPSTREAM version does NOT have
        local_matches=$(grep -c "$pattern" "$local_file" 2>/dev/null || true)
        local_matches=${local_matches:-0}
        upstream_matches=$(grep -c "$pattern" "$upstream_file" 2>/dev/null || true)
        upstream_matches=${upstream_matches:-0}
        if [[ $local_matches -gt $upstream_matches ]]; then
            has_personal=true
            break
        fi
    done

    if $has_personal; then
        # Has personal customizations — needs careful merge
        NEEDS_MERGE=$((NEEDS_MERGE + 1))
        MERGE_SKILLS+=("$skill_name")
    else
        # Changed upstream, no personal info in local — safe to replace
        SAFE_UPDATE=$((SAFE_UPDATE + 1))
        SAFE_SKILLS+=("$skill_name")
    fi
done

# Check for local-only skills (not in upstream)
for skill_dir in "$LOCAL_DIR"/*/; do
    skill_name=$(basename "$skill_dir")
    upstream_skill="$UPSTREAM_DIR/$skill_name"
    if [[ ! -d "$upstream_skill" ]] && [[ "$skill_name" != "shared-references" ]]; then
        LOCAL_ONLY=$((LOCAL_ONLY + 1))
        LOCAL_SKILLS+=("$skill_name")
    fi
done

# Report
echo -e "${GREEN}✅ Identical (no action needed): ${IDENTICAL}${NC}"
for s in "${IDENTICAL_SKILLS[@]:-}"; do [[ -n "$s" ]] && echo "   $s"; done
echo ""

echo -e "${GREEN}🆕 New skills (safe to add): ${NEW}${NC}"
for s in "${NEW_SKILLS[@]:-}"; do [[ -n "$s" ]] && echo "   $s"; done
echo ""

echo -e "${BLUE}🔄 Updated upstream, no personal info (safe to replace): ${SAFE_UPDATE}${NC}"
for s in "${SAFE_SKILLS[@]:-}"; do [[ -n "$s" ]] && echo "   $s"; done
echo ""

echo -e "${YELLOW}⚠️  Updated upstream + local customizations (needs manual merge): ${NEEDS_MERGE}${NC}"
for s in "${MERGE_SKILLS[@]:-}"; do
    [[ -n "$s" ]] && echo "   $s"
    if [[ -n "$s" ]]; then
        # Show what personal patterns were found
        local_file="$LOCAL_DIR/$s/SKILL.md"
        for pattern in "${PERSONAL_PATTERNS[@]}"; do
            match=$(grep -n "$pattern" "$local_file" 2>/dev/null | head -1)
            if [[ -n "$match" ]]; then
                echo -e "     ${YELLOW}→ contains: ${match}${NC}"
                break
            fi
        done
    fi
done
echo ""

echo -e "${NC}📦 Local-only skills (yours, not in upstream): ${LOCAL_ONLY}"
for s in "${LOCAL_SKILLS[@]:-}"; do [[ -n "$s" ]] && echo "   $s"; done
echo ""

# Summary
TOTAL=$((NEW + IDENTICAL + SAFE_UPDATE + NEEDS_MERGE))
echo -e "${BLUE}━━━ Summary ━━━${NC}"
echo -e "Total upstream skills: $TOTAL"
echo -e "  ${GREEN}Up to date:  $IDENTICAL${NC}"
echo -e "  ${GREEN}New to add:  $NEW${NC}"
echo -e "  ${BLUE}Safe update: $SAFE_UPDATE${NC}"
echo -e "  ${YELLOW}Need merge:  $NEEDS_MERGE${NC}"
echo -e "  Local only: $LOCAL_ONLY"
echo ""

if $APPLY; then
    echo -e "${BLUE}Applying safe updates...${NC}"

    # Add new skills
    for s in "${NEW_SKILLS[@]:-}"; do
        if [[ -n "$s" ]]; then
            cp -r "$UPSTREAM_DIR/$s" "$LOCAL_DIR/"
            echo -e "  ${GREEN}+ Added: $s${NC}"
        fi
    done

    # Replace safely updated skills
    for s in "${SAFE_SKILLS[@]:-}"; do
        if [[ -n "$s" ]]; then
            cp -r "$UPSTREAM_DIR/$s" "$LOCAL_DIR/"
            echo -e "  ${BLUE}↑ Updated: $s${NC}"
        fi
    done

    # Update shared-references
    if [[ -d "$UPSTREAM_DIR/shared-references" ]]; then
        cp -r "$UPSTREAM_DIR/shared-references" "$LOCAL_DIR/"
        echo -e "  ${BLUE}↑ Updated: shared-references${NC}"
    fi

    echo ""
    echo -e "${GREEN}Done! $NEW new + $SAFE_UPDATE updated.${NC}"

    if [[ $NEEDS_MERGE -gt 0 ]]; then
        echo -e "${YELLOW}⚠️  $NEEDS_MERGE skills have personal customizations and were NOT updated.${NC}"
        echo -e "${YELLOW}   Review manually: ${MERGE_SKILLS[*]}${NC}"
        echo -e "${YELLOW}   Tip: diff ~/.claude/skills/<name>/SKILL.md skills/<name>/SKILL.md${NC}"
    fi
else
    echo -e "Dry run complete. Run with ${GREEN}--apply${NC} to perform updates:"
    echo -e "  ${GREEN}bash tools/smart_update.sh --apply${NC}"
fi
