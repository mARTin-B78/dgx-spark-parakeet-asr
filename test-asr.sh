#!/usr/bin/env bash
# =============================================================================
# test-asr.sh — Test the Parakeet ASR endpoint
#
# Usage:
#   ./test-asr.sh                          # Test with built-in sample
#   ./test-asr.sh my-audio.wav             # Test with your file
#   ./test-asr.sh my-audio.mp3 de          # Test with language hint
# =============================================================================

set -euo pipefail

ASR_URL="${ASR_URL:-http://localhost:8020}"
FILE="${1:-}"
LANGUAGE="${2:-auto}"

# ── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${CYAN}Parakeet TDT v3 — ASR Test${NC}"
echo "─────────────────────────────────────"

# ── Step 1: Health check ─────────────────────────────────────────────────────
echo -e "\n${CYAN}1. Health check...${NC}"
HEALTH=$(curl -sf "$ASR_URL/health" 2>/dev/null || echo "FAILED")

if [[ "$HEALTH" == "FAILED" ]]; then
    echo -e "${YELLOW}  Service not reachable at $ASR_URL${NC}"
    echo "  Is the container running? Check: docker ps | grep parakeet"
    exit 1
fi

echo "  $HEALTH" | python3 -m json.tool 2>/dev/null || echo "  $HEALTH"

# ── Step 2: Check models ─────────────────────────────────────────────────────
echo -e "\n${CYAN}2. Available models...${NC}"
curl -sf "$ASR_URL/v1/models" | python3 -m json.tool 2>/dev/null || echo "  (could not fetch models)"

# ── Step 3: Generate or use test audio ────────────────────────────────────────
if [[ -z "$FILE" ]]; then
    echo -e "\n${CYAN}3. Generating test audio with sox...${NC}"

    if ! command -v sox &>/dev/null; then
        echo "  sox not installed. Generating a sine wave with ffmpeg..."

        if command -v ffmpeg &>/dev/null; then
            # Generate 3 seconds of silence + tone (just to test the pipeline)
            ffmpeg -y -f lavfi -i "sine=frequency=440:duration=3" \
                -ac 1 -ar 16000 -sample_fmt s16 \
                /tmp/parakeet-test.wav 2>/dev/null
            FILE="/tmp/parakeet-test.wav"
            echo "  Generated test tone: $FILE"
            echo -e "  ${YELLOW}(This is just a tone — the transcript will be empty or noise)${NC}"
        else
            echo ""
            echo "  No test audio available. Please provide an audio file:"
            echo "    ./test-asr.sh your-audio.wav"
            echo "    ./test-asr.sh your-audio.mp3 de    # with language hint"
            exit 1
        fi
    else
        # Use sox to generate a short silence for testing
        sox -n -r 16000 -c 1 /tmp/parakeet-test.wav trim 0 3 2>/dev/null
        FILE="/tmp/parakeet-test.wav"
        echo "  Generated 3s test audio: $FILE"
    fi
fi

if [[ ! -f "$FILE" ]]; then
    echo "  File not found: $FILE"
    exit 1
fi

FILE_SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat --printf="%s" "$FILE" 2>/dev/null || echo "?")
echo "  File: $FILE ($FILE_SIZE bytes)"

# ── Step 4: Transcribe ──────────────────────────────────────────────────────
echo -e "\n${CYAN}4. Transcribing (language=$LANGUAGE)...${NC}"

START=$(date +%s%N)

RESULT=$(curl -sf "$ASR_URL/v1/audio/transcriptions" \
    -F "file=@$FILE" \
    -F "language=$LANGUAGE" \
    -F "response_format=json" \
    2>&1)

END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))

echo ""
echo "  Response ($ELAPSED ms):"
echo "  $RESULT" | python3 -m json.tool 2>/dev/null || echo "  $RESULT"

# ── Step 5: Test verbose_json (with timestamps) ─────────────────────────────
echo -e "\n${CYAN}5. Verbose JSON (with timestamps)...${NC}"

VERBOSE=$(curl -sf "$ASR_URL/v1/audio/transcriptions" \
    -F "file=@$FILE" \
    -F "language=$LANGUAGE" \
    -F "response_format=verbose_json" \
    2>&1)

echo "  $VERBOSE" | python3 -m json.tool 2>/dev/null || echo "  $VERBOSE"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}Test complete!${NC}"
echo ""
echo "Try with real audio:"
echo "  ./test-asr.sh recording.wav"
echo "  ./test-asr.sh podcast.mp3 en"
echo "  ./test-asr.sh meeting.wav de"
echo ""
echo "Supported languages: bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk"
echo ""
