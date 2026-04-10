#!/bin/bash
###############################################################################
# VASU - Camera Test Script for Redmi 7A
# Tests libcamera functionality before full deployment.
###############################################################################

set -euo pipefail

echo "═══ VASU Camera Test ═══"

# Check libcamera exists
if ! command -v libcamera-hello &>/dev/null; then
    echo "ERROR: libcamera-hello not found. Install: apk add libcamera libcamera-tools"
    exit 1
fi

# List cameras
echo ""
echo "Available cameras:"
libcamera-hello --list-cameras 2>&1 || {
    echo "WARNING: No cameras detected via libcamera."
    echo "Try: dmesg | grep -i camera"
    exit 1
}

# Test rear camera capture
echo ""
echo "Testing rear camera (camera 0)..."
TEST_IMG="/tmp/vasu_cam_test_rear.jpg"
if libcamera-still -o "$TEST_IMG" --immediate --timeout 2000 --camera 0 2>/dev/null; then
    SIZE=$(du -h "$TEST_IMG" | cut -f1)
    echo "  Rear camera: OK ($SIZE)"
    # Check image dimensions
    python3 -c "
from PIL import Image
img = Image.open('$TEST_IMG')
print(f'  Resolution: {img.size[0]}x{img.size[1]}')
print(f'  Format: {img.format}')
" 2>/dev/null || echo "  (PIL not available for dimension check)"
else
    echo "  Rear camera: FAILED"
fi

# Test front camera capture
echo ""
echo "Testing front camera (camera 1)..."
TEST_IMG="/tmp/vasu_cam_test_front.jpg"
if libcamera-still -o "$TEST_IMG" --immediate --timeout 2000 --camera 1 2>/dev/null; then
    SIZE=$(du -h "$TEST_IMG" | cut -f1)
    echo "  Front camera: OK ($SIZE)"
else
    echo "  Front camera: FAILED or not available"
fi

# Test video capture (brief)
echo ""
echo "Testing video capture (2 seconds)..."
TEST_VID="/tmp/vasu_cam_test.h264"
if libcamera-vid -o "$TEST_VID" --timeout 2000 --camera 0 2>/dev/null; then
    SIZE=$(du -h "$TEST_VID" | cut -f1)
    echo "  Video capture: OK ($SIZE)"
else
    echo "  Video capture: FAILED"
fi

# Cleanup
rm -f /tmp/vasu_cam_test_*.{jpg,h264}

echo ""
echo "═══ Camera Test Complete ═══"
