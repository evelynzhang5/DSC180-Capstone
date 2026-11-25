#!/bin/bash
# Clone the NA-RFM paper code and Training-Free-Guidance (TFG) at the commits used in Quarter 1, into downloads/.
# steering/sd_dof/sd_dof.py imports the RFM training code from downloads/narfm/na-rfm.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
W=$ROOT/downloads/narfm
mkdir -p "$W"
[ -d "$W/na-rfm" ] || git clone -q https://github.com/isotrivial/na-rfm.git "$W/na-rfm"
git -C "$W/na-rfm" checkout -q 8e9619bc6735b03d73056bfdc13dd3659b96e127
[ -d "$W/Training-Free-Guidance" ] || git clone -q https://github.com/YWolfeee/Training-Free-Guidance.git "$W/Training-Free-Guidance"
git -C "$W/Training-Free-Guidance" checkout -q f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77
echo "na-rfm and TFG ready in $W"
