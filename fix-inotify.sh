#!/bin/bash
# Script to fix inotify watch limit issue

echo "Current inotify max_user_watches: $(cat /proc/sys/fs/inotify/max_user_watches)"
echo ""
echo "To fix this issue, you need to increase the inotify limit."
echo ""
echo "Option 1: Temporary fix (until reboot):"
echo "  sudo sysctl fs.inotify.max_user_watches=524288"
echo ""
echo "Option 2: Permanent fix (recommended):"
echo "  echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf"
echo "  sudo sysctl -p"
echo ""
echo "After running one of the above commands, try running your agent again:"
echo "  python myagent.py dev"
