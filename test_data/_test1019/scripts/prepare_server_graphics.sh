#!/usr/bin/env bash
# Rebuild the tested Ubuntu 20.04 graphics runtime after replacing this instance.
# Use a task-local source list to avoid the image's retired Debian buster source.
set -euo pipefail
TEST_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /etc/os-release
[[ "$ID" == ubuntu && "$VERSION_ID" == 20.04 ]] || {
    echo 'This tested graphics setup expects Ubuntu 20.04.' >&2; exit 1;
}
cat > "$TEST_ROOT/cache/ubuntu-graphics.sources.list" <<'SOURCES'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu focal main universe restricted multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu focal-updates main universe restricted multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu focal-security main universe restricted multiverse
SOURCES
opts=(-o "Dir::Etc::sourcelist=$TEST_ROOT/cache/ubuntu-graphics.sources.list" -o 'Dir::Etc::sourceparts=-')
sudo -n apt-get "${opts[@]}" update
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get "${opts[@]}" install -y --no-install-recommends libegl1 libegl-mesa0 libgles2 libosmesa6 libgbm1
