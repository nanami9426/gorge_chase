#!/bin/bash
set -e

if command -v sudo >/dev/null 2>&1; then
    sudo yum install -y git
else
    yum install -y git
fi

git config --global user.name "zws"
git config --global user.email "zhangwenshuai18@gmail.com"

echo "Git 安装完成，当前配置如下："
git --version
git config --global --get user.name
git config --global --get user.email