#!/usr/bin/env bash

export LC_ALL=C
apt-get update
apt-get -y install htop nano libgtk2.0-dev screen libvips
mkdir /mnt/ramdisk
mount -t tmpfs -o rw,size=8G tmpfs /mnt/ramdisk
df -h /mnt/ramdisk/