#!/bin/bash
export LANG=pt_BR.UTF-8
export LANGUAGE=pt_BR:pt
export LC_ALL=pt_BR.UTF-8

locale-gen pt_BR.UTF-8

exec "$@"