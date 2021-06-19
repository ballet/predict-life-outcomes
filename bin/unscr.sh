#!/usr/bin/env bash

export $(base64 -d -i - <<<"${ZZSCR}")
