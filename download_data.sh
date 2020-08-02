#!/usr/bin/env bash

id="1P7xMf4kQaAmuuLNYXKqVjpx_244zqWIU"
curl -c cookie.tmp -s -L "https://drive.google.com/uc?export=download&id=${id}" > /dev/null
curl -Lb cookie.tmp "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' cookie.tmp`&id=${id}" -o data.zip
unzip data.zip
rm cookie.tmp data.zip