#!/bin/sh

oad_out_path="../../weights/OAD/"
uow_out_path="../../weights/UOW/"

mkdir -p $oad_out_path
mkdir -p $uow_out_path

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2 -P $oad_out_path
  rm -rf /tmp/cookies.txt
}

gdrive_download 1n20frXNNvP5c0L822g7sCv2PPlDWnr8Q $oad_out_path$'oad_model.pt'
gdrive_download 1nq95NnZUe0CmpoFr1W9zjIT7BIy63cPO $uow_out_path$'uow_model.pt'

read -n 1 -s -r -p "Weights download complete. Press any key to exit."