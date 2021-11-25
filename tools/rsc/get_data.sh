#!/bin/sh

oad_out_path="../../data/OAD/"
uow_out_path="../../data/UOW/"

mkdir -p $oad_out_path
mkdir -p $uow_out_path

function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2 -P $oad_out_path
  rm -rf /tmp/cookies.txt
}

gdrive_download 17nMqNGPmy7U8pl-3Zm9Q_EYsf7huiI7M $oad_out_path$'oad_train_data.npy'
gdrive_download 1zi0qA7awtVUi7MSAHxUG-kwlQoeuOXxL $oad_out_path$'oad_train_label.pkl'
gdrive_download 1L4pI0g0OaXl3OXF0OEEhezINymf4CIX6 $oad_out_path$'oad_val_data.npy'
gdrive_download 19ZJimtj0pDv6-kCXvEjyZVkvcmGI2tPe $oad_out_path$'oad_val_label.pkl'
gdrive_download 1q3jggKoMGODGFsa4APc9AaT7X7uOq0r0 $oad_out_path$'oad_test_data.npy'
gdrive_download 1ammuDRBqMUdulYmbPblppj7vpXkd3hml $oad_out_path$'oad_test_label.pkl'

gdrive_download 1MwaNM6oXTh5HKN4Su6A2GVN0D8XPQ6Al $uow_out_path$'uow_train_data.npy'
gdrive_download 11t0z4cG1p0VP7k15yNs_sGpUjG8DXrTj $uow_out_path$'uow_train_label.pkl'
gdrive_download 1EBvwopBgApnFtiRtx0-Yrnw6jV8NmhoC $uow_out_path$'uow_val_data.npy'
gdrive_download 1jpyOVvXxfbLWCkzNsmJi5jZHi_jFvMKT $uow_out_path$'uow_val_label.pkl'
gdrive_download 1r_Ps7MGME8q4tnjSTNEKOqNJxcgwR-jh $uow_out_path$'uow_test_data.npy'
gdrive_download 1fndB3axAIIPnIBwX7Lyhk5eMfbKF2fvT $uow_out_path$'uow_test_label.pkl'

read -n 1 -s -r -p "Data download complete. Press any key to exit."