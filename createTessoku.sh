#!/bin/bash

# 引数の数が正しいか確認
if [ $# -ne 2 ]; then
    echo "使用法: $0 <ファイル名>"
    exit 1
fi

# 引数からファイル名を取得
file_type="$1"
file_number="$2"



# tessoku フォルダが存在しない場合、作成する
if [ ! -d "tessoku" ]; then
    mkdir "tessoku"
fi

# main.py をコピーしてファイルを作成
cp "main.py" "tessoku/${file_type}/${file_type}${file_number}.py"

# ファイルが正常に作成されたか確認
if [ $? -eq 0 ]; then
    echo "ファイル tessoku/${file_type}${file_number}å が正常に作成されました。"
else
    echo "ファイルの作成中にエラーが発生しました。"
fi
