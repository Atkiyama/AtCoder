#!/bin/bash

# 引数の数が正しいか確認
if [ $# -ne 1 ]; then
    echo "使用法: $0 <ファイル番号>"
    exit 1
fi

# 引数からファイル番号を取得
file_number="$1"

# 引数の数字を三桁にフォーマット
formatted_number=$(printf "%03d" "$file_number")

# kyopro_educational_90 フォルダが存在しない場合、作成する
if [ ! -d "kyopro_educational_90" ]; then
    mkdir "kyopro_educational_90"
fi

# main.py をコピーしてファイルを作成
cp "main.py" "kyopro_educational_90/${formatted_number}.py"

# ファイルが正常に作成されたか確認
if [ $? -eq 0 ]; then
    echo "ファイル kyopro_educational_90/${formatted_number}.py が正常に作成されました。"
else
    echo "ファイルの作成中にエラーが発生しました。"
fi
