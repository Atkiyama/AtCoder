#!/bin/bash

# 引数の数が正しいか確認
if [ $# -ne 1 ]; then
    echo "使用法: $0 <ディレクトリ名>"
    exit 1
fi

# 引数からディレクトリ名を取得
dir_name="$1"

# ディレクトリを作成
mkdir "$dir_name"

# ディレクトリが正常に作成されたか確認
if [ $? -eq 0 ]; then
    echo "ディレクトリ $dir_name が正常に作成されました。"
else
    echo "ディレクトリの作成中にエラーが発生しました。"
    exit 1
fi

# main.pyからa.py、b.py、c.py、d.pyをコピー
cp "main.py" "$dir_name/$dir_name""A.py"
cp "main.py" "$dir_name/$dir_name""B.py"
cp "main.py" "$dir_name/$dir_name""C.py"
cp "main.py" "$dir_name/$dir_name""D.py"
cp "main.py" "$dir_name/$dir_name""E.py"


# ファイルが正常にコピーされたか確認
if [ $? -eq 0 ]; then
    echo "ファイル a.py, b.py, c.py, d.py が正常にコピーされました。"
else
    echo "ファイルのコピー中にエラーが発生しました。"
fi
