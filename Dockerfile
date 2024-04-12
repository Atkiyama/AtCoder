# ベースイメージ
FROM python:latest

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y git

# 競技プログラミング用のPythonライブラリのインストール
RUN pip install numpy scipy matplotlib networkx

# コンテナ内にAtCoderフォルダを作成し、そこに移動
WORKDIR /AtCoder

# ホストのAtCoderフォルダをコンテナ内のAtCoderフォルダにコピー
COPY ./ /AtCoder/

# デフォルトのコマンドを/bin/bashに設定
CMD ["/bin/bash"]
