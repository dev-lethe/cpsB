# 手順説明
前期のC言語でつかったUbuntuのターミナルを使うのがよいと思われ

1. フォルダのインストール
    git cloneでインストールしちゃおう
    ```
    git clone https://github.com/dev-lethe/cpsB
    ```
2. 仮想環境の作成
    インストールしたフォルダの中に入ったら，プログラムを回すためにいろいろ入れちゃおう
    ```
    cd cps
    python3 -m venv cpsenv
    source cpsenv/bin/activate
    pip3 install -r requirements.txt
    ```
3. コードの修正
    img2featはnumpyのバージョンが古いときに作られたぽくてエラーの原因になるのでちょっと修正してみます．
    もっといい方法があるかもしれないけどあくまで一例
    ```
    cd cpsenv/lib/python3.10/site-packages/img2feat
    code antbee.py
    ```
    などでantbee.pyを開きます．
    Ctrl+fなどで文字列置換をします．
    ```
    np.int -> np.int32
    ```
    とすればエラーがなくなります．
4. プログラムを回す
    もう回せちゃうから順番に試しちゃう
    - Linear.py
        線形回帰による分類
    - Logistic.py
        ロジスティク回帰による分類
    - kNN.py
        k近傍法による分類
        default k=3
    - NN.py
        NNによる分類
        精度はよくないので，改良必須
    
