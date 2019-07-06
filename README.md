# はじめに

このリポジトリは、[pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3)と[yoloface](https://github.com/sthanhng/yoloface)を元に作成しました。  
このリポジトリをcloneして、実行してもらえればすぐに動くようにしました。  
(ライブラリ等のインストールは必要だが、それ以外に外部からデータを学習させたりする必要がないということ)  

学習しているものは、yoloのデフォルトのやつと、顔認識専用のやつです。  
顔認識の学習データセットは[WIDER FACE](http://shuoyang1213.me/WIDERFACE/)を用いています。  

後々コメントを詳細に書いていきます。  

# 環境

 * Ubuntu18.04LTS
 * Python3
 * Opencv
 * Pytorch
 ※GPU(cuda)対応

 # フォルダ構成
 - README.md
 - video_demo.py
 - cfg      ←形式を記述しよう〜
 - weights　←自分で学習したやつあるならここに重みおいてね〜
 - input    ←ここに動画をいれてってね〜

# 実行方法