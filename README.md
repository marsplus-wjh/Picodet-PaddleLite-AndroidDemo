# Picodet-PaddleLite-AndroidDemo
本项目使用百度自研 PP-PicoDet 模型，通过 PaddleLite 部署方式，将 Paddle Detection 中原 Android APP 的 Demo 应用更换为 PicoDet 模型，并修改了前后处理模块。

如需更换模型可参考步骤：
1、android_demo/build.gradle archives 列表中添加模型下载路径
2、android_demo/app/src/main/res/values/string.xml 将 MODEL_DIR_DEFAULT 字段值改为新模型路径
也可直接下载好模型后将模型放入 android_demo/app/src/main/assets/models，按照步骤二更换模型路径即可
*更换新模型后，或需更换 android_demo/app/src/main/cpp/Pipeline.cc 中的前、后处理方法
*以下为 PP-PicoDet small 416*416 下载路径：
  https://paddledet.bj.bcebos.com/deploy/paddlelite/picodet_s_416.tar.gz
