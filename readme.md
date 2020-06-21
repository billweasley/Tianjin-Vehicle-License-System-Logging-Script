# [天津市小客车调控管理系统](http://xkctk.jtys.tj.gov.cn/)个人账户定时登录脚本

一个自(lian)用(shou)的登录脚本，该脚本会每天定时登录天津市的[摇号网站](http://xkctk.jtys.tj.gov.cn/)，获取车牌摇号信息，并通过[Server酱](http://sc.ftqq.com/3.version)接口发送摇号状态通知到微信。  

验证码识别部分用到的模型是简单的CNN + BiLSTM + CTC。
模型训练脚本CTC_Training.ipynb, 使用Google Colab在1000张标注的验证码（标注的验证码这里就不公开了）上训练，验证码标注由某打码平台完成。训练后模型文件存储在ctc_best.h5中。模型比较小，推断并不需要GPU。

## 如何开始

0. 首先你需要像我一样在进行牌照摇号，所以你会有调控管理系统上注册的手机号和密码; 你还需要一个Server酱的```SCKEY```和要发送到的微信号, 在他们的[网站](http://sc.ftqq.com/3.version)上可以得到```SCKEY```和绑定账号。

1. 打开 ```vehicle.py```, 你应该可以看到下面三行。其中```phone```是你在调控系统上的电话号码，
```password```是你在调控系统上的密码，```ftqq_id```是你的Server酱```SCKEY```。

```python
phone = ''
password = ''
ftqq_id = ''
```

2. 将上面信息填好，保存文件；同时你的微信需要关注Server酱的公众号来接受消息。

3. 如果你的服务器/本地上已经安装有```requirements.txt```的各种库，你应该可以直接运行``` python vehicle.py ```进行测试；同时可以设置Crontab来让这个脚本变成定时任务。

我在使用的时候将其封装到了Docker里面，目前没有将Docker image上传的计划。

如要向我一样在Docker内运行请执行如下操作：
- 安装并启动Docker
- 检查```Dockerfile```, 你可能需要修改该文件里的下面这一行，来让脚本定时运行在你觉得合适的时间。注意这个时间是你运行的脚本机器的本地时间。默认值为每天早上8点和下午15点脚本运行一次。
```
RUN (crontab -l ; echo "0 8,15 * * * PYTHONIOENCODING=utf-8 python3 /app/vehicle_logging/vehicle.py >> /var/log/cron.log") | crontab
```
关于如何修改定时时间，请参照crontab的[文档](https://man7.org/linux/man-pages/man5/crontab.5.html)

- 本地构建Docker image:
```bash
cd <the path where the repository are>
docker build -t <your docker image name> . 
# e.g.
# cd vehicle_logging_public
# docker build -t vehicle-logging-tf-private:latest . 
```
- 运行 Docker container
```bash
docker run -d --name <your docker container name> <your docker image name>
# e.g.
# docker run -d --name vehicle-tf vehicle-logging-tf-private
```
- 如果脚本运行正常，接下来你应该能在每天你设定的时间在你的微信看到通知，如果没有你可以检查你的docker container，下面的命令或许会有用处：
```bash
# Open bash in your container
docker exec -ti <your docker container name> bash 
# e.g. docker exec -ti vehicle-tf bash
# exit() to exit the docker container environment

# To check if the job is scheduled
docker exec -ti <your docker container name> bash -c "crontab -l"
# e.g. docker exec -ti vehicle-tf bash -c "crontab -l"

# To check if the cron service is running
docker exec -ti <your docker container name> bash -c "pgrep cron"
# e.g. docker exec -ti vehicle-tf bash -c "pgrep cron"
```