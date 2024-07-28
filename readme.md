# 天津市小客车调控管理系统 - 个人账户定时登录脚本

一个自(lian)用(shou)的登录脚本，该脚本会每天定时登录天津市的[摇号网站](http://xkctk.jtys.tj.gov.cn/)，获取车牌摇号信息，并通过[Server酱](http://sc.ftqq.com/3.version)接口发送摇号状态通知到微信，用来提醒摇号状态；避免错过摇到的车牌。

验证码识别部分用到的模型是简单的CNN + BiLSTM + CTC。
模型训练脚本CTC_Training.ipynb, 使用Google Colab在1000张标注的验证码（标注的验证码这里就不公开了）上训练，验证码标注由某打码平台完成。训练后模型文件存储在ctc_best.h5中。模型比较小，推断并不需要GPU。


== Updated in 2024.07 ==

**注意： 因为许久没有维护，牌照调控官网的二维码已经更新过，需要重新训练模型。**

早期验证码示例：[](./doc/old_cap.png)

当前验证码示例：[](./doc/new_cap.png)

===

## 如何开始

<del>0. 首先你需要像我一样在进行牌照摇号，所以你会有调控管理系统上注册的手机号和密码; 你还需要一个Server酱的```SCKEY```和要发送到的微信号, 在他们的[网站](http://sc.ftqq.com/3.version)上可以得到```SCKEY```和绑定账号。</del>

== Updated in 2024.07 ==  
**TODO： 通知部分需要重新实现**
1. 创建一个Python环境（比如说conda），因为模型代码实现的非常古早，目前需要Python 3.6环境。同时，你需要selenium, 对应的浏览器，和driver(比如Chrome，driver请参见 [这个页面](https://googlechromelabs.github.io/chrome-for-testing/) 下载).
2. 在第一步的创建的环境内安装本包
```bash
cd Tianjin-Vehicle-License-System-Logging-Script/
# "-e" enable code editing after installation
pip install -e . 
```
3. 参见 `config_example.yaml`, 修改配置
4. 运行
```
python ./vehicle_license_checker/vehicle_status_check.py --config ./config.yaml
```
(因为模型失效，目前暂时需要手动填入验证码。)  
5. 运行成功以后应该能看到:

[img](./doc/result.png)

对应网站截图：
[img](./doc/official.png)

===