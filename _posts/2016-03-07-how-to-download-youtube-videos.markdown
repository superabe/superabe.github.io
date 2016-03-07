---
layout: post
title: "Youtube上的视频如何下载？"
tags: Youtube
categories: 杂记
---
- Youtube是google旗下的一个视频网站，上面有海量的视频资源，不过可惜的是在天朝已被墙多年,作为一个科研狗，这就错过许多实用的教程、实验protocal、优秀大学的公开课程blablabla。。。这是一个多么悲伤的故事。但是，幸运的你如果懂得如何[翻墙](https://blog.phpgao.com/shadowsocks_on_linux.html)（世界这么大，墙外的更美好！），那么，我们就来了解一下，如何去下载Youtube上的视频，毕竟以国内这么渣的网速，还是下载到本地看视频的好。  

## <font color='green'>1. 简便类</font>
- 有许多免费的网页下载可以使用，只需要把视频的地址粘贴一下，一键下载。
- 第一个网站：[**savefrom.net**](http://en.savefrom.net/)，这个网站提供了比较简单的视频下载，可以选择视频的清晰度，如果你是土豪，流量无限，一定要选择4k。。。
![savefrom.net](/media/images/20160307/savefrom.png)


- 第二个网站：[**keepvid.com**](http://keepvid.com/)，这个网站要更厉害一些，提供视频和声音的单独下载，视频大小选择更丰富，如果运气比较好，还能下载字幕。
![keepvid.com](/media/images/20160307/keepvid.png)  


## <font color='green'>2. Geek类</font>
如果你是一个无尽折腾的青年，熟悉命令行，那么[youtube-dl](https://rg3.github.io/youtube-dl/)是你的理想选择。这个工具的运行需要**python**，下载[在这里](https://www.python.org/downloads/)。

### Windows用户
有windows的[安装程序](https://yt-dl.org/downloads/2016.03.06/youtube-dl.exe)（2016.03.06版本）。

### Unix用户（Linux, OS X, etc.）
**用curl或者wget, whatever you like.**  
 
`sudo curl https://yt-dl.org/downloads/2016.03.06/youtube-dl -o /usr/local/bin/youtube-dl`  
`sudo chmod a+rx /usr/local/bin/youtube-dl`  

**使用pip**  
`sudo pip install youtube-dl`  

**OS X用户还可以使用brew**   
`brew install youtube-dl`  
  

### 使用方法简介      
[在这里](https://github.com/rg3/youtube-dl/blob/master/README.md#readme)有详尽的使用方法，我只是简单介绍一下，有兴趣的自己去研究～  

**最简单的应用**  
`youtube-dl url`

例如下载上面图片里的视频：   
`youtube-dl https://www.youtube.com/watch?v=Kpoo6M3S9E8`

**使用代理（proxy）**    
`youtube-dl --proxy YOUR_PROXY video_url`

**视频选择**    
如果要下载的视频在一个播放列表里面，可以选择从第一个下载到最后一个（默认），也可以自定某一个视频或者某一些，甚至连片头的广告也可以下载。。。  
`youtube-dl --playlist-start 1 --playlist-end 5 --match-title VIDEO_TITLE --include-ads video_url`  

**设置下载最大速度**  
`youtube-dl -r 500k video_url`  

**把所有要下载的视频地址都存储到一个文档urls.txt里面**     
`youtube-dl -a urls.txt`  
就可以把文件里所有视频都下载下来啦  

**下载字幕**   
`youtube-dl --write-sub video_url`  

是不是很酷，它的功能不止这些，还可以设置下载视频的命名、选择下载视频格式、不同的分辨率、仅下载音频，甚至在下载之后自动转换格式（需要ffmpeg）！等等等等。是不是很强大？

就写这么多吧，欢迎拍砖～～～

 
