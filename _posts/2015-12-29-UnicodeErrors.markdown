---
layout: post
title: UnicodeErrors in Python 
---
>### 什么是[Unicode](https://en.wikipedia.org/wiki/Unicode)
计算机能看懂文字只有0和1，而这些01数字串对我们是没有意义的，我们需要语言文字以及各种字符表情进行交流。这就需要一个标准将计算机语言和我们的语言统一起来。于是ASCII编码应运而生，之后又出现了ISO 8891-1、ISO Latin 1等编码。但是这些编码有一个缺陷，就是表示字符个数有限制，比如ASCII最多只能表示128个字符，而世上语言千千万，显然是不够的。为了解决这个矛盾，上个世纪80年代提出了[Unicode](https://en.wikipedia.org/wiki/Unicode)编码，能够提供110多万的字符空间，UTF-8是最常用的Unicode字符集。  

>### Str vs Unicode in Python 2
Python 2有两种string类型，**str**和**unicode**。分别是以byte和unicode的形式存储string。
![Two Types of Strings](/media/images/20151229/strunicode.png)
byte类型可以通过*.decode(encoding)*方法转化为unicode，unicode则可以通过*.encode(encoding)*方法转化为byte。**默认的*encoding*一般都是*ascii***。

>#### Implicit Conversion
在python 2中，结合一个byte string和一个unicode string时，python会自动把byte string解码，解码是按照默认的encoding进行，也就是**ascii**。如果不幸的是这个byte string不能被**ascii**解码，就会报错UnicodeDecodeError。看下面的例子：
![UnicodeDecodeError Example](/media/images/20151229/unicodedecodeerror.png)
这就是python 2中UnicodeErrors错误的一个重要原因，它想让我们从两种类型的繁琐转换中解脱出来，提供一种默认的自动转换，不过事与愿违，很多情况下，我们需要直接去面对两种string类型，手动处理。    

>### Stricter rules in Python 3
同样，Python 3中也有两种string类型，**bytes**和**str**，是不是很眼熟？没错，跟Python 2一样的，只是换了名字，2中的str和unicode分别变成了3里的bytes和str。
![two types of String in python3](/media/images/20151229/bytestr.png)
Python 3禁止bytes到unicode的自动转换，因此，
![no explicit conversion in python 3](/media/images/20151229/noconversion.png)
这种严格的规定，迫使我们必须清楚，当前处理的string到底是bytes还是str？否则会立即报错。  
 

>### [Pro tips] [1]
1. 接收来的bytes数据立即转换成unicode，整个程序内部都使用unicode，输出的时候再转换成bytes。
2. 明确正在处理文本的类型，以及bytes数据的encoding。
3. 测试对Unicode的支持。 


[1]: http://nedbatchelder.com/text/unipain.html "reference"
