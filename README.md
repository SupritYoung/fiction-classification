# 小说文本分类任务

## 代码链接

[https://github.com/a1097304791/fiction-classification](https://github.com/a1097304791/fiction-classification)



## 数据集

数据集有从起点中文网上爬取的13个分类，每个分类20本，每本10章，共260部小说，3600章。

## 所用算法

采用支持向量机（SVM）算法，考虑使用一对多法，训练时依次把某个类别的样本归为一类,其他剩余的样本归为另一类，这样k个类别的样本就构造出了k个SVM。分类时将未知样本分类为具有最大分类函数值的那类。

## 1. 加载数据集

从文件夹读取数据，按照种类名->对应小说list，小说名->对应内容关系建立多个字典。为了方便后续的 nlp 处理，将每本小说的10章内容合成一张。

```python
import os

# 系统中每个目录都有隐藏文件夹，为避免误读所以要删除
def delete_hidden_floder(files):
    for f in files:
        if f[0]== '.': 
            files.remove(f)
    return files

# 加载所有小说
genre_datas = os.listdir('起点小说')
# 删除隐藏文件夹
genre_datas =  delete_hidden_floder(genre_datas)

# 依次打开所有文件夹，读取所有小说并存入字典
# 建立种类名->对应小说list，小说名->对应内容list
fiction_datas = {}
name_datas = {}
for i in range(0, len(genre_datas)):
    name_datas[genre_datas[i]] = os.listdir('起点小说/'+genre_datas[i])
    name_datas[genre_datas[i]] = delete_hidden_floder(name_datas[genre_datas[i]])
    # 为方便分析，每个小说的 10 章list合并为 1 章
    for j in range(0, len(name_datas[genre_datas[i]])):
        fiction_datas[name_datas[genre_datas[i]][j]] = []
        dic_file = os.listdir('起点小说/'+genre_datas[i]+'/'+name_datas[genre_datas[i]][j])
        dic_file = delete_hidden_floder(dic_file)
        merged_file = []
        for k in range(0, len(dic_file)):
            merged_file += open('起点小说/'+genre_datas[i]+'/'+name_datas[genre_datas[i]][j]+'/'+dic_file[k], encoding='utf-8').readlines()
        fiction_datas[name_datas[genre_datas[i]][j]]= merged_file

```

检验一下我们的加载结果：

```python
genre_datas, name_datas[genre_datas[0]]
```

```text
(['悬疑',
  '轻小说',
  '都市',
  '历史',
  '仙侠',
  '玄幻',
  '科幻',
  '奇幻',
  '现实',
  '军事',
  '游戏',
  '武侠',
  '体育'],
 ['我有一座冒险屋',
  '魔临',
  '深夜书屋',
  '万界疯人院',
  '助灵为乐系统',
  '熟睡之后',
  '颤栗高空',
  '诡神冢',
  '老九门',
  '鬼吹灯II',
  '全球崩坏',
  '我在黄泉有座房',
  '捡了一片荒野',
  '我不是真的想惹事啊',
  '好想有个系统掩饰自己',
  '我能回档不死',
  '盗墓笔记',
  '鬼吹灯（盗墓者的经历）',
  '人间苦',
  '黎明医生'])
```

```python
fiction_datas['我有一座冒险屋'][0:5]
```

```text
['手机上这个以恐怖屋大门为图标的应用软件，很像是市面上流行的模拟经营类手游，只不过其经营的不是饭店、水族馆、宠物乐园，而是鬼屋。\n',
 '陈歌盯着屏幕，他怎么也想不通，为什么父母遗留下的手机里，会有这样一个奇怪的小游戏。\n',
 '他仔细翻看应用界面，里面所有信息都和他的鬼屋相吻合，包括每日游览人数和馆内设施场景，这游戏让陈歌产生了一种奇怪的感觉，好像游戏里需要经营的鬼屋，就是他现实中的鬼屋一样。\n',
 '同样糟糕的处境，同样是濒临倒闭，两者之间有太多的共同点。\n',
 '“难道这个游戏就是以我的鬼屋为原型制作的吗？那如果在游戏里改变了鬼屋，现实中是不是也能受益？”\n']
```

## 2. 数据预处理

### 2.1 生成标签

我们加载的数据无法直接被算法模型处理，所以我们需要对数据进行一系列预处理准备工作。

前面我们按照文件层级结构建立了多个字典，为方便数据处理，还需要给小说生成对应标签，并统一转为 list 类型（长度为 260 的小说 list）。

```python
import numpy as np

# 生成供训练使用的 list 类型数据和标签
labels = np.ones(len(fiction_datas)).tolist()
datas = np.ones(len(fiction_datas)).tolist()
genre_id = 0
name_id = 0
for genre in genre_datas:
    for name in name_datas[genre]:
        datas[name_id] = fiction_datas[name]
        labels[name_id] = genre_id
        name_id += 1
    genre_id += + 1
```

### 2.2 划分训练集和测试集

使用 scikit-learn 工具里面的 train_test_split 类在 10001 个样本当中，随机划出 25% 个样本和标签来作为测试集，剩下的 75% 作为训练集来进行训练我们的分类器。

```python
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(datas, labels, test_size=0.25, random_state=5)
```

简单查看一下样本和标签：

```python
train_x[5][0:5]
```

```text
['第十章苍茫大地\n',
 '漆黑的青铜棺椁内部渐渐安静下来，没有人再说话，所有人皆充满惧意，望着前方装殓尸体的青铜棺，众人发出的粗重的呼声，每一个人内心都很紧张。\n',
 '青铜棺绿锈斑驳，内部到底装殓了怎样的人物？\n',
 '“这一切都应与泰山上的五色祭坛有关。”\n',
 '过了很久，众人才低声议论起来，他们想知道这一切为何会发生。\n']
```

```python
test_y[0:10]
```

```text
[5, 11, 12, 4, 2, 4, 6, 9, 12, 6]
```

### 2.3 分词

nlp 工作往往是以词语作为基本特征进行分析，所以需要对文本进行分词。这里为了代码简洁方便理解，将分词设计成 tokenize_words 函数，供后续直接调用。我们使用 jieba 分词库。

```python
import jieba

def tokenize_words(corpus):
    tokenized_words = jieba.cut(corpus) # 调用 jieba 分词
    tokenized_words = [token.strip() for token in tokenized_words] # 去掉回车符，转为list类型
    return tokenized_words
```

```python
# 随便输入一句话调用函数验证一下
a = '青铜巨棺古朴无华，上面有一些模糊的古老图案\n'
b = tokenize_words(a)
b
```

```text
['青铜', '巨棺', '古朴', '无华', '，', '上面', '有', '一些', '模糊', '的', '古老', '图案', '']
```

### 2.4 去除停用词

在自然语言中，很多字词是没有实际意义的，比如：【的】【了】【得】等，因此要将其剔除。首先加载我们刚刚下载好的停用词表。这里也可以自行在网上下载，编码格式为 utf-8，每行一个停用词。为了方便调用，我们将去除停用词的操作放到 remove_stopwords 函数当中。

```python
!wget - nc "http://labfile.oss.aliyuncs.com/courses/1208/stop_word.txt"
```

```text
--2021-09-30 16:57:12--  http://-/
正在解析主机 - (-)... 失败：nodename nor servname provided, or not known。
wget: 无法解析主机地址 “-”
--2021-09-30 16:57:12--  http://nc/
正在解析主机 nc (nc)... 失败：nodename nor servname provided, or not known。
wget: 无法解析主机地址 “nc”
--2021-09-30 16:57:14--  http://labfile.oss.aliyuncs.com/courses/1208/stop_word.txt
正在解析主机 labfile.oss.aliyuncs.com (labfile.oss.aliyuncs.com)... 47.110.177.159
正在连接 labfile.oss.aliyuncs.com (labfile.oss.aliyuncs.com)|47.110.177.159|:80... 已连接。
已发出 HTTP 请求，正在等待回应... 200 OK
长度：15185 (15K) [text/plain]
正在保存至: “stop_word.txt”

stop_word.txt       100%[===================>]  14.83K  --.-KB/s  用时 0.04s   

2021-09-30 16:57:15 (367 KB/s) - 已保存 “stop_word.txt” [15185/15185])

下载完毕 --2021-09-30 16:57:15--
总用时：2.6s
下载了：1 个文件，0.04s (367 KB/s) 中的 15K
```

```python
def remove_stopwords(corpus): # 函数输入为全部样本集（包括训练和测试）
    sw = open('stop_word.txt', encoding='utf-8') # 加载停用词表
    sw_list = [l.strip() for l in sw] # 去掉回车符存放至list中
    # 调用分词函数
    tokenized_data = tokenize_words(corpus)
    # 使用list生成式对停用词进行过滤
    filtered_data = [data for data in tokenized_data if data not in sw_list]
    # 用' '将 filtered_data 串起来赋值给 filtered_datas（不太好介绍，可以看一下下面处理前后的截图对比）
    filtered_datas = ' '.join(filtered_data)
    # 返回是去除停用词后的字符串
    return filtered_datas
```

不妨再用一句话来检验一下去除停用词和分词的结果：

```python
a = '李凡从衣服内兜里掏出了一个很小的小本放在了桌子上。\n'
b = remove_stopwords(a)
b
```

```text
'李凡 衣服 兜里 掏出 很小 小本 放在 桌子'
```

接下来，构建一个函数整合分词和剔除停用词的预处理工作，调用 tqdm 模块显示进度。

```python
from tqdm.notebook import tqdm

def preprocessing_datas(datas):
    preprocessing_datas = []
    # 对 datas 当中的每一个 data 进行去停用词操作
    # 并添加到上面刚刚建立的 preprocessed_datas 当中
    for data in tqdm(datas):
        preprocessing_data = ''
        for sentence in data:
            sentence = remove_stopwords(sentence)
            preprocessing_data += sentence
        preprocessing_datas.append(preprocessing_data)
    # 返回预处理后的样本集
    return preprocessing_datas
```

最后直接调用上面的与处理函数对训练集和测试集进行预处理，可能会稍微有些慢：

```python
pred_train_x = preprocessing_datas(train_x)
pred_test_x = preprocessing_datas(test_x)
```

```text
  0%|          | 0/195 [00:00<?, ?it/s]



  0%|          | 0/65 [00:00<?, ?it/s]
```

## 3. 特征提取

在进行分词及去停用词处理过后，得到的是一个分词后的文本。现在我们的分类器是 SVM，而 SVM 的输入要求是数值型的特征。这意味着我们要将前面所进行预处理的文本数据进一步处理，将其转换为数值型数据。

转换的方法有很多种，这里使用最经典的 TF-IDF 方法。

在 Python 当中，我们可以通过 scikit-learn 来实现 TF-IDF 模型。并且，使用 scikit-learn 库将会非常简单。这里主要用到了 `TfidfVectorizer()` 类。

接下来我们开始使用这个类将文本特征转换为相应的 TF-IDF 值。

首先加载 `TfidfVectorizer` 类，并定义 TF-IDF 模型训练器 `vectorizer` 。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1))
```

对预处理过后的 `pred_train_d` 进行特征提取：

```python
tfidf_train_features = vectorizer.fit_transform(pred_train_x)
tfidf_train_features
```

```text
<195x211797 sparse matrix of type '<class 'numpy.float64'>'
  with 592801 stored elements in Compressed Sparse Row format>
```

通过这一步，我们得到了 195 个 211797 维数的向量作为我们的训练特征集。我们可以查看转换结果：

```python
np.array(tfidf_train_features.toarray()).shape
```

```text
(195, 211797)
```

用训练集训练好特征后的 `vectorizer` 来提取测试集的特征：

⚠️注意这里不能用 `vectorizer.fit_transform()` 要用 `vectorizer.transform()`，否则，将会对测试集单独训练 TF-IDF 模型，而不是在训练集的词数量基础上做训练。这样词总量跟训练集不一样多，排序也不一样，将会导致维数不同，最终无法完成测试。

```python
tfidf_test_features = vectorizer.transform(pred_test_x)
tfidf_test_features
```

```text
<65x211797 sparse matrix of type '<class 'numpy.float64'>'
  with 143305 stored elements in Compressed Sparse Row format>
```

完成之后，我们得到 65 个 28335 维数的向量作为我们的测试特征集。

## 3. 构建并训练分类器

在获得 TF-IDF 特征之后，我们才能真正执行分类任务。我们不需要自己手动构建算法模型，可以调用 `sklearn` 中的 `svm.SVC` 类来训练一对多 SVM 分类器。

```python
from sklearn import svm

clf = svm.SVC(kernel='linear', C=10, gamma='scale', decision_function_shape='ovo')
clf.fit(tfidf_train_features, train_y)
```

```text
SVC(C=10, decision_function_shape='ovo', kernel='linear')
```

## 4. 查看预测结果

为了直观显示分类的结果，我们用 scikit-learn 库中的 accuracy_score 函数来计算一下分类器的准确率（准确率即 test_l 中与 prediction 相同的比例）。

```python
predictions = clf.predict(tfidf_test_features)
predictions
```

```text
array([ 7, 10, 12,  5, 10, 11,  0,  7, 12,  7,  0,  5, 11, 12,  3,  0, 11,
        1,  9, 10, 12,  7,  0,  9,  0, 12,  3,  3,  0,  0, 11, 11, 11,  3,
        0,  0,  0,  7,  0,  0,  6,  0, 11,  0,  0,  7,  5,  5,  3,  0,  5,
        0, 10,  7,  5,  1,  0,  5,  3,  0,  5,  0, 10, 10,  8])
```

可以看到我们的准确率达到了34%，虽然不是很高，但是考虑到有13个分类之多，而且仅仅只有200个训练集，这个数据还算比较乐观了，如果想要更多的数据集，考虑获取更多的数据，或者使用更高级的分类算法。

```python
from sklearn import metrics

accuracy_score = np.round(metrics.accuracy_score(test_y, predictions), 2)
print('准确率为'+str(accuracy_score*100)+'%')
```

```text
准确率为34.0%
```

# 参考文章









[文本分类概述](https://www.wolai.com/eFQQux6j6v9TmYAh8A6Syg)
