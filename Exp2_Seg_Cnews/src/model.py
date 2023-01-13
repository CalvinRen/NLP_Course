# -*- coding: utf-8 -*-
import tensorflow.keras as keras
from config import Config
from preprocess import preprocesser
import os
from sklearn import metrics
import numpy as np


class TextCNN(object):

    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()

    def model(self):
        num_classes = self.config.get("CNN_training_rule", "num_classes")
        vocab_size = self.config.get("CNN_training_rule", "vocab_size")
        seq_length = self.config.get("CNN_training_rule", "seq_length")

        conv1_num_filters = self.config.get("CNN_training_rule", "conv1_num_filters")
        conv1_kernel_size = self.config.get("CNN_training_rule", "conv1_kernel_size")

        conv2_num_filters = self.config.get("CNN_training_rule", "conv2_num_filters")
        conv2_kernel_size = self.config.get("CNN_training_rule", "conv2_kernel_size")

        hidden_dim = self.config.get("CNN_training_rule", "hidden_dim")
        dropout_keep_prob = self.config.get("CNN_training_rule", "dropout_keep_prob")

        model_input = keras.layers.Input((seq_length,), dtype='float64')
        embedding_layer = keras.layers.Embedding(vocab_size+1, 256, input_length=seq_length)
        embedded = embedding_layer(model_input)

        # conv1形状[batch_size, seq_length, conv1_num_filters]
        conv_1 = keras.layers.Conv1D(conv1_num_filters, conv1_kernel_size, padding="SAME")(embedded)
        conv_2 = keras.layers.Conv1D(conv2_num_filters, conv2_kernel_size, padding="SAME")(conv_1)
        max_poolinged = keras.layers.GlobalMaxPool1D()(conv_2)

        full_connect = keras.layers.Dense(hidden_dim)(max_poolinged)
        droped = keras.layers.Dropout(dropout_keep_prob)(full_connect)
        relued = keras.layers.ReLU()(droped)
        model_output = keras.layers.Dense(num_classes, activation="softmax")(relued)
        model = keras.models.Model(inputs=model_input, outputs=model_output)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        print(model.summary())
        return model

    def train(self, epochs):
        trainingSet_path = self.config.get("data_path", "trainingSet_path")
        valSet_path = self.config.get("data_path", "valSet_path")
        seq_length = self.config.get("CNN_training_rule", "seq_length")
        model_save_path = self.config.get("result", "CNN_model_path")
        batch_size = self.config.get("CNN_training_rule", "batch_size")

        x_train, y_train = self.pre.word2idx(trainingSet_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(valSet_path, max_length=seq_length)

        model = self.model()
        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=1,
                      validation_data=(x_val, y_val))
            model.save(model_save_path, overwrite=True)

    def test(self):
        model_save_path = self.config.get("result", "CNN_model_path")
        testingSet_path = self.config.get("data_path", "testingSet_path")
        seq_length = self.config.get("CNN_training_rule", "seq_length")

        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        x_test, y_test = self.pre.word2idx(testingSet_path, max_length=seq_length)
        pre_test = model.predict(x_test)

        print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))
    
    def predict(self):
        model_save_path = self.config.get("result", "CNN_model_path")

        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()
        x_test = self.pre.word2idx_for_sample("银华基金杨靖聊政策性主题投资机会实录新浪财经讯 银华和谐主题基金拟任基金经理助理杨靖于3月25日做客新浪财经会客厅，与投资者聊09年政策性主题投资机会，以下为当日文字实录。     主持人乔旎：亲爱的新浪网友们大家好，欢迎收看今天的新浪财经会客厅栏目，我是主持人乔旎。我们今天邀请到的嘉宾是银华和谐主题基金拟任基金经理助理杨靖先生，跟网友打声招呼。杨靖：大家好。主持人乔旎：我今天跟您说今年市场是一个有熊心的小牛，熊心在哪儿？小牛在哪儿？杨靖：说有熊心的小牛，是因为今年总体市场来讲还是一个振荡向上的走势，底部出现盘高振荡，总体来讲应该是一个牛市。为什么说是小牛呢？可能也不会像以前我们想的大牛市，这种机会可能性不大。总体来讲应该是一个振荡向上的小牛市。为什么是熊心呢？很多人可以理解。毕竟经历了08年一年的暴跌，去年一年全球排名第二的跌幅。所以，很多人都还是有一颗熊心，都是往坏的方向去想，所以这个小牛确实走一步看一步，所以今年可能是一个振荡市。总体来讲应该是这样的情况。主持人乔旎：整体来看是一个牛市，但还是有一定的熊心，因为不管是我们的投资者朋友也好，还是其他很多人也好，心里来说还是比较谨慎一些。杨靖：对，各方面都比较谨慎。确实有些经济数据也是短时间内有转好的迹象，但还是需要再观察。所以，有一点熊心某种意义上讲也是对的，需要谨慎一些。主持人乔旎：在大家都向好的环境下有一定的熊心可能更加能保持一个理性的头脑。杨靖：对。主持人乔旎：进行一个谨慎的分析。杨靖：也不能盲目乐观。主持人乔旎：您怎么看待今年宏观消息、宏观政策对市场的影响？因为我们也发现最近出台的一些宏观政策比较多。杨靖：这次无论是从国外看还是从中国看，各国政府在面对经济危机的时候都表现出来一种前所未有的决心，一定要捍卫自己的经济增长，可能美国他们是希望能够实现零增长就很满意了。但是我们国家因为确实处在一个罕见的高速成长期的刚起步，我们改革开放才30年，我们觉得后边中国经济高速增长，至少还有20年。我们政府要捍卫的可能就是8%左右的成长。主持人乔旎：要维持这样一个增长速度。杨靖：对，所以我们政府其实有很多政策还没有使出来，因为我们的经济基础还是比较好的。所以，我们对政府的这些政策还是比较有信心的。所以，相关的其实很多投资机会也是伴随着政策产生的。主持人乔旎：这其中有什么样的投资机会呢？杨靖：很多，比如说我们政府现在投资的一些大型项目，比如说铁路、公路，还有市政建设等等，那就会拉动相关的，比如说工程机械、水泥等等这些行业，都会从中受益，它可能会抵销掉一部分出口和内需下降造成的负面影响。所以，我们今年紧跟政策是一大投资主线。主持人乔旎：那就是说今年这一轮行情更多可能是由政府推动。杨靖：对，今年的主线其实就是政策驱动，既包括中国政府自己的政策，其实也包括美国政府的政策。比如前几天美国政府剥离不良资产去买有毒资产，另外发行3000亿美元买它的长期国债，向市场注入流动性，在资本市场反映的就是有色金属暴涨，黄金价格暴涨，因为美元在贬值。这也是投资机会。国内反应过来也是有色板块在上涨。主持人乔旎：这个机会蕴藏的不仅仅是国内的这些政策，其中包括美国出台的一系列国际上的政策，可能在这其中都会蕴藏着一些投资机会，就看你怎么抓。杨靖：对。主持人乔旎：在这样一系列的投资会，我们怎么才能抓住其中的投资机会？杨靖：那就是你预判的能力。在什么样的情况下你预判政府会采取什么样的措施，你要提前一步最好，如果不行的话，半步也可以，看到一点端倪以后，你就提前稍微布局一下，然后收获会不错。主持人乔旎：我们知道前一段时间出台的十大产业振兴规划，是不是这其中也是蕴含着一些投资机会？杨靖：对，十大产业振兴计划，每一次出政策的前后相关的一些行业和板块表现得都非常好，那就是大家对政府的政策决心还是很信任的，觉得政府能够通过振兴计划把十大产业确实真正振兴起来，不是说说而已，确实我们还是比较有信心的，可能真的通过这些政策，即使今年可能不一定能够有非常大的效果，但是看长一点，确实很多行业还是能够从中受益很多。主持人乔旎：但是很多人比较担心的是这些政策只是这样的一个短时间的效应，怕长期维持不住。您刚才也提到了，可能从长期来看有些行业是不错的，但是可能其中有一些行业，很多网友认为可能只是其中一个短线机会，就怕进去以后又被套住，从股市上讲。杨靖：你说的对，但是有一个问题，基本面和估值还是要比较的，刚才我们谈到政策驱动的都是企业或者是一个行业基本面的情况，但还要考虑股票估值的问题。如果估值已经达到比较合理的状况，这时再出政策也不能轻易地进行投资，要考虑这些问题。再好的股票有一个估值的问题。主持人乔旎：可是很多人也有担心，政策要等它真正效果反映出来还是需要一些时间的，可能我们这么频繁地出政策，是不是也是对我们可能很多朋友来说比较担心这一点，这么频繁出政策，会不会效果还没有完全体现，然后又加了一针强心剂上。杨靖：您是怕政策跟不上是吧。主持人乔旎：我觉得政策有点太过于频繁。杨靖：其实总理说的话是对的，出手要稳、准、快、时。因为确实2008年这次全球的危机，由金融危机最后演变成金融危机，确实出乎所有人的预料，很多经济学家搞不清楚，包括像巴菲特这样的高手也套在美国股市里。遇到的危机是前所未有的，如果找一个可以比拟就是1929年大危机。所以，这种情况下政府必须采取一些非常快速的应急的手段，才能够扭转经济形势。否则的话，如果出手要慢的话，政策如果不到位，可能真的就会出现经济衰退。主持人乔旎：你的意思就是说，我们是在这样一个经济刺激计划下，在不断地对它进行刺激，在它没有完全反应出来或者消化掉进行刺激，可以确保经济绝对不会出现衰退。杨靖：对。政府的措施我感觉还是比较快、比较到位的，在这种情况下，今年即使不能够实现8%的增长，但肯定应该增长率也是不错的，应该不会出现经济衰退、通货紧缩这种情况，是不太可能的。主持人乔旎：我们也知道最近市场上也是有一定的回暖，包括这两天的股市也是出现了几连阳，您怎么看待现在这样一个情况呢？这样的一个回暖的迹象是不是表明了在这其中有一定的投资机会？包括我们的政策也在起着一定的作用？视频：银华拟任基金经理助理杨靖聊政策性投资机会", 600)
        pre_res = model.predict(x_test)
        categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        res = np.argmax(pre_res)
        print('predict result: ', categories[res])


class LSTM(object):

    def __init__(self):
        self.config = Config()
        self.pre = preprocesser()

    def model(self):
        seq_length = self.config.get("LSTM", "seq_length")
        num_classes = self.config.get("LSTM", "num_classes")
        vocab_size = self.config.get("LSTM", "vocab_size")

        model_input = keras.layers.Input((seq_length))
        embedding = keras.layers.Embedding(vocab_size+1, 256, input_length=seq_length)(model_input)
        LSTM = keras.layers.LSTM(256)(embedding)
        FC1 = keras.layers.Dense(256, activation="relu")(LSTM)
        droped = keras.layers.Dropout(0.5)(FC1)
        FC2 = keras.layers.Dense(num_classes, activation="softmax")(droped)

        model = keras.models.Model(inputs=model_input, outputs=FC2)

        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.RMSprop(),
                      metrics=["accuracy"])
        model.summary()
        return model

    def train(self, epochs):
        trainingSet_path = self.config.get("data_path", "trainingSet_path")
        valSet_path = self.config.get("data_path", "valSet_path")
        seq_length = self.config.get("LSTM", "seq_length")
        model_save_path = self.config.get("result", "LSTM_model_path")
        batch_size = self.config.get("LSTM", "batch_size")

        model = self.model()

        x_train, y_train = self.pre.word2idx(trainingSet_path, max_length=seq_length)
        x_val, y_val = self.pre.word2idx(valSet_path, max_length=seq_length)

        for _ in range(epochs):
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      validation_data=(x_val, y_val),
                      epochs=1)
            model.save(model_save_path, overwrite=True)

    def test(self):
        model_save_path = self.config.get("result", "LSTM_model_path")
        testingSet_path = self.config.get("data_path", "testingSet_path")
        seq_length = self.config.get("LSTM", "seq_length")

        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()

        x_test, y_test = self.pre.word2idx(testingSet_path, max_length=seq_length)
        pre_test = model.predict(x_test)

        print(metrics.classification_report(np.argmax(pre_test, axis=1), np.argmax(y_test, axis=1)))

    def predict(self):
        model_save_path = self.config.get("result", "LSTM_model_path")

        if os.path.exists(model_save_path):
            model = keras.models.load_model(model_save_path)
            print("-----model loaded-----")
            model.summary()
        x_test = self.pre.word2idx_for_sample("银华基金杨靖聊政策性主题投资机会实录新浪财经讯 银华和谐主题基金拟任基金经理助理杨靖于3月25日做客新浪财经会客厅，与投资者聊09年政策性主题投资机会，以下为当日文字实录。     主持人乔旎：亲爱的新浪网友们大家好，欢迎收看今天的新浪财经会客厅栏目，我是主持人乔旎。我们今天邀请到的嘉宾是银华和谐主题基金拟任基金经理助理杨靖先生，跟网友打声招呼。杨靖：大家好。主持人乔旎：我今天跟您说今年市场是一个有熊心的小牛，熊心在哪儿？小牛在哪儿？杨靖：说有熊心的小牛，是因为今年总体市场来讲还是一个振荡向上的走势，底部出现盘高振荡，总体来讲应该是一个牛市。为什么说是小牛呢？可能也不会像以前我们想的大牛市，这种机会可能性不大。总体来讲应该是一个振荡向上的小牛市。为什么是熊心呢？很多人可以理解。毕竟经历了08年一年的暴跌，去年一年全球排名第二的跌幅。所以，很多人都还是有一颗熊心，都是往坏的方向去想，所以这个小牛确实走一步看一步，所以今年可能是一个振荡市。总体来讲应该是这样的情况。主持人乔旎：整体来看是一个牛市，但还是有一定的熊心，因为不管是我们的投资者朋友也好，还是其他很多人也好，心里来说还是比较谨慎一些。杨靖：对，各方面都比较谨慎。确实有些经济数据也是短时间内有转好的迹象，但还是需要再观察。所以，有一点熊心某种意义上讲也是对的，需要谨慎一些。主持人乔旎：在大家都向好的环境下有一定的熊心可能更加能保持一个理性的头脑。杨靖：对。主持人乔旎：进行一个谨慎的分析。杨靖：也不能盲目乐观。主持人乔旎：您怎么看待今年宏观消息、宏观政策对市场的影响？因为我们也发现最近出台的一些宏观政策比较多。杨靖：这次无论是从国外看还是从中国看，各国政府在面对经济危机的时候都表现出来一种前所未有的决心，一定要捍卫自己的经济增长，可能美国他们是希望能够实现零增长就很满意了。但是我们国家因为确实处在一个罕见的高速成长期的刚起步，我们改革开放才30年，我们觉得后边中国经济高速增长，至少还有20年。我们政府要捍卫的可能就是8%左右的成长。主持人乔旎：要维持这样一个增长速度。杨靖：对，所以我们政府其实有很多政策还没有使出来，因为我们的经济基础还是比较好的。所以，我们对政府的这些政策还是比较有信心的。所以，相关的其实很多投资机会也是伴随着政策产生的。主持人乔旎：这其中有什么样的投资机会呢？杨靖：很多，比如说我们政府现在投资的一些大型项目，比如说铁路、公路，还有市政建设等等，那就会拉动相关的，比如说工程机械、水泥等等这些行业，都会从中受益，它可能会抵销掉一部分出口和内需下降造成的负面影响。所以，我们今年紧跟政策是一大投资主线。主持人乔旎：那就是说今年这一轮行情更多可能是由政府推动。杨靖：对，今年的主线其实就是政策驱动，既包括中国政府自己的政策，其实也包括美国政府的政策。比如前几天美国政府剥离不良资产去买有毒资产，另外发行3000亿美元买它的长期国债，向市场注入流动性，在资本市场反映的就是有色金属暴涨，黄金价格暴涨，因为美元在贬值。这也是投资机会。国内反应过来也是有色板块在上涨。主持人乔旎：这个机会蕴藏的不仅仅是国内的这些政策，其中包括美国出台的一系列国际上的政策，可能在这其中都会蕴藏着一些投资机会，就看你怎么抓。杨靖：对。主持人乔旎：在这样一系列的投资会，我们怎么才能抓住其中的投资机会？杨靖：那就是你预判的能力。在什么样的情况下你预判政府会采取什么样的措施，你要提前一步最好，如果不行的话，半步也可以，看到一点端倪以后，你就提前稍微布局一下，然后收获会不错。主持人乔旎：我们知道前一段时间出台的十大产业振兴规划，是不是这其中也是蕴含着一些投资机会？杨靖：对，十大产业振兴计划，每一次出政策的前后相关的一些行业和板块表现得都非常好，那就是大家对政府的政策决心还是很信任的，觉得政府能够通过振兴计划把十大产业确实真正振兴起来，不是说说而已，确实我们还是比较有信心的，可能真的通过这些政策，即使今年可能不一定能够有非常大的效果，但是看长一点，确实很多行业还是能够从中受益很多。主持人乔旎：但是很多人比较担心的是这些政策只是这样的一个短时间的效应，怕长期维持不住。您刚才也提到了，可能从长期来看有些行业是不错的，但是可能其中有一些行业，很多网友认为可能只是其中一个短线机会，就怕进去以后又被套住，从股市上讲。杨靖：你说的对，但是有一个问题，基本面和估值还是要比较的，刚才我们谈到政策驱动的都是企业或者是一个行业基本面的情况，但还要考虑股票估值的问题。如果估值已经达到比较合理的状况，这时再出政策也不能轻易地进行投资，要考虑这些问题。再好的股票有一个估值的问题。主持人乔旎：可是很多人也有担心，政策要等它真正效果反映出来还是需要一些时间的，可能我们这么频繁地出政策，是不是也是对我们可能很多朋友来说比较担心这一点，这么频繁出政策，会不会效果还没有完全体现，然后又加了一针强心剂上。杨靖：您是怕政策跟不上是吧。主持人乔旎：我觉得政策有点太过于频繁。杨靖：其实总理说的话是对的，出手要稳、准、快、时。因为确实2008年这次全球的危机，由金融危机最后演变成金融危机，确实出乎所有人的预料，很多经济学家搞不清楚，包括像巴菲特这样的高手也套在美国股市里。遇到的危机是前所未有的，如果找一个可以比拟就是1929年大危机。所以，这种情况下政府必须采取一些非常快速的应急的手段，才能够扭转经济形势。否则的话，如果出手要慢的话，政策如果不到位，可能真的就会出现经济衰退。主持人乔旎：你的意思就是说，我们是在这样一个经济刺激计划下，在不断地对它进行刺激，在它没有完全反应出来或者消化掉进行刺激，可以确保经济绝对不会出现衰退。杨靖：对。政府的措施我感觉还是比较快、比较到位的，在这种情况下，今年即使不能够实现8%的增长，但肯定应该增长率也是不错的，应该不会出现经济衰退、通货紧缩这种情况，是不太可能的。主持人乔旎：我们也知道最近市场上也是有一定的回暖，包括这两天的股市也是出现了几连阳，您怎么看待现在这样一个情况呢？这样的一个回暖的迹象是不是表明了在这其中有一定的投资机会？包括我们的政策也在起着一定的作用？视频：银华拟任基金经理助理杨靖聊政策性投资机会", 600)
        pre_res = model.predict(x_test)
        categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        res = np.argmax(pre_res)
        print('predict result: ', categories[res])



if __name__ == '__main__':
    test = TextCNN()
    # test.train(3)
    test.predict()

