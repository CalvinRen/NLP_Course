import os.path
from preprocess import Preprocess
from keras.models import load_model
from keras.utils import np_utils
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


class lstm:
    def __init__(self):
        self.utils = Preprocess()
        self.padded_doc = None
        self.Y = None
        self.num_words = None
        self.embedding_matrix = None
        self.max_length = None

    def data_load(self):
        X, Y = self.utils.get_data()
        docs = self.utils.preprocessing_dataset(X)
        del X
        padded_doc, Y, word_index, max_length = self.utils.pad_input(docs, Y)
        file_name = 'glove.6B.300d.txt'
        embedding_matrix = self.utils.create_embedding_matrix(file_name, word_index, 300)
        num_words = len(word_index) + 1

        self.padded_doc = padded_doc
        self.Y = Y
        self.num_words = num_words
        self.embedding_matrix = embedding_matrix
        self.max_length = max_length

    def model(self):
        x = Sequential()
        x.add(Dense(128, activation='relu', input_shape=(self.max_length,)))
        x.add(Dense(10, activation='softmax'))
        return x

    def train(self):
        x_train, y_train = None
        lstm = self.model()
        his = lstm.fit(x_train, y_train, epochs=16, batch_size=128)
        lstm.save('model_lstm.h5')


    def test(self):
        if os.path.exists('model_word_embedding_lstm.h5'):
            lstm = load_model('model_word_embedding_lstm.h5', custom_objects={'precision_m': self.utils.precision_m, 'recall_m': self.utils.recall_m, 'f1_m': self.utils.f1_m})
            print('load model')
            lstm.summary()
        else:
            raise Exception('model not found, please train first')

        y_ = model.predict(x_test)
        print(model.metrics_names)
        print(res)
        print(classification_report(y_test, np.argmax(model.predict(x_test), axis=1)))



        embedding_layer = Embedding(self.num_words, 300, embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_length, trainable=False)
        model.add(embedding_layer)
        model.add(CuDNNLSTM(256))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy', self.utils.precision_m, self.utils.recall_m, self.utils.f1_m])
        model.summary()
        return model

    def train(self, epochs):
        train_x = self.padded_doc[:11314]
        test_x = self.padded_doc[11315:]
        train_y = np.asarray(self.Y[:11314])
        encoded_train_y = np_utils.to_categorical(train_y)
        test_y = np.asarray(self.Y[11315:])
        encoded_test_y = np_utils.to_categorical(test_y)

        model = self.model()
        history = model.fit(train_x, encoded_train_y, validation_data=(test_x, encoded_test_y), epochs=epochs, batch_size=128)
        model.save('model_word_embedding_lstm.h5')
        print('save model')

        # 绘制训练 & 验证的准确率值
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def test(self):
        if os.path.exists('model_lstm.h5'):
            model = load_model('model_lstm.h5', custom_objects={'precision_m': self.utils.precision_m, 'recall_m': self.utils.recall_m, 'f1_m': self.utils.f1_m})
            print('load model')
            model.summary()
        else:
            raise Exception('model not found, please train first')

        # load test dataset
        test_x = self.padded_doc[11315:]
        test_y = np.asarray(self.Y[11315:])
        encoded_test_y = np_utils.to_categorical(test_y)

        res = model.evaluate(test_x, encoded_test_y)
        for (name, value) in zip(model.metrics_names, res):
            print(name, value)
        print(classification_report(test_y, np.argmax(model.predict(test_x), axis=1)))


if __name__ == '__main__':
    model = lstm()
    model.data_load()
    model.train(epochs=15)
    model.test()

