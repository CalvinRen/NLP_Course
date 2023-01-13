import os.path
from preprocess import Preprocess
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Conv1D, Flatten, GlobalMaxPool1D, Concatenate, InputLayer
from keras.layers import concatenate, Activation, Dense, Input, Add
from keras.models import load_model
from keras.initializers import Constant
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


class cnn:
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
        input_layer = Input(shape=(self.max_length,))
        embedding_layer = Embedding(self.num_words, 300, embeddings_initializer=Constant(self.embedding_matrix), trainable=False)(input_layer)
        model_a = Conv1D(filters=32, kernel_size=2)(embedding_layer)
        model_a = Activation('relu')(model_a)
        model_a = GlobalMaxPool1D()(model_a)

        model_b = Conv1D(filters=32, kernel_size=3)(embedding_layer)
        model_b = Activation('relu')(model_b)
        model_b = GlobalMaxPool1D()(model_b)
        added = Concatenate()([model_a, model_b])
        out = Dense(20, activation='softmax')(added)

        model_combined = Model(inputs=input_layer, outputs=out)
        model_combined.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.utils.precision_m, self.utils.recall_m, self.utils.f1_m])
        model_combined.summary()

        return model_combined

    def train(self, epochs):
        train_x = self.padded_doc[:11314]
        test_x = self.padded_doc[11315:]
        train_y = np.asarray(self.Y[:11314])
        encoded_train_y = np_utils.to_categorical(train_y)
        test_y = np.asarray(self.Y[11315:])
        encoded_test_y = np_utils.to_categorical(test_y)

        model = self.model()
        history = model.fit(train_x, encoded_train_y, validation_data=(test_x, encoded_test_y), epochs=epochs, batch_size=32)
        model.save('model_cnn.h5')
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
        if os.path.exists('model_cnn.h5'):
            model = load_model('model_cnn.h5', custom_objects={'precision_m': self.utils.precision_m, 'recall_m': self.utils.recall_m, 'f1_m': self.utils.f1_m})
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
    model = cnn()
    model.data_load()
    model.train(epochs=15)
    model.test()
