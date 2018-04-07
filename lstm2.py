# 목 1551 문장
import glob
import os
import tensorflow as tf
import numpy as np
import time


posSentenceDir = '../SkipGram/Content/posSentence/'
posSentGlob = glob.glob(os.path.join(posSentenceDir, '*.txt'))
# vecPosSentDir = '../SkipGram/Content/vec_posSent/'
# vecPosSentGlob = glob.glob(os.path.join(vecPosSentDir, '*.vec'))
vecDir = '../SkipGram/Content/vec_posSent/all_word.vec'


# for posDir, vecDir in zip(posSentGlob, vecPosSentGlob):
for posDir in posSentGlob:
    # 목 파일만 테스트, 전체 돌릴때는 if삭제
    if posDir.count('mok') > 0:
        posOpen = open(posDir, 'r', encoding='utf-8')
        posList = posOpen.readlines()
        vecOpen = open(vecDir, 'r', encoding='utf-8')
        vecList = vecOpen.readlines()

        vecDic = {}
        # a는 단어랑 그 단어의 벡터 값
        # vecEojul는 단어, vec는 단어의 벡터값
        for wordCount, a in enumerate(vecList):
            if wordCount != 0:
                vecEojul = a.split(' ')[0]
                vec = []
                for b in a.split(' ')[1:301]:
                    vec.append(b)
                # print(vecEojul, vec)
                vecDic[vecEojul] = vec

        # vecDic 출력 확인용
        # a는 어절, vecDic[a]는 a의 vec값
        # print(vecDic)
        # for a in vecDic:
        #     print(a, vecDic[a])
        #     if a.count('해부') > 0:


        # sentVecDIc는 문장의 단어 벡터 저장한 딕셔너리
        sentVecDic = {}

        # sentence는 내용어만 추출된 문장
        for sentence in posList:
            # word는 단어, vecDic[a]는 그 단어의 벡터 값
            # sentVec는 문장의 단어 벡터들의 리스트
            sentVec = []
            for word in sentence.split(' ')[0:-1]:
                for a in vecDic:
                    if a == word:
                        sentVec.append(vecDic[a])

            # 문장의 벡터 리스트를 딕셔너리에 저장
            sentVecDic[sentence] = sentVec



        #########################
        # CREATE CELL
        #########################
        hidden_size = 300
        input_dim = 300
        batch_size = 1
        sequence_temp = 18

        correctCount = 0
        allCount = 0
        errorCount = 0


        # hihello 예제로 구현한 lstm
        X = tf.placeholder(tf.float32,
                           [batch_size, sequence_temp, input_dim])
        Y = tf.placeholder(tf.int32,
                           [batch_size, sequence_temp])
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                            state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, X,
                                             initial_state=initial_state, dtype=tf.float32)

        # outputs이 얼마나 좋은가를 sequence_loss를 이용하여 계산해야됨
        weights = tf.ones([batch_size, sequence_temp])

        # outputs의 값을 바로 logits에 사용하는 것은 좋지 않음
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                         targets=Y, weights=weights)
        loss = tf.reduce_mean(sequence_loss)

        # ============= 학습, 테스트 다 짜고나서 다시 한번 Optimizer 테스트 필요
        # train = tf.train.AdagradOptimizer(learning_rate=0.002).minimize(loss)
        train = tf.train.FtrlOptimizer(learning_rate=0.002).minimize(loss)
        # train = tf.train.ProximalAdagradOptimizer(learning_rate=0.002).minimize(loss)

        prediction = tf.argmax(outputs, axis=2)



        # 그래프 리셋
        try:
            tf.reset_default_graph()
        except AssertionError:
            continue

        # with tf.Session() as sess:
        sess = tf.Session()
        start_time = time.time()
        # 문장별로 학습
        for sentence_content in posList:
            sentence_time = time.time()

            sentence_words = []
            for a in sentence_content.split(' ')[0:-1]:
                # print(sentChar)
                # sentChar.append(a)
                flag = True
                for b in sentence_words:
                    if b == a:
                        flag = False
                    else:
                        flag = True
                if flag == True:
                    sentence_words.append(a)

                # 한 문장에 중의성 단어가 2번 나오는 경우 처리해줘야 됨
                # 처음 시작할때 문장.count()해서 문장에 몇번있는지 체크해서?
                # 중의성 단어 변수로 바꿔야됨
                if a.split('__')[0] == '목':
                    break

            # 입력 단어 리스트, 출력 단어 리스트
            # 전체 단어 리스트, 전체 단어 리스트 Index
            # 출력 단어 리스트 Index
            input_words = []
            output_words = []
            total_words = []
            wordsDic = {}
            output_data = [[]]
            sentChar_len = sentence_words.__len__()
            for num, a in enumerate(sentence_content.split(' ')[0:-1]):
                if a.split('__')[0] == '목':
                    output_words.append(a)
                    total_words.append(a)
                    break
                if num == 0:
                    input_words.append(a)
                    total_words.append(a)
                elif num > 0:
                    input_words.append(a)
                    output_words.append(a)

                for b in total_words:
                    if b == a:
                        break
                    else:
                        total_words.append(a)
                        break
            # sequence_length = sentChar_len - 1
            sequence_length = output_words.__len__()

            for num, a in enumerate(total_words):
                wordsDic[a] = num

            for a in output_data:
                for b in output_words:
                    a.append(wordsDic[b])


            # 문장의 embedding vector
            x_vec = [[]]
            for a in x_vec:
                # for b in a:
                for words in input_words:
                    # 단어 벡터 딕셔너리에서 해당 단어를 찾아서 벡터값 append
                    # print("words : ", words)
                    for c in vecDic:
                        if c == words:
                            # print("append word : ", c)
                            a.append(vecDic[c])
                            break

            # with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1200):
                try:
                    l, _ = sess.run([loss, train], feed_dict={X : x_vec, Y : output_data})
                    result = sess.run(prediction, feed_dict={X : x_vec})
                    if(i==1199):
                        print(i, "loss: ", l, "prediction: ", result, "true Y: ", output_data)
                        temp = True
                        for a, b, in zip(result, output_data):
                            for c, d in zip(a, b):
                                if(c == d):
                                    temp = True
                                else:
                                    temp = False
                                    break

                        if(temp == True):
                            correctCount += 1
                            allCount += 1
                        else:
                            allCount += 1
                        # print text
                        # result_str = [total_words[c] for c in np.squeeze(result)]
                        # print("\tPrediction str: ", ''.join(result_str),)
                except ValueError:
                    print(i, "loss: ", "prediction: ", "true Y: ", output_data)
                    errorCount += 1
                    allCount += 1
                    break
            print("sentence time : ", time.time() - sentence_time, "seconds")
                # break
        print()
        print(time.time() - start_time, "seconds")
        print("correct : ", correctCount)
        print("error : ", errorCount)
        print("all : ", allCount)
