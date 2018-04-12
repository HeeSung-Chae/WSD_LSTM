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

wsd_word_eng = 'mok'
wsd_word_kor = '목'
# for posDir, vecDir in zip(posSentGlob, vecPosSentGlob):
for posDir in posSentGlob:
    # 목 파일만 테스트, 전체 돌릴때는 if삭제
    if posDir.count(wsd_word_eng) > 0:
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
                    vec.append(float(b.replace("'", "")))
                # print(vecEojul, vec)
                vecDic[vecEojul] = vec

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


        # 전체 문장(중복단어제거) 리스트, 전체 문장 리스트 Index ( 중의성 단어까지의 어절들)
        # 입력 문장 리스트, 입력 문장 임베딩 벡터
        # 출력 문장 리스트, 출력 문장 리스트 Index
        total_sent = []
        total_sent_index = []
        input_sent = []
        input_sent_vec = []
        output_sent = []
        output_sent_data = []
        # sentChar_len = sentence_words.__len__()

        start_time = time.time()
        # 문장들 필요한 정보들 저장
        for sentence_content in posList:
            # 중의성 단어까지의 어절 저장
            sentence_words = []
            for a in sentence_content.split(' ')[0:-1]:
                flag = True
                for b in sentence_words:
                    if b == a:
                        flag = False
                    else:
                        flag = True
                if flag == True:
                    sentence_words.append(a)
                if a.split('__')[0] == wsd_word_kor:
                    break


            temp_total_sent = []
            temp_total_sent_index = {}
            temp_input_sent = []
            temp_input_sent_vec = []
            temp_output_sent = []
            temp_output_sent_data = [[]]

            wrongSent = False
            for num, a in enumerate(sentence_content.split(' ')[0:-1]):
                # 중의성 단어까지를 문장 길이로
                if a.split('__')[0] == wsd_word_kor:
                    # 첫 단어가 목인 경우, 순방향이기 때문에 패스
                    if num == 0:
                        wrongSent = True
                        break
                    else:
                        temp_output_sent.append(a)
                        temp_total_sent.append(a)
                        break

                if num == 0:
                    temp_input_sent.append(a)
                    temp_total_sent.append(a)
                elif num > 0:
                    temp_input_sent.append(a)
                    temp_output_sent.append(a)

                    flag_total_sent = True
                    for b in temp_total_sent:
                        if b == a:
                            flag_total_sent = False
                            break
                    if flag_total_sent.__eq__(True):
                        temp_total_sent.append(a)


            if wrongSent.__eq__(True):
                continue
            else:
                # temp_total_sent 안해도되나?
                # temp_total_sent.reverse()
                temp_input_sent.reverse()
                temp_output_sent.reverse()
                while True:
                    if temp_input_sent.__len__() > 18:
                        # temp_total_sent.pop()
                        temp_input_sent.pop()
                        temp_output_sent.pop()
                    elif temp_input_sent.__len__() <= 18:
                        break
                # temp_total_sent.reverse()
                temp_input_sent.reverse()
                temp_output_sent.reverse()

                for num, a in enumerate(temp_total_sent):
                    temp_total_sent_index[a] = num + 1

                for a in temp_output_sent_data:
                    for b in temp_output_sent:
                        a.append(temp_total_sent_index[b])

                # for a in temp_input_sent_vec:
                for words in temp_input_sent:
                    for b in vecDic:
                        # if b == "0":
                        #     temp_input_sent_vec.append(0)
                        if b == words:
                            temp_input_sent_vec.append(vecDic[b])
                            break

                total_sent.append(temp_total_sent)
                total_sent_index.append(temp_total_sent_index)
                input_sent.append(temp_input_sent)
                input_sent_vec.append(temp_input_sent_vec)
                output_sent.append(temp_output_sent)
                output_sent_data.append(temp_output_sent_data)

            #     print("total_sent : ", temp_total_sent)
            #     print("total_sent_index : ", temp_total_sent_index)
            #     print("input_sent : ", temp_input_sent)
            #     print("input_sent_vec : ", temp_input_sent_vec.__len__())
            #     print("output_sent : ", temp_output_sent)
            #     print("output_sent_data : ", temp_output_sent_data)
            #     print()
        # break


        # 문자열이라 "0" 값을 수동으로 append
        # 근데 왜 만들었지?...
        for a in input_sent:
            a.reverse()
            while True:
                if a.__len__() < 18:
                    a.append("0")
                else:
                    break
            a.reverse()
        #     print(a.__len__(), a)
        # break



        left = 0
        right = 0
        up = 0
        down = 0

        # inputData = []
        inputVecData = []
        outputData = []
        # 입력 embedding vector 18로 padding
        for a in input_sent_vec:
            if a.__len__() < 18:
                up = 18 - a.__len__()
            elif a.__len__() == 18:
                up = 0
            inputVec_constant = tf.constant(a)
            paddings = tf.constant([[up, down], [left, right]])

            inputVec_result = tf.pad(inputVec_constant, paddings, "CONSTANT")
            inputVecData.append(inputVec_result)

        # 출력 index 길이 18로 조정
        for a in output_sent_data:
            for b in a:
                b.reverse()
                while True:
                    if b.__len__() == 18:
                        break
                    elif b.__len__() < 18:
                        b.append(0)
                    elif b.__len__() > 18:
                        b.remove()
                b.reverse()
                # outputData.append(tf.constant(b))
                outputData.append(b)

        sentence_time = time.time()
        # 문장 정보 저장 종료


        # for a, b in zip(inputVecData, outputData):
        #     print(a)
        # print(inputVecData)
        # asdf = tf.expand_dims(inputVecData, 0)
        # print(inputVecData)
        # break




        # shape 수정한게 적용이 안됨
        # 일단 임시로 sess에서 수정해서..
        # shape=(18, 300) --> shape=(1, 18, 300)
        # for a in inputVecData:
        #     a = tf.expand_dims(a, 0)



        hidden_size = 300
        input_dim = 300
        batch_size = 1

        sequence_length = 18

        X = tf.placeholder(tf.float32, [batch_size, sequence_length, input_dim])
        Y = tf.placeholder(tf.int32, [batch_size, sequence_length])
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, X,
                                             initial_state=initial_state, dtype=tf.float32)

        weights = tf.ones([batch_size, sequence_length])

        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs,
                                                         targets=Y, weights=weights)
        loss = tf.reduce_mean(sequence_loss)

        train = tf.train.FtrlOptimizer(learning_rate=0.002).minimize(loss)

        prediction = tf.argmax(outputs, axis=2)

        correctCount = 0
        errorCount = 0
        allCount = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for inputVec, outData in zip(inputVecData, outputData):
                dim = tf.expand_dims(inputVec, 0)
                print(inputVec)
                print(dim)
                inputRun = sess.run(dim)
                for i in range(1200):
                    try:
                        l, _ = sess.run([loss, train], feed_dict={X: inputRun, Y: outData})
                        result = sess.run(prediction, feed_dict={X: inputRun})

                        if(i == 1199):
                            print(i, "loss: ", l, "prediction: ", result, "true Y: ", outData)

                            # print char using dic
                            result_str = [total_sent[c] for c in np.squeeze(result)]
                            print("\tPrediction str: ", ''.join(result_str))
                            print()

                            temp = True
                            for aa, bb in zip(result, outData):
                                for cc, dd in zip(aa, bb):
                                    if(cc == dd):
                                        temp = True
                                    else:
                                        temp = False
                                        break

                            if(temp == True):
                                correctCount += 1
                                allCount += 1
                            else:
                                allCount += 1
                    except ValueError as errorMessage:
                        print(i, "ValueError ", "ErrorMessage: ", errorMessage, "true Y: ", outData)
                        errorCount += 1
                        allCount += 1
                        # 아래 break 2개 제거
                        break
                break

        print()
        print("sentecne Time : ", sentence_time - start_time, "seconds")
        print("training Time : ", time.time() - sentence_time, "seconds")
        print("Time : ", time.time() - start_time, "seconds")
        print("correct : ", correctCount)
        print("  error  : ", errorCount)
        print("    all    : ", allCount)

