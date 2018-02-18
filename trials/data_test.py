import pandas as pd
import collections
import json
import ast
import csv
import os
from tabulate import tabulate
import numpy as np
from lib.text_processor import TextProcessor
from lib.tfidf_calc import TfidfCalculator
from lib.file_ops import *
from lib.feedforward_network import FeedforwardNetwork

DIAGNOSE_COLUMN = 'Код_диагноза'
COMPLAIN_COLUMN = 'Жалобы'
PATIENT_COLUMN = 'Id_Пациента'
DOCTOR_COLUMN = 'Услуга'
AGE_COLUMN = 'Возраст'
GENDER_COLUMN = 'Пол'
CLINIC_COLUMN = 'Клиника'
RECORD_ID_COLUMN = 'Id_Записи'

TEST_DATA = './data/test_data.csv'
TRAIN_DATA = './data/train_data.csv'
TRAIN_COMPLAINS = './data/train_cmpl.json'
DOCTORS_FILENAME = './data/doctors.txt'
WORDS_FILENAME = './data/words.txt'
OUTPUT_FILENAME = './data/submission.csv'
MODELS_DIR = './data/models/'
MOST_FREQUENT_DIAG_FILENAME = './data/mf_diag.txt'
TEMP_OUTPUT_FILENAME = './data/temp_output.txt'
HANDLED_DOCTORS_FILENAME = './data/handled_doctors.txt'

DIAGNOSE_FREQ_THRESHOLD = 50

typos = load_dict_from_file('./data/typos.txt')


def translit(string):
    lower_case_letters = {u'а': u'a',
                          u'б': u'b',
                          u'в': u'v',
                          u'г': u'g',
                          u'д': u'd',
                          u'е': u'e',
                          u'ё': u'e',
                          u'ж': u'zh',
                          u'з': u'z',
                          u'и': u'i',
                          u'й': u'y',
                          u'к': u'k',
                          u'л': u'l',
                          u'м': u'm',
                          u'н': u'n',
                          u'о': u'o',
                          u'п': u'p',
                          u'р': u'r',
                          u'с': u's',
                          u'т': u't',
                          u'у': u'u',
                          u'ф': u'f',
                          u'х': u'h',
                          u'ц': u'ts',
                          u'ч': u'ch',
                          u'ш': u'sh',
                          u'щ': u'sch',
                          u'ъ': u'',
                          u'ы': u'y',
                          u'ь': u'',
                          u'э': u'e',
                          u'ю': u'yu',
                          u'я': u'ya', }

    translit_string = ""

    for index, char in enumerate(string.lower()):
        if char in lower_case_letters:
            char = lower_case_letters[char]
        translit_string += char

    return translit_string


def collect_text_corpus(dataframe, diagnose):
    corpus = []
    for index, row in dataframe.iterrows():
        if row[DIAGNOSE_COLUMN] == diagnose:
            corpus.append(row[COMPLAIN_COLUMN])
    return corpus


def take_top(dataframe, column, top=0):
    column_data = dataframe[column].tolist()

    counter = collections.Counter(column_data)
    counter = list(counter.items())
    counter.sort(key=lambda tup: tup[1], reverse=True)
    if top:
        return counter[0:top]
    else:
        return counter


def show_column_stats(dataframe, df_name, column):
    column_data = dataframe[column].tolist()

    counter = collections.Counter(column_data)
    counter = list(counter.items())
    counter.sort(key=lambda tup: tup[1], reverse=True)

    print('{} {} column stats:'.format(df_name, column))
    print('count distinct:', len(counter))

    sum_first_50 = sum(tup[1] for tup in counter[0:50])
    print('sum_first_50', sum_first_50)

    tbl = tabulate(counter[0:5])
    print(tbl)


def store_text_column(dataframe, idx_column, text_column, filename):
    dict = {}
    for index, row in dataframe.iterrows():
        idx = row[idx_column]
        if not idx in dict:
            dict[idx] = []
        dict[idx].append(row[text_column])

    with open(filename, mode='wt', encoding='utf-8') as output_file:
        print(json.dumps(dict), file=output_file)


def load_complains(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
         return dict(ast.literal_eval(file.read()))


def tfidf_complains(dataframe, text_processor, complains_dict, trash_words):
    tfidf = TfidfCalculator(text_processor=text_processor, trash_words=trash_words)

    diagnoses = take_top(dataframe, DIAGNOSE_COLUMN)

    complain_corpus = []
    for (diagnose, _) in diagnoses:
        complain_corpus.extend(complains_dict[diagnose])
    tfidf.load_corpus(complain_corpus)

    words = set()
    for (diagnose, _) in diagnoses:
        tfidf.calculate_tfidf(complains_dict[diagnose])
        top = tfidf.get_top_terms(20)
        words |= set(top)
        # print(diagnose, top)
    return list(words)


def tfidf_doctors(dataframe, text_processor, trash_words):
    tfidf = TfidfCalculator(text_processor=text_processor, trash_words=trash_words)

    doctors_corpus = dataframe[DOCTOR_COLUMN].tolist()
    tfidf.load_corpus(doctors_corpus)

    tfidf.calculate_tfidf(doctors_corpus)
    top = tfidf.get_top_terms(34)

    for term in top:
        print(term)

    return top


def get_doctor(doctors, text_processor, line):
    words = text_processor.extract_words_from_text(line)
    for doctor in doctors:
        if doctor in words:
            return doctor
    return 'UNKNOWN'


def get_doctors(dataframe):
    doctors_set = load_list_from_file(DOCTORS_FILENAME)
    doctors_corpus = dataframe[DOCTOR_COLUMN].tolist()
    text_processor = TextProcessor()
    doctors_list = []
    for line in doctors_corpus:
        doc = get_doctor(doctors_set, text_processor, line)
        if doc:
            doctors_list.append(doc)
        else:
            doctors_list.append('UNKNOWN')

    counter = collections.Counter(doctors_list)
    counter = list(counter.items())
    counter.sort(key=lambda tup: tup[1], reverse=True)
    tbl = tabulate(counter)
    print(tbl)


def train_network(dataframe, doctor, text_processor):
    dataset, input_size, output_size = make_dataset(dataframe, doctor, text_processor)
    if output_size > 50:
        hsize1 = 3 * output_size
        hsize2 = 2 * output_size
        batch_ratio = 0.1
    elif output_size > 20:
        hsize1 = 6 * output_size
        hsize2 = 4 * output_size
        batch_ratio = 0.2
    else:
        hsize1 = 100
        hsize2 = 50
        batch_ratio = 0.5

    nn = FeedforwardNetwork(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(hsize1, hsize2),
        batch_ratio=batch_ratio)
    nn.fit(dataset, number_of_epochs=200)
    model_dir = MODELS_DIR + translit(doctor)
    check_make_dir(model_dir)
    nn.save_model(model_dir, 'model')


def check_make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def make_dataset(dataframe, doctor, text_processor, for_train=True):
    dataset = []
    doctors = load_list_from_file(DOCTORS_FILENAME)
    dictionary = load_list_from_file(WORDS_FILENAME)
    words_span = len(dictionary)
    clinic_index_dict = {2: 0, 3: 1, 5: 2, 6: 3, 15: 4, 19: 5}
    age_span = 10
    gender_span = 2
    clinic_span = 6
    # doctors_span = len(doctors)
    size = words_span + age_span + gender_span + clinic_span
    diagnoses = []
    data = []
    for index, row in dataframe.iterrows():
        cur_doctor = get_doctor(doctors, text_processor, row[DOCTOR_COLUMN])
        if cur_doctor != doctor:
            continue

        X = np.zeros((size))
        if for_train:
            diagnose = row[DIAGNOSE_COLUMN]
            diagnoses.append(diagnose)

        complains = row[COMPLAIN_COLUMN]
        words = text_processor.extract_words_from_text(complains)
        for word in words:
            if word in dictionary:
                idx = dictionary.index(word)
                X[idx] = 1.0
        age_pos = get_age_position(row[AGE_COLUMN])
        X[words_span + age_pos] = 1.0
        gender_pos = row[GENDER_COLUMN] - 1
        X[words_span + age_span + gender_pos] = 1.0
        rec_id = row[RECORD_ID_COLUMN]
        clinic_id = row[CLINIC_COLUMN]
        clinic_index = clinic_index_dict[clinic_id]
        X[words_span + age_span + gender_span + clinic_index] = 1.0

        patient_id = row[PATIENT_COLUMN]
        if for_train:
            data.append((rec_id, X, diagnose))
        else:
            data.append((X, rec_id, patient_id))

    if not for_train:
        return data, size

    diagnoses_freq = collections.Counter(diagnoses)
    diagnoses_filtered = list({x: diagnoses_freq[x] for x in diagnoses_freq if diagnoses_freq[x] >= DIAGNOSE_FREQ_THRESHOLD}.keys())

    save_list_to_file(diagnoses_filtered, MODELS_DIR + 'diag_' + doctor + '.txt')

    for (_, X, diagnose) in data:
        if diagnose in diagnoses_filtered:
            idx_diag = diagnoses_filtered.index(diagnose)
            Y = np.zeros((len(diagnoses_filtered)))
            Y[idx_diag] = 1.0
            dataset.append((X, Y))

    return dataset, size, len(diagnoses_filtered)


def predict(dataframe, predictions, doctor, text_processor):
    patient_diagnoses = load_dict_from_file(MOST_FREQUENT_DIAG_FILENAME)
    dataset, input_size = make_dataset(dataframe, doctor, text_processor, for_train=False)
    diagnoses = load_list_from_file(MODELS_DIR + 'diag_' + doctor + '.txt')
    output_size = len(diagnoses)
    nn = FeedforwardNetwork(input_size, output_size, hidden_sizes=())
    model_name = translit(doctor)
    model_dir = MODELS_DIR + model_name
    nn.load_model(model_dir, 'model')
    prediction = nn.predict(dataset)

    for i, (_, rec_id, patient_id) in enumerate(dataset):
        pid = str(patient_id)
        if pid in patient_diagnoses:
            diag = patient_diagnoses[pid]
        else:
            try:
                idx = np.argmax(prediction[i])
                diag = diagnoses[idx]
            except BaseException as e:
                print(doctor + ' !!!')
                raise e
        predictions[rec_id] = diag


def get_age_position(age):
    if age < 10:
        return 0
    if age > 65:
        return 9
    return int((age - 10) / 7) + 1


def store_output(predictions, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[RECORD_ID_COLUMN, DIAGNOSE_COLUMN], delimiter=',')
        writer.writeheader()
        for (rec_id, diag) in predictions:
            writer.writerow({RECORD_ID_COLUMN: rec_id, DIAGNOSE_COLUMN: diag})


def get_most_frequent_diagnose(dataframe, filename):
    data = {}
    for index, row in dataframe.iterrows():
        patient_id = row[PATIENT_COLUMN]
        if not patient_id in data:
            data[patient_id] = []
        data[patient_id].append(row[DIAGNOSE_COLUMN])
    for patient_id in data:
        counter = collections.Counter(data[patient_id])
        freq = list(counter.items())
        freq.sort(key=lambda tup: tup[1], reverse=True)
        data[patient_id] = freq[0][0]
    save_dict_to_file(data, filename)


def main():
    df_test = pd.read_csv(TEST_DATA, sep=';')
    df_train = pd.read_csv(TRAIN_DATA, sep=';')
    text_processor = TextProcessor(typos=typos)
    # get_most_frequent_diagnose(df_train, MOST_FREQUENT_DIAG_FILENAME)
    # exit()
    # complains = load_complains(TRAIN_COMPLAINS)
    # trash_words = load_list_from_file('./data/trash_words.txt')
    # # trash_words_doctors = load_list_from_file('./data/trash_words_doctors.txt')
    # words = tfidf_complains(df_train, text_processor, complains, trash_words)
    # save_list_to_file(words, WORDS_FILENAME)
    # exit()

    if os.path.exists(HANDLED_DOCTORS_FILENAME):
        handled_doctors = load_list_from_file(HANDLED_DOCTORS_FILENAME)
    else:
        handled_doctors = []

    doctors = load_list_from_file(DOCTORS_FILENAME)
    # for doctor in doctors:
    #     if doctor in handled_doctors:
    #         continue
    #     train_network(df_train, doctor, text_processor)
    #     handled_doctors.append(doctor)
    #     save_list_to_file(handled_doctors, HANDLED_DOCTORS_FILENAME)
    #     exit()

    if os.path.exists(TEMP_OUTPUT_FILENAME):
        predictions = load_dict_from_file(TEMP_OUTPUT_FILENAME)
    else:
        predictions = {}

    for doctor in doctors:
        if doctor in handled_doctors:
            continue
        predict(df_test, predictions, doctor, text_processor)
        handled_doctors.append(doctor)
        save_list_to_file(handled_doctors, HANDLED_DOCTORS_FILENAME)
        save_dict_to_file(predictions, TEMP_OUTPUT_FILENAME)
        break

    output = []
    for rec_id in predictions:
        output.append((int(rec_id), predictions[rec_id]))
    output.sort(key=lambda tup: tup[0])

    store_output(output, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()