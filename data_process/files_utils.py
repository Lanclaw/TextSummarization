import config
import jieba
import json
import os


def read_samples(filename):
    samples = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            samples.append(line.strip())

    return samples


def write_samples(filename, samples, opt='w'):
    with open(filename, opt, encoding='utf8') as f:
        for line in samples:
            f.write(line)
            f.write('\n')


def json2txt():
    if os.path.exists(config.samples_path) is not True:
        with open(config.json_data_path, 'r', encoding='utf8') as file:
            jsf = json.load(file)

        samples = set()

        for jsobj in jsf.values():
            title = jsobj['title'] + ' '
            kb = jsobj['kb'].items()
            kb_merged = ''
            for key, val in kb:
                kb_merged += key + ' ' + val + ' '  # Merge attributes.

            ocr = ' '.join(list(jieba.cut(jsobj['ocr'])))

            text = title + ocr + kb_merged
            reference = ' '.join(list(jieba.cut(jsobj['reference'])))
            sample = text + '<sep>' + reference  # Seperate source and reference.
            samples.add(sample)

        write_samples(config.samples_path, samples)


def partition(samples):
    if os.path.exists(config.test_data_path) is not True:
        train, val, test = [], [], []
        count = 0
        for sample in samples:
            count += 1
            if count < 1000:
                test.append(sample)
            elif count < 6000:
                val.append(sample)
            else:
                train.append(sample)

        write_samples(config.train_data_path, train)
        write_samples(config.val_data_path, val)
        write_samples(config.test_data_path, test)


if __name__ == '__main__':
    json2txt()
    samples = read_samples(config.samples_path)
    partition(samples)



