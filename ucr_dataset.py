import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class UCRDataset():
    def __init__(self, data_path: str, train_ratio: float, normalize: bool, num_of_dataset=200, data_name_list=[], is_on_the_colabpratory=False):
        self.dict = {}
        cur_num_of_dataset = 0
        for dir in os.listdir(data_path):
            if is_on_the_colabpratory:
                from drive.MyDrive.auto_aug.auto_aug.utils.constants import MAX_SEQUENCE_LENGTH_LIST, dataset_map
                data_path = '/content/drive/MyDrive/datasets/UCRArchive_2018'
            else:
                from utils.constants import MAX_SEQUENCE_LENGTH_LIST, dataset_map
                data_path = 'UCRArchive_2018'
            did = dataset_map[dir]
            if os.path.isdir(os.path.join(data_path, dir)) and (len(data_name_list) == 0 or dir in data_name_list) and ((is_on_the_colabpratory and MAX_SEQUENCE_LENGTH_LIST[did] > 400) or (not is_on_the_colabpratory and MAX_SEQUENCE_LENGTH_LIST[did] <= 400)):
                train_and_valid_data = self.__sortByCategory(
                    np.loadtxt(os.path.join(data_path, dir, dir + '_TRAIN.tsv'), delimiter='\t', dtype=np.float32))
                test_data = self.__sortByCategory(
                    np.loadtxt(os.path.join(data_path, dir, dir + '_TEST.tsv'), delimiter='\t', dtype=np.float32))

                train_and_valid_signals = torch.from_numpy(train_and_valid_data[:, 1:]).float()
                train_and_valid_labels = torch.from_numpy(train_and_valid_data[:, 0]).long()

                test_signals = torch.from_numpy(test_data[:, 1:]).float()
                test_labels = torch.from_numpy(test_data[:, 0]).long()

                if train_and_valid_data[0][0] == 1:
                    train_and_valid_labels = train_and_valid_labels - 1
                    test_labels = test_labels - 1

                if normalize:
                    train_and_valid_signals = (train_and_valid_signals - torch.mean(
                        train_and_valid_signals)) / torch.std(train_and_valid_signals)
                    test_signals = (test_signals - torch.mean(test_signals)) / torch.std(test_signals)

                # print(train_and_valid_signals)
                num_classes = train_and_valid_labels[-1].item() - train_and_valid_labels[0].item() + 1
                cur_category = 0
                former_pos = 0
                for cur_pos in range(len(train_and_valid_signals)):
                    if train_and_valid_labels[cur_pos].item() > cur_category:
                        num = cur_pos - former_pos
                        train_num = round(num * train_ratio)
                        # valid_num = num - train_num
                        if cur_category == 0:
                            train_signals = train_and_valid_signals[former_pos: former_pos + train_num]
                            train_labels = train_and_valid_labels[former_pos: former_pos + train_num]

                            valid_signals = train_and_valid_signals[former_pos + train_num: cur_pos]
                            valid_labels = train_and_valid_labels[former_pos + train_num: cur_pos]
                        else:
                            train_signals = torch.cat(
                                (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
                            train_labels = torch.cat(
                                (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)

                            valid_signals = torch.cat(
                                (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos]), 0)
                            valid_labels = torch.cat(
                                (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos]), 0)
                        cur_category = train_and_valid_labels[cur_pos].item()
                        former_pos = cur_pos

                num = len(train_and_valid_signals) - former_pos
                train_num = round(num * train_ratio)
                # valid_num = num - train_num
                train_signals = torch.cat(
                    (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
                train_labels = torch.cat(
                    (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)

                valid_signals = torch.cat(
                    (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos + 1]), 0)
                valid_labels = torch.cat(
                    (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos + 1]), 0)
            # print(train_signals)
                self.dict[dir] = {
                    'num_classes': num_classes,
                    'train': torch.utils.data.TensorDataset(train_signals, train_labels),
                    'valid': torch.utils.data.TensorDataset(valid_signals, valid_labels),
                    'test': torch.utils.data.TensorDataset(test_signals, test_labels),
                }
                cur_num_of_dataset = cur_num_of_dataset + 1
                if cur_num_of_dataset == num_of_dataset:
                    break
        # print(self.dict['ACSF1']['train'][:])

    # def __init__(self, data_path: str, train_ratio: float, normalize: bool, data_name_list):
    #     self.dict = {}
    #     for dir in os.listdir(data_path):
    #         if os.path.isdir(os.path.join(data_path, dir)) and dir in data_name_list:
    #             train_and_valid_data = self.__sortByCategory(
    #                 np.loadtxt(os.path.join(data_path, dir, dir + '_TRAIN.tsv'), delimiter='\t', dtype=np.float32))
    #             test_data = self.__sortByCategory(
    #                 np.loadtxt(os.path.join(data_path, dir, dir + '_TEST.tsv'), delimiter='\t', dtype=np.float32))
    #
    #             train_and_valid_signals = torch.from_numpy(train_and_valid_data[:, 1:]).float()
    #             train_and_valid_labels = torch.from_numpy(train_and_valid_data[:, 0]).long()
    #
    #             test_signals = torch.from_numpy(test_data[:, 1:]).float()
    #             test_labels = torch.from_numpy(test_data[:, 0]).long()
    #
    #             if train_and_valid_data[0][0] == 1:
    #                 train_and_valid_labels = train_and_valid_labels - 1
    #                 test_labels = test_labels - 1
    #
    #             if normalize:
    #                 train_and_valid_signals = (train_and_valid_signals - torch.mean(
    #                     train_and_valid_signals)) / torch.std(train_and_valid_signals)
    #                 test_signals = (test_signals - torch.mean(test_signals)) / torch.std(test_signals)
    #
    #             # print(train_and_valid_signals)
    #             num_classes = train_and_valid_labels[-1].item() - train_and_valid_labels[0].item() + 1
    #             cur_category = 0
    #             former_pos = 0
    #             for cur_pos in range(len(train_and_valid_signals)):
    #                 if train_and_valid_labels[cur_pos].item() > cur_category:
    #                     num = cur_pos - former_pos
    #                     train_num = round(num * train_ratio)
    #                     # valid_num = num - train_num
    #                     if cur_category == 0:
    #                         train_signals = train_and_valid_signals[former_pos: former_pos + train_num]
    #                         train_labels = train_and_valid_labels[former_pos: former_pos + train_num]
    #
    #                         valid_signals = train_and_valid_signals[former_pos + train_num: cur_pos]
    #                         valid_labels = train_and_valid_labels[former_pos + train_num: cur_pos]
    #                     else:
    #                         train_signals = torch.cat(
    #                             (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
    #                         train_labels = torch.cat(
    #                             (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)
    #
    #                         valid_signals = torch.cat(
    #                             (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos]), 0)
    #                         valid_labels = torch.cat(
    #                             (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos]), 0)
    #                     cur_category = train_and_valid_labels[cur_pos].item()
    #                     former_pos = cur_pos
    #
    #             num = len(train_and_valid_signals) - former_pos
    #             train_num = round(num * train_ratio)
    #             # valid_num = num - train_num
    #             train_signals = torch.cat(
    #                 (train_signals, train_and_valid_signals[former_pos: former_pos + train_num]), 0)
    #             train_labels = torch.cat(
    #                 (train_labels, train_and_valid_labels[former_pos: former_pos + train_num]), 0)
    #
    #             valid_signals = torch.cat(
    #                 (valid_signals, train_and_valid_signals[former_pos + train_num: cur_pos + 1]), 0)
    #             valid_labels = torch.cat(
    #                 (valid_labels, train_and_valid_labels[former_pos + train_num: cur_pos + 1]), 0)
    #         # print(train_signals)
    #         self.dict[dir] = {
    #             'num_classes': num_classes,
    #             'train': torch.utils.data.TensorDataset(train_signals, train_labels),
    #             'valid': torch.utils.data.TensorDataset(valid_signals, valid_labels),
    #             'test': torch.utils.data.TensorDataset(test_signals, test_labels),
    #         }
    #         cur_num_of_dataset = cur_num_of_dataset + 1

    def getDatasetByName(self, name):
        return self.dict[name]

    def getNameList(self):
        return self.dict.keys()

    def getAllDataset(self):
        return self.dict

    def __sortByCategory(self, data):
        # print(data)
        reverse_transpose = data[:, ::-1].T
        index = np.lexsort(reverse_transpose)
        # print(data[index])
        return data[index]


if __name__ == '__main__':
    ucrDataset = UCRDataset(
        data_path='UCRArchive_2018',
        normalize=True,
        train_ratio=0.5,
    )
    print(ucrDataset.getDatasetByName('DistalPhalanxTW')['num_classes'])
    print(ucrDataset.getDatasetByName('DistalPhalanxTW')['train'][:])
    print(ucrDataset.getDatasetByName('DistalPhalanxTW')['valid'][:])
    print(ucrDataset.getDatasetByName('DistalPhalanxTW')['test'][:])

    # for key, value in ucrDataset.getAllDataset().items():
    #     print(key)
    #     print(value['num_classes'])
    #     print(value['train'][:])
    #     print(value['valid'][:])
    #     print(value['test'][:])
