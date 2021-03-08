import os

import torch

from ucr_dataset import UCRDataset
from utils.constants import NB_CLASSES_LIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTM_FCN(torch.nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, channel_3, kernel_size_1, kernel_size_2, kernel_size_3,
                 hidden_size, num_class):
        super(LSTM_FCN, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size)
        self.dropout = torch.nn.Dropout(0.8)

        self.conv1 = torch.nn.Conv1d(in_channels=in_channel, out_channels=channel_1, kernel_size=kernel_size_1,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv1.weight)
        self.bn1 = torch.nn.BatchNorm1d(channel_1)
        self.conv2 = torch.nn.Conv1d(in_channels=channel_1, out_channels=channel_2, kernel_size=kernel_size_2,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv2.weight)
        self.bn2 = torch.nn.BatchNorm1d(channel_2)
        self.conv3 = torch.nn.Conv1d(in_channels=channel_2, out_channels=channel_3, kernel_size=kernel_size_3,
                                     padding=1)
        torch.nn.init.kaiming_uniform_(self.conv3.weight)
        self.bn3 = torch.nn.BatchNorm1d(channel_3)

        self.label_classifier = torch.nn.Linear(128 + hidden_size, num_class)

    def forward(self, input):
        # print('input', input.shape)
        input = input.unsqueeze(1)
        # print('unsqueeze', input.shape)
        x = input.permute(2, 0, 1)
        # print('permute', x.shape)
        _, (x, cell) = self.lstm(x)
        x = self.dropout(x)

        y = self.conv1(input)
        y = self.bn1(y)
        y = torch.nn.functional.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = torch.nn.functional.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = torch.nn.functional.relu(y)
        y = torch.nn.functional.avg_pool1d(y, kernel_size=y.shape[2])

        # print('x', x.shape)
        # print('y', y.shape)
        feature = torch.cat((x.squeeze(), y.squeeze()), dim=1)
        # print('cat', feature.shape)

        label = self.label_classifier(feature)

        return label


def train_model(model, dname, epochs, batch_size, ucrDataset, K=1):
    model.to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1 / pow(2, 1 / 3), patience=100,
                                               verbose=True,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.0001)
    for k in range(K):
        dataloader = torch.utils.data.DataLoader(ucrDataset.getDatasetByName(dname)['train'], batch_size=batch_size,
                                                 shuffle=True)
        for epoch in range(epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y.squeeze())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(" train loss:", total_loss, "epoch:", k * epochs + epoch)
        # batch_size //= 2

    if total_loss < 1:
        if not os.path.exists('/content/drive/MyDrive/auto_aug/saved_model/%s' % dname):
            os.makedirs('/content/drive/MyDrive/auto_aug/saved_model/%s' % dname)
        torch.save(model, '/content/drive/MyDrive/auto_aug/saved_model/%s/%s_%f.pkl' % (dname, dname, total_loss))

    return total_loss


def test_model(model, dname, batch_size, ucrDataset):
    dataset = ucrDataset.getDatasetByName(dname)['test']
    test_length = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    correct = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            # loss = loss_fn(pred, y)
            output = pred.argmax(dim=1)
            # print(output.shape[0])
            # y = y[:output.shape[0]]
            # print(y.shape)
            # print((output == y).float())
            if output.shape[0] < batch_size:
                correct += torch.cat(
                    ((output == y), torch.full([batch_size - output.shape[0]], 0, dtype=torch.long).to(device)),
                    dim=0).float()
            else:
                correct += (output == y).float()
            # print(correct)
    # print(correct)
    acc = correct.sum() / test_length
    print(' acc:', acc)
    return acc


def main():
    ucrDataset = UCRDataset(
        data_path='/content/drive/MyDrive/datasets/UCRArchive_2018',
        normalize=True,
        train_ratio=1,
        num_of_dataset=2,
        data_name_list=['Adiac', 'Beef']
    )
    dataset_map = {'Adiac': 0,
                   'ArrowHead': 1,
                   'ChlorineConcentration': 2,
                   'InsectWingbeatSound': 3,
                   'Lightning7': 4,
                   'Wine': 5,
                   'WordSynonyms': 6,
                   # '50words': 7,
                   'Beef': 8,
                   'DistalPhalanxOutlineAgeGroup': 9,
                   'DistalPhalanxOutlineCorrect': 10,
                   'DistalPhalanxTW': 11,
                   'ECG200': 12,
                   'ECGFiveDays': 13,
                   'BeetleFly': 14,
                   'BirdChicken': 15,
                   'ItalyPowerDemand': 16,
                   'SonyAIBORobotSurface1': 17,
                   'SonyAIBORobotSurface2': 18,
                   'MiddlePhalanxOutlineAgeGroup': 19,
                   'MiddlePhalanxOutlineCorrect': 20,
                   'MiddlePhalanxTW': 21,
                   'ProximalPhalanxOutlineAgeGroup': 22,
                   'ProximalPhalanxOutlineCorrect': 23,
                   'ProximalPhalanxTW': 24,
                   'MoteStrain': 25,
                   'MedicalImages': 26,
                   'Strawberry': 27,
                   'ToeSegmentation1': 28,
                   'Coffee': 29,
                   'CricketX': 30,
                   'CricketY': 31,
                   'CricketZ': 32,
                   'UWaveGestureLibraryX': 33,
                   'UWaveGestureLibraryY': 34,
                   'UWaveGestureLibraryZ': 35,
                   'ToeSegmentation2': 36,
                   'DiatomSizeReduction': 37,
                   'Car': 38,
                   'CBF': 39,
                   'CinCECGTorso': 40,
                   'Computers': 41,
                   'Earthquakes': 42,
                   'ECG5000': 43,
                   'ElectricDevices': 44,
                   'FaceAll': 45,
                   'FaceFour': 46,
                   'FacesUCR': 47,
                   'Fish': 48,
                   'FordA': 49,
                   'FordB': 50,
                   'GunPoint': 51,
                   'Ham': 52,
                   'HandOutlines': 53,
                   'Haptics': 54,
                   'Herring': 55,
                   'InlineSkate': 56,
                   'LargeKitchenAppliances': 57,
                   'Lightning2': 58,
                   'Mallat': 59,
                   'Meat': 60,
                   'NonInvasiveFetalECGThorax1': 61,
                   'NonInvasiveFetalECGThorax2': 62,
                   'OliveOil': 63,
                   'OSULeaf': 64,
                   'PhalangesOutlinesCorrect': 65,
                   'Phoneme': 66,
                   'Plane': 67,
                   'RefrigerationDevices': 68,
                   'ScreenType': 69,
                   'ShapeletSim': 70,
                   'ShapesAll': 71,
                   'SmallKitchenAppliances': 72,
                   'StarLightCurves': 73,
                   'SwedishLeaf': 74,
                   'Symbols': 75,
                   'SyntheticControl': 76,
                   'Trace': 77,
                   # 'Patterns': 78,
                   'TwoLeadECG': 79,
                   'UWaveGestureLibraryAll': 80,
                   'Wafer': 81,
                   'Worms': 82,
                   'WormsTwoClass': 83,
                   'Yoga': 84,
                   'ACSF1': 85,
                   'AllGestureWiimoteX': 86,
                   'AllGestureWiimoteY': 87,
                   'AllGestureWiimoteZ': 88,
                   'BME': 89,
                   'Chinatown': 90,
                   'Crop': 91,
                   'DodgerLoopDay': 92,
                   'DodgerLoopGame': 93,
                   'DodgerLoopWeekend': 94,
                   'EOGHorizontalSignal': 95,
                   'EOGVerticalSignal': 96,
                   'EthanolLevel': 97,
                   'FreezerRegularTrain': 98,
                   'FreezerSmallTrain': 99,
                   'Fungi': 100,
                   'GestureMidAirD1': 101,
                   'GestureMidAirD2': 102,
                   'GestureMidAirD3': 103,
                   'GesturePebbleZ1': 104,
                   'GesturePebbleZ2': 105,
                   'GunPointAgeSpan': 106,
                   'GunPointMaleVersusFemale': 107,
                   'GunPointOldVersusYoung': 108,
                   'HouseTwenty': 109,
                   'InsectEPGRegularTrain': 110,
                   'InsectEPGSmallTrain': 111,
                   'MelbournePedestrian': 112,
                   'MixedShapesRegularTrain': 113,
                   # 'MixedShapesSmallTrain': 114,
                   'PickupGestureWiimoteZ': 115,
                   'PigAirwayPressure': 116,
                   'PigArtPressure': 117,
                   'PigCVP': 118,
                   'PLAID': 119,
                   'PowerCons': 120,
                   'Rock': 121,
                   'SemgHandGenderCh2': 122,
                   'SemgHandMovementCh2': 123,
                   'SemgHandSubjectCh2': 124,
                   'ShakeGestureWiimoteZ': 125,
                   'SmoothSubspace': 126,
                   'UMD': 127
                   }

    print("Num datasets : ", len(dataset_map))
    print()

    base_log_name = '%s_%d_cells_new_datasets.csv'

    MODELS = [
        ('lstmfcn', LSTM_FCN)
        # ('alstmfcn', generate_alstmfcn),
    ]

    # Number of cells
    CELLS = [64]

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            successes = []
            failures = []

            if not os.path.exists('/content/drive/MyDrive/auto_aug/result/'):
                os.makedirs('/content/drive/MyDrive/auto_aug/result/')
            if not os.path.exists('/content/drive/MyDrive/auto_aug/result/' + base_log_name % (MODEL_NAME, cell)):
                file = open('/content/drive/MyDrive/auto_aug/result/' + base_log_name % (MODEL_NAME, cell), 'w')
                file.write('%s,%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy', 'loss'))
                file.close()
            for dname in ucrDataset.getNameList():
                did = dataset_map[dname]
                NB_CLASS = NB_CLASSES_LIST[did]

                file = open(base_log_name % (MODEL_NAME, cell), 'a+')

                model = model_fn(1, 128, 256, 128, 8, 5, 3, cell, NB_CLASS)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                loss = train_model(model, dname, epochs=2000, batch_size=128, ucrDataset=ucrDataset)

                acc = test_model(model, dname, batch_size=128, ucrDataset=ucrDataset)

                s = "%d,%s,%s,%0.6f,%0.6f\n" % (did, dname, dname, acc, loss)

                file.write(s)
                file.flush()

                successes.append(s)

                file.close()
                del model

            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            for line in successes:
                print(line)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)
            print()

            for line in failures:
                print(line)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
