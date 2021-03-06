import json
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

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


def train_model(model, dname, epochs, batch_size, ucrDataset, patience, K=1):
    writer = SummaryWriter(comment="-" + dname)
    model.to(device)
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1 / pow(2, 1 / 3),
                                                           patience=patience,
                                                           verbose=True,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=0.0001)
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
            scheduler.step(total_loss)
            print(" train loss:", total_loss, "epoch:", k * epochs + epoch)

            acc = test_model(model, dname, batch_size=batch_size, ucrDataset=ucrDataset)
            writer.add_scalar('loss', total_loss, global_step=k * epochs + epoch)
            writer.add_scalar('acc', acc, global_step=k * epochs + epoch)
            model.train()
        # batch_size //= 2
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
    acc = (correct.sum() / test_length).item()
    print(dname + ' acc:', acc)
    return acc


def main(is_on_the_colabpratory, epochs=2000, batch_size=128, cell=64, is_aug=False, num_of_dataset=200,
         data_name_list=[], patience=100):
    if is_on_the_colabpratory:
        from drive.MyDrive.auto_aug.auto_aug.ucr_dataset import UCRDataset
        from drive.MyDrive.auto_aug.auto_aug.utils.constants import NB_CLASSES_LIST, MAX_SEQUENCE_LENGTH_LIST, \
            dataset_map
        data_path = '/content/drive/MyDrive/datasets/UCRArchive_2018'
    else:
        from ucr_dataset import UCRDataset
        from utils.constants import NB_CLASSES_LIST, MAX_SEQUENCE_LENGTH_LIST, dataset_map
        data_path = 'UCRArchive_2018'
    ucrDataset = UCRDataset(
        data_path=data_path,
        normalize=True,
        train_ratio=1,
        num_of_dataset=num_of_dataset,
        data_name_list=data_name_list,
        is_on_the_colabpratory=is_on_the_colabpratory
    )

    print("Num datasets : ", len(dataset_map))
    print()

    base_log_name = '%s_%d_cells_new_datasets.csv'

    MODELS = [
        ('lstmfcn', LSTM_FCN)
        # ('alstmfcn', generate_alstmfcn),
    ]

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        successes = []
        failures = []
        if (is_on_the_colabpratory):
            result_path = '/content/drive/MyDrive/auto_aug/auto_aug/result/'
            saved_model_path = '/content/drive/MyDrive/auto_aug/auto_aug/saved_model/'
        else:
            result_path = './result/'
            saved_model_path = './saved_model/'
        # if not os.path.exists(result_path):
        #     os.makedirs(result_path)
        # if not os.path.exists(
        #         result_path + base_log_name % (MODEL_NAME, cell)):
        #     file = open(result_path + base_log_name % (MODEL_NAME, cell),
        #                 'w')
        #     file.write(
        #         '%s,%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy', 'loss'))
        #     file.close()
        for dname in ucrDataset.getNameList():
            try:
                did = dataset_map[dname]
                NB_CLASS = NB_CLASSES_LIST[did]
                # if not is_on_the_colabpratory and MAX_SEQUENCE_LENGTH_LIST[did] > 400:
                #     continue
                #
                # if is_on_the_colabpratory and MAX_SEQUENCE_LENGTH_LIST[did] <= 400:
                #     continue

                # file = open(result_path + base_log_name % (MODEL_NAME, cell), 'a+')

                model = model_fn(1, 128, 256, 128, 8, 5, 3, cell, NB_CLASS)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                loss = train_model(model, dname, epochs=epochs, batch_size=batch_size, ucrDataset=ucrDataset,
                                   patience=patience)

                acc = test_model(model, dname, batch_size=batch_size, ucrDataset=ucrDataset)

                try:
                    with open("hyperparameters_of_model.json", mode="r", encoding="utf-8") as f:
                        hyperparameters_of_model = json.loads(f.read())
                    # hyperparameters_of_model = np.load('hyperparameters_of_model.npy', allow_pickle=True).item()
                except:
                    hyperparameters_of_model = {}

                if (not is_aug and (dname not in hyperparameters_of_model.keys() or acc >
                                    hyperparameters_of_model[dname]['acc'])):
                    hyperparameter = {}
                    hyperparameter['acc'] = acc
                    hyperparameter['loss'] = loss
                    hyperparameter['batch_size'] = batch_size
                    hyperparameter['epochs'] = epochs
                    hyperparameter['cell'] = cell
                    hyperparameters_of_model[dname] = hyperparameter
                    # np.save('hyperparameters_of_model.npy', hyperparameters_of_model)
                    with open("hyperparameters_of_model.json", mode="w", encoding="utf-8") as f:
                        f.write(json.dumps(hyperparameters_of_model))
                    if not os.path.exists(saved_model_path + '%s' % dname):
                        os.makedirs(saved_model_path + '%s' % dname)
                    torch.save(model, saved_model_path + '%s/%s_acc%f_e%d_b%d_c%d_p%d.pkl' % (
                        dname, dname, acc, epochs, batch_size, cell, patience))
                s = "%d,%s,%s,%0.6f,%0.6f\n" % (did, dname, dname, acc, loss)

                # file.write(s)
                # file.flush()

                successes.append(s)

                # file.close()
                del model
            except:
                pass

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
    main(is_on_the_colabpratory=False, epochs=2000, batch_size=128, cell=64, is_aug=False, num_of_dataset=200,
         data_name_list=[], patience=100)
