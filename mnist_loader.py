import torch
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_data(batch_size=100, flipped=False, outline=False, normalize=False):
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    if flipped:
        train_dataset = list(map(lambda x: (abs(1-x[0]), x[1]), train_dataset))
        test_dataset = list(map(lambda x: (abs(1-x[0]), x[1]), test_dataset))
    if normalize:
        train_dataset = list(map(lambda x: (x[0]-.5, x[1]), train_dataset))
        test_dataset = list(map(lambda x: (x[0]-.5, x[1]), test_dataset))
    if outline:
        train_dataset = list(map(lambda x: (1-(abs(.5-x[0]) * 2), x[1]), train_dataset))
        test_dataset = list(map(lambda x: (1-(abs(.5-x[0]) * 2), x[1]), test_dataset))

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    indices = range(len(test_dataset))
    indices_val = indices[:5000]
    data.TensorDataset()
    indices_test = indices[5000:]

    sampler_val = sampler.SubsetRandomSampler(indices_val)
    sampler_test = sampler.SubsetRandomSampler(indices_test)

    # not sure if there is any benefit to batching validation and test data
    validation_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                    sampler=sampler_val)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                              sampler=sampler_test)

    return train_loader, validation_loader, test_loader