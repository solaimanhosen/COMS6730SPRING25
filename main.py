import torch
import argparse
from network_with_dropout import ResNet
from model import set_random, load_data, train, test

def get_args():
    parser = argparse.ArgumentParser(description='Args for training networks')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-resnet_version', type=int, default=1, help='ResNet version')
    parser.add_argument('-resnet_size', type=int, default=3, help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument('-num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('-num_epochs', type=int, default=20, help='num epochs')
    parser.add_argument('-batch', type=int, default=16, help='batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('-drop', type=float, default=0.3, help='dropout rate')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print(args.resnet_version, args.num_epochs)
    set_random(args.seed)
    trainloader, testloader = load_data(args.batch)

    for lr in [0.08, 0.01, 0.05]:
        for drop in [0.1, 0.3]:
            for epochs in [30]:
                args.lr = lr
                args.drop = drop
                args.num_epochs = epochs

                net = ResNet(args)
                if torch.cuda.is_available():
                    net.cuda()
                train(args, net, trainloader)

                print('lr={} drop={} epochs={}'.format(lr, drop, epochs))
                test(net, testloader)
