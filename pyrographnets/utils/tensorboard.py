from os.path import isdir

from torch.utils.tensorboard import SummaryWriter


def new_writer(directory: str, suffix=""):
    i = 0

    def name(index):
        return directory + "%04d" % index + suffix

    while isdir(name(i)):
        i += 1
    dirname = name(i)
    print("New writer at '{}'".format(dirname))
    return SummaryWriter(dirname)
