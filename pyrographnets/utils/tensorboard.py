from os.path import isdir

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    SummaryWriter = None

if SummaryWriter is None:

    def new_writer(*args, **kwargs):
        raise ImportError("`tensorboard` not installed")


else:

    def new_writer(directory: str, suffix=""):
        i = 0

        def name(index):
            return directory + "%04d" % index + suffix

        while isdir(name(i)):
            i += 1
        dirname = name(i)
        print("New writer at '{}'".format(dirname))
        return SummaryWriter(dirname)
