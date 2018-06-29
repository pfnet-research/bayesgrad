import numpy


def save_npz(filepath, datasets):
    if not isinstance(datasets, (list, tuple)):
        datasets = (datasets, )
    numpy.savez(filepath, *datasets)


def load_npz(filepath):
    load_data = numpy.load(filepath)
    result = []
    i = 0
    while True:
        key = 'arr_{}'.format(i)
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    if len(result) == 1:
        result = result[0]
    return result
