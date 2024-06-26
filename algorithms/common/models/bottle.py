# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple, **kwargs):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)), **kwargs)
    if isinstance(y, tuple):
        output = (item.view(x_sizes[0][0], x_sizes[0][1], *item.size()[1:]) for item in y)
    else:
        y_size = y.size()
        output = y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])
    return output
