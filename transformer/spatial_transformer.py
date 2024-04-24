# This is spatial transformer network
# This is the implementation of the paper "Spatial Transformer Networks" by Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
# Ref: https://github.com/AdaCompNUS/pfnet/blob/master/transformer/spatial_transformer.py
# This is the Pytorch version of the code provided in PFNet

import torch

def transformer(U, theta, out_size):

    # Transformer layer
    def repeat(x, n_repeats):
        rep = torch.ones((n_repeats,), dtype=torch.int32).unsqueeze(0)
        x = torch.mm(x.view((-1, 1)), rep)
        return x.view(-1)

    def _interpolate(im, x, y, downsample_factor):
        # constants
        num_batch, height, width, channels = im.shape
        height_f = torch.tensor(height, dtype=torch.float32)
        width_f = torch.tensor(width, dtype=torch.float32)
        out_height = torch.tensor(height_f // downsample_factor, dtype=torch.int64)
        out_width = torch.tensor(width_f // downsample_factor, dtype=torch.int64)
        zero = torch.tensor(0, dtype=torch.int64)
        max_y = torch.tensor(im.shape[1] - 1, dtype=torch.int64)
        max_x = torch.tensor(im.shape[2] - 1, dtype=torch.int64)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).to(torch.int64)
        x1 = x0 + 1
        y0 = torch.floor(y).to(torch.int64)
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width*height
        base = repeat(torch.arange(num_batch, dtype=torch.int64)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.view((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.to(torch.float32)
        x1_f = x1.to(torch.float32)
        y0_f = y0.to(torch.float32)
        y1_f = y1.to(torch.float32)
        wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
        wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
        wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
        wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)
        output = wa*Ia + wb*Ib + wc*Ic + wd*Id
        return output

    def _linespace(start, end, num):
        start = start.to(torch.float32)
        end = end.to(torch.float32)
        num = num.to(torch.float32)
        step = (end - start) / (num - 1)
        return torch.arange(num, dtype=torch.float32)*step + start

    def _meshgrid(height, width):
        x_t = torch.matmul(torch.ones((height, 1)), torch.linspace(-1.0, 1.0, width).reshape(1, -1))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).reshape(-1, 1), torch.ones((1, width)))

        x_t_flat = x_t.reshape((1, -1))
        y_t_flat = y_t.reshape((1, -1))
        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], dim=0)
        return grid

    def _transform(theta, input_dim, downsample_factor):
        num_batch, num_channels, height, width = input_dim
        theta = theta.view(-1, 2, 3)
        theta = theta / downsample_factor

        height_f = height.to(torch.float32)
        width_f = width.to(torch.float32)
        out_height = (height_f // downsample_factor).to(torch.int64)
        out_width = (width_f // downsample_factor).to(torch.int64)
        grid = _meshgrid(out_height, out_width)

        # transform a x (x_t, y_t, 1)^t -> (x_s, y_s)
        T_g = torch.dot(theta, grid)
        x_s = T_g[:, 0]
        y_s = T_g[:, 1]
        x_s_flat = x_s.reshape(1, -1)
        y_s_flat = y_s.reshape(1, -1)

        # dimshuffle input to  (bs, height, width, channels)
        input_dim = input_dim.permute(0, 2, 3, 1)
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, downsample_factor)
        output = input_transformed.reshape(num_batch, out_height, out_width, num_channels)
        output = output.permute(0, 3, 1, 2)
        return output
    
    output = _transform(theta, U, out_size)
    return output


def batch_transformer(U, theta, out_size):
    num_batch, num_transforms = map(int, list(theta.size())[:2])
    indices = [[i]*num_transforms for i in range(num_batch)]
    indices = torch.tensor(indices).reshape(-1)
    input_repeated = torch.index_select(U, 0, indices)
    return transformer(input_repeated, theta, out_size)