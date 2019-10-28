import os
import numpy as np
import pandas as pd
from time import time as tc

def get_iou(labels, preds):
    iou_mean = np.zeros(labels.shape[1] - 1)
    iou_sum_mean = 0
    for i in range(labels.shape[0]):
        intersection = (labels[i] == 1) & (preds[i] == 1)
        union = (labels[i] == 1) | (preds[i] == 1)
        iou = np.nan_to_num(intersection.sum(axis=1) / union.sum(axis=1))
        iou_mean += iou[1:]
        iou_sum_mean += np.mean(iou_mean)
    iou_mean = np.hstack((iou_sum_mean, iou_mean))
    return iou_mean / labels.shape[0]

def generatebatch(X, Y, batch_size, random_state=0):
    import numpy as np
    n_examples = X.shape[0]
    np.random.seed(random_state)
    ind = np.random.permutation(n_examples) # 随机打乱样本顺序
    ind = np.int32(ind)
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[ind[start:end]]
        batch_ys = Y[ind[start:end]]
        yield batch_xs, batch_ys # 生成每一个batch


def label2hot(label, class_num):
    result = np.zeros((label.shape[0], class_num, label.shape[1]))
    for i in range(label.shape[0]):
        index = np.array(label[i]).reshape(-1).astype(np.int32)
        result[i, ...] = np.eye(class_num)[index].T
    return result

def file_merge(src_path, dst_path, batch_size=100, is_training=True):
    intensity_path = os.path.join(src_path, 'intensity')
    pts_path = os.path.join(src_path, 'pts')
    category_path = os.path.join(src_path, 'category')

    fns = [fn for fn in os.listdir(pts_path) if fn[-4:] == '.csv']
    n_files = len(fns)

    flame_index = 0
    tic = tc()
    for batch_i in range(int(np.ceil(n_files / batch_size))):
        print('batch: {}/{} {:.2f}s.'.format(batch_i + 1, round(n_files / batch_size + 0.5), tc() - tic))
        start = batch_i * batch_size
        end = start + batch_size

        index, pts, intensity, category = [], [], [], []
        test_fn_index = []
        for fn in fns[start:end]:
            tmp1 = np.loadtxt(os.path.join(pts_path, fn), dtype=np.float32, delimiter=',')
            pts.append(tmp1)
            tmp2 = np.loadtxt(os.path.join(intensity_path, fn), dtype=np.float32).reshape((-1, 1))
            intensity.append(tmp2)
            if is_training:
                category.append(np.loadtxt(os.path.join(category_path, fn), dtype=np.float32).reshape((-1, 1)))
            index.append(flame_index*np.ones_like(tmp2, np.float32))
            test_fn_index.append([flame_index, fn])
            flame_index += 1

        index = np.vstack(index)
        pts = np.vstack(pts)
        intensity = np.vstack(intensity)
        if is_training:
            category = np.vstack(category)
            merged_data = np.hstack((index, pts, intensity, category))
            save_path = os.path.join(dst_path, 'train_batch_{}.npy'.format(str(batch_i)))
        else:
            merged_data = np.hstack((index, pts, intensity))
            save_path = os.path.join(dst_path, 'test_batch_{}.npy'.format(str(batch_i)))
            pd.DataFrame(test_fn_index).to_csv(
                os.path.join(dst_path, 'test_batch_{}.name'.format(str(batch_i))),
                header=None,
                index=False,
                encoding='utf-8'
            )

        np.float32(merged_data).tofile(save_path)


def npy_read(path, is_training=True):
    fns = sorted([fn for fn in os.listdir(path) if fn[-4:] == '.npy'])
    test_fn_index = None
    n_files = len(fns)
    for i, fn in enumerate(fns):
        print('( {}/{} ):{}'.format(i+1, n_files, fn))
        data = np.fromfile(os.path.join(path, fn), np.float32)
        if is_training:
            data = data.reshape((-1, 6))
        else:
            data = data.reshape((-1, 5))
            test_fn_index = pd.read_csv(os.path.join(path, fn[:-4] + '.name'), header=None, index_col=0)

        yield data, test_fn_index

def npy_item_read(path, idx=0, is_training=True):
    test_fn_index = None
    if is_training:
        fn = '{}_batch_{}.npy'.format('train', idx)
        data = np.fromfile(os.path.join(path, fn), np.float32)
        data = data.reshape((-1, 6))
    else:
        fn = '{}_batch_{}.npy'.format('test', idx)
        data = np.fromfile(os.path.join(path, fn), np.float32)
        data = data.reshape((-1, 5))
        test_fn_index = pd.read_csv(os.path.join(path, fn[:-4] + '.name'), header=None, index_col=0)

    return data, test_fn_index

def flame_len_fix(pts, intensity, category, flame_size=500000):

    fix_len = flame_size - pts.shape[0]
    if fix_len > 0:
        fix_head = int(fix_len * 0.5)
        fix_tail = fix_len - fix_head

        pts = np.vstack((pts[-fix_head:, :], pts, pts[:fix_tail, :]))
        intensity = np.hstack((intensity[- fix_head:], intensity, intensity[:fix_tail])).reshape((-1, 1))
        category = np.hstack((category[- fix_head:], category, category[:fix_tail]))
        # print('fixed shape:', pts.shape, intensity.shape, category.shape)
    return pts, intensity, category

def flame_len_fix_test(pts, intensity, flame_size=500000):

    fix_len = flame_size - pts.shape[0]
    fix_head = int(fix_len * 0.5)
    fix_tail = fix_len - fix_head
    if fix_len > pts.shape[0]:
        fix_head = 0
        fix_tail = - pts.shape[0]
        stack_len = round(flame_size / pts.shape[0] + 0.5)
        print(stack_len)
        pts = np.vstack([pts for _ in range(stack_len)])
        intensity = np.hstack([intensity for _ in range(stack_len)]).reshape((-1, 1))
        pts = pts[:flame_size]
        intensity = intensity[:flame_size].reshape((-1, 1))
    elif fix_len > 0:
        pts = np.vstack((pts[-fix_head:, :], pts, pts[:fix_tail, :]))
        intensity = np.hstack((intensity[- fix_head:], intensity, intensity[:fix_tail])).reshape((-1, 1))

    return pts, intensity, fix_head, fix_tail

def genBatchData(data, flame_idxs, num_points):
    X = np.zeros((len(flame_idxs), num_points, 4))
    Y = np.zeros((len(flame_idxs), num_points))
    for i, flame_idx in enumerate(flame_idxs):
        flame_data = data[np.where(data[:, 0] == flame_idx)]
        pts, intensity, category = flame_data[:, 1 : 4], flame_data[:, 4], flame_data[:, 5]
        pts, intensity, category = flame_len_fix(pts, intensity, category, num_points)
        pts[:, 0], pts[:, 1], pts[:, 2] = pts[:, 0] / 263, pts[:, 1] / 219, pts[:, 2] / 90
        intensity = intensity * 2 - 1
        X[i, ...] = np.hstack((pts, intensity))
        Y[i, ...] = category
    return X, Y

def genBatchDataTest(data, flame_idxs, num_points):
    X = np.zeros((len(flame_idxs), num_points, 4))
    for i, flame_idx in enumerate(flame_idxs):
        flame_data = data[np.where(data[:, 0] == flame_idx)]
        pts, intensity = flame_data[:, 1 : 4], flame_data[:, 4]
        pts, intensity, fix_head, fix_tail = flame_len_fix_test(pts, intensity, num_points)
        pts[:, 0], pts[:, 1], pts[:, 2] = pts[:, 0] / 263, pts[:, 1] / 219, pts[:, 2] / 90
        intensity = intensity * 2 - 1
        X[i, ...] = np.hstack((pts, intensity))
    return X, fix_head, fix_tail

if __name__ == '__main__':

    pts = np.zeros((1056, 3))
    intensity = np.zeros((1056,))
    pts, intensity, fix_head, fix_tail = flame_len_fix_test(pts, intensity, flame_size=58368)
    print(pts.shape)





