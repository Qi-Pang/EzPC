import numpy as np

common_dim = 768
output_dim = 768
test_attention = False
test_layernorm = True
input_prune = True

if (common_dim == 3072 and output_dim == 768):
    x = np.loadtxt('/home/qipang/mnt/d1/linear/EzPC/SCI/build/bin/txt/random_X_inter1.txt', delimiter=',').astype(np.int64)
    w = np.loadtxt('/home/qipang/mnt/d2/original/mrpc/weights_txt/bert.encoder.layer.0.output.dense.weight.txt', delimiter=',').astype(np.int64)
    b = np.loadtxt('/home/qipang/mnt/d2/original/mrpc/weights_txt/bert.encoder.layer.0.output.dense.bias.txt', delimiter=',').astype(np.int64)
elif (common_dim == 768 and output_dim == 3072):
    x = np.loadtxt('/home/qipang/mnt/d1/clive/EzPC/SCI/build/bin/weights_txt/ln_1.txt', delimiter=',').astype(np.int64)
    w = np.loadtxt('/home/qipang/mnt/d2/sparse/mrpc/weights_txt/bert.encoder.layer.0.intermediate.dense.weight.txt', delimiter=',').astype(np.int64)
    b = np.loadtxt('/home/qipang/mnt/d2/sparse/mrpc/weights_txt/bert.encoder.layer.0.intermediate.dense.bias.txt', delimiter=',').astype(np.int64)
elif (common_dim == 768 and output_dim == 768):
    x = np.loadtxt('/home/qipang/mnt/d1/clive/EzPC/SCI/build/bin/weights_txt/softmax_v.txt', delimiter=',').astype(np.int64)
    w = np.loadtxt('/home/qipang/mnt/d2/sparse/mrpc/weights_txt/bert.encoder.layer.0.attention.output.dense.weight.txt', delimiter=',').astype(np.int64)
    b = np.loadtxt('/home/qipang/mnt/d2/sparse/mrpc/weights_txt/bert.encoder.layer.0.attention.output.dense.bias.txt', delimiter=',').astype(np.int64)

if test_attention:
    x = np.loadtxt('/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/inputs_0.txt', delimiter=',').astype(np.int64)
    w = np.loadtxt('/home/qipang/mnt/d2/sparse/mrpc/weights_txt/bert.encoder.layer.0.attention.self.key.weight.txt', delimiter=',').astype(np.int64)
    b = np.loadtxt('/home/qipang/mnt/d2/secure-bert/robert/sparse/sst-2/weights_txt/bert.encoder.layer.0.attention.self.key.bias.txt', delimiter=',').astype(np.int64)

    p = 2**37

    w = w.reshape((12, 768, 64))
    b = b.reshape((12, 64))

    for i in range(1):
        np_res = np.matmul(x, w[i]) + b[i]
        np_res = np_res.astype(np.int64)
        np_res = (np_res + p) % p
        print(np_res.shape)
        print(np_res)
elif test_layernorm:
    layernorm_prime = 4295049217
    x1 = np.zeros((128, 768))
    x2 = np.zeros((128, 768))
    gamma = np.zeros(768)
    for i in range(128):
        for j in range(768):
            x1[i, j] = (i * 1000 + j) % layernorm_prime
            x2[i, j] = (j * 10 + i) % layernorm_prime
    for i in range(768):
        gamma[i] = i % 27
    
    np_res = ((x1 + x2) * gamma + layernorm_prime) % layernorm_prime
    np_res = np_res.astype(np.int64)

    print(np_res)
    fhe_res = np.loadtxt('/home/qipang/mnt/d1/linear/EzPC/SCI/build/bin/txt/layernorm_result.txt', delimiter=',').astype(np.int64)

    print((np_res == fhe_res).sum(), x1.shape[0] * x1.shape[1])


else:
    p = 557057
    # p = 137438953472
    # p = 137439010817
    # p = 67108864

    if input_prune:
        x = x[:64]

    print(x.shape, w.shape, b.shape)

    # x = (x + p) % p
    # w = (w + p) % p
    # b = (b + p) % p

    np_res = (np.matmul(x, w) + b).astype(np.int64)
    # print(np_res.max(), np_res.min())
    # print(np_res.dtype, ((np_res + p) % p).dtype)
    np_res = (np_res + p).astype(np.int64) % p

    fhe_res = np.loadtxt('/home/qipang/mnt/d1/linear/EzPC/SCI/build/bin/txt/ct-pt-result.txt', delimiter=',').astype(np.int64)

    # fhe_res = np.loadtxt('/home/qipang/mnt/d1/linear/EzPC/SCI/build/bin/txt/iron-ct-pt-result.txt', delimiter=',').astype(np.int64)

    # fhe_res[fhe_res > p // 2] -= p
    # fhe_res = fhe_res.astype(np.int64)

    print(np_res)

    print(fhe_res)

    print((np_res == fhe_res).sum(), x.shape[0] * w.shape[1])