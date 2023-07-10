import numpy as np

p = 536903681

x = np.loadtxt('/home/qipang/mnt/d2/quantize/mrpc/weights_txt/inputs_0_data.txt', delimiter=',')
weights_v = np.loadtxt('/home/qipang/mnt/d2/quantize/mrpc/weights_txt/bert.encoder.layer.0.attention.self.value.weight.txt', delimiter=',')
bias_v = np.loadtxt('/home/qipang/mnt/d2/quantize/mrpc/weights_txt/bert.encoder.layer.0.attention.self.value.bias.txt')

V = []

for i in range(12):
    V.append(np.matmul(x, weights_v[768 * i:768 * (i + 1)]) + bias_v[64 * i:64 * (i + 1)])
V = np.array(V)
V = (V + p) % p
V = V.astype(np.int64)

S1 = np.zeros((128 * 12, 128))

for i in range(128 * 12):
    for j in range(128):
        S1[i, j] = (i * 100 + j) % 1000

S2 = np.zeros((128 * 12, 128))

for i in range(128 * 12):
    for j in range(128):
        S2[i, j] = i * 1000 + j

R = np.zeros((128 * 12, 64))

for i in range(128 * 12):
    for j in range(64):
        R[i, j] = (i * 100 + j) % 1000

result = []

S1 = S1.astype(np.int64)
S2 = S2.astype(np.int64)
R = R.astype(np.int64)

for i in range(12):
    temp1 = (np.matmul(S2[128 * i:128 * (i + 1)].T, V[i]) + p) % p
    temp2 = (np.matmul(S1[128 * i:128 * (i + 1)], R[128 * i:128 * (i + 1)]) + p) % p

    result.append((temp1 + temp2) % p)

for i in range(12):
    result[i] += np.matmul(S1[128 * i:128 * (i + 1)], V[i])
    result[i] = result[i] % p

print(result[0][0])