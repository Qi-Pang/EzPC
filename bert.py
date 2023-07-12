import numpy as np
import sys
from transformers import glue_compute_metrics as compute_metrics
import warnings, argparse
warnings.filterwarnings("ignore")

np.set_printoptions(threshold=sys.maxsize)

p_large = 536903681
p_small = 557057

max_linear1 = np.zeros(12)
max_linear2 = np.zeros(12)
max_linear3 = np.zeros(12)
max_linear4 = np.zeros(12)

global x_scale
global w_scale
global b_scale

def automatic_scale(max_ln1, max_ln2, max_ln3, max_ln4, original=False):
    x_scale = np.ones((4, 12))
    w_scale = np.ones((4, 12))
    b_scale = np.ones((4, 12))

    for layer_num in range(12):
        x_scale[0, layer_num] = 5
        w_scale[0, layer_num] = 6
        b_scale[0, layer_num] = 11

        x_scale[1, layer_num] = 6
        w_scale[1, layer_num] = 6
        b_scale[1, layer_num] = 12

        x_scale[2, layer_num] = 5
        w_scale[2, layer_num] = 6
        b_scale[2, layer_num] = 11

        x_scale[3, layer_num] = 4
        w_scale[3, layer_num] = 5
        b_scale[3, layer_num] = 9
    
    x_scale[3, 9] = 4
    w_scale[3, 9] = 4
    b_scale[3, 9] = 8

    x_scale[3, 10] = 4
    w_scale[3, 10] = 4
    b_scale[3, 10] = 8

    if original:
        return x_scale, w_scale, b_scale

    budget = np.zeros((4, 12))
    for layer_num in range(12):
        budget[0, layer_num] = np.floor(np.log2(np.floor(p_large / 2 / max_ln1[layer_num])))
        budget[1, layer_num] = np.floor(np.log2(np.floor(p_small / 2 / max_ln2[layer_num])))
        budget[2, layer_num] = np.floor(np.log2(np.floor(p_small / 2 / max_ln3[layer_num])))
        budget[3, layer_num] = np.floor(np.log2(np.floor(p_small / 2 / max_ln4[layer_num])))
    
    for layer_num in range(12):
        for linear_num in range(4):
            coin = 0
            while budget[linear_num, layer_num] > 0:
                if coin == 0:
                    x_scale[linear_num, layer_num] += 1
                else:
                    w_scale[linear_num, layer_num] += 1
                b_scale[linear_num, layer_num] += 1
                budget[linear_num, layer_num] -= 1
                coin = (coin + 1) % 2
    return x_scale, w_scale, b_scale

def truncate(x, bits = 12):
    return np.floor(x * 2**bits) / 2**bits

def softmax(matrix):
    matrix = np.floor(matrix)
    max_values = np.max(matrix, axis=1)
    matrix = matrix - max_values[:, np.newaxis]
    # noise = np.random.normal(0, 1e-2, matrix.shape)
    # exp_res = exp_approx(matrix)
    exp_res = exp_approx_int(matrix)
    recp = np.sum(exp_res, axis=1, keepdims=True)
    recp = recp / 2**12
    recp = np.floor(1.0 / recp * 2**12)
    return np.floor(exp_res * recp / 2**12)

def exp_approx(x):
    # x <= 0
    x = truncate(x, bits=12)
    l = np.floor(x * truncate(1.0 / np.log(2), bits=12) * (-1))
    p = truncate(x + l * truncate(np.log(2), bits=12), bits=12)
    fp = truncate(0.3585) * truncate((p + truncate(1.353)) ** 2) + truncate(0.344)
    fp = truncate(fp)
    res = truncate(fp / 2**l)
    return res

def exp_approx_int(x):
    x = np.floor(x)
    temp_l = np.floor(x * truncate(1.0 / np.log(2), bits=12) * (-1))
    l = np.floor(temp_l / 2**12)
    p = x + np.floor(l * truncate(np.log(2), bits=12) * 2**12)
    c1 = np.floor(0.3585 * 2**12)
    c2 = np.floor(1.353 * 2**12)
    c3 = np.floor(0.344 * 2**12)
    fp = np.floor((c1 * np.floor(((p + c2) ** 2) / 2**12)) / 2**12) + c3
    fp = np.floor(fp / 2**l)
    return fp

def sort_online_prune(softmax_out):
    assert softmax_out.shape[0] == 12

    scores = None
    for i in range(12):
        if scores is None:
            scores = softmax_out[i].sum(axis=0)
        else:
            scores += softmax_out[i].sum(axis=0)
    preserved_words = np.argsort(scores)[::-1]
    preserved_words = preserved_words[:96]
    preserved_words = np.sort(preserved_words)
    return preserved_words

def layer_norm(matrix, W, B):
    matrix = np.floor(matrix * 2**12) / 2**12
    mean = np.mean(matrix, axis=1, keepdims=True)
    variance = np.var(matrix, axis=1, keepdims=True)

    # Calculate the layer normalization
    epsilon = 1e-8  # small value to avoid division by zero
    normalized_matrix = (matrix - mean) / np.sqrt(variance + epsilon)

    normalized_matrix = normalized_matrix * W / 2**12 + B / 2**12
    # normalized_matrix = (matrix - mean)
    # noise = np.random.normal(0, 1e-2, normalized_matrix.shape)
    normalized_matrix = np.floor(normalized_matrix * 2**12) / 2**12
    return normalized_matrix

def layer_norm_fix(matrix, W, B):

    sum = np.sum(matrix, axis=1, keepdims=True)
    array_size = matrix.shape[1]

    dn = np.floor(1.0 / array_size * 2**24)

    avg = np.floor(sum*dn / (2**24))
    # mean_ref = np.floor(np.mean(matrix, axis=1, keepdims=True))

    x_avg = matrix - avg

    x_avg_square = x_avg**2
    x_avg_square = np.floor(x_avg_square / (2**12))

    x_avg_square_sum = np.sum(x_avg_square, axis=1, keepdims=True)
    
    x_avg_square_sum_avg = np.floor(x_avg_square_sum*dn / (2**24))

    sigma = np.floor((1 / np.sqrt(x_avg_square_sum_avg / (2**12)))*2**12)

    x_avg_sigma = np.floor(x_avg*sigma / (2**12))

    ln_w = np.floor(x_avg_sigma*W / (2**12))

    ln_w_b = ln_w + B

    return ln_w_b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def approx_gelu(x):
    # x = truncate(x)

    c1 = 0.14439048359960427
    c2 = 0.7077117131613893
    c3 = 4.5702822654246535
    c4 = 8.15444702051307
    c5 = 16.382265425072532

    c1 = np.floor(c1 * 2**11)
    c2 = np.floor(c2 * 2**11)
    c3 = np.floor(c3 * 2**11)
    c4 = np.floor(c4 * 2**11)
    c5 = np.floor(c5 * 2**11)

    abs_x = np.abs(x)
    # y = truncate((truncate(c1 * abs_x, bits=12) - c2) * abs_x, bits=12) + c3
    # res = truncate((y + truncate(c1 * abs_x, bits=12) - c4) * y, bits=12) + c5 + 0.5 * x
    # res = truncate(res, bits=12)
    # res[x > 2.7] = x[x > 2.7]
    # res[x < -2.7] = 0

    temp_y = np.floor(c1 * abs_x / 2**11) - c2
    y = np.floor(temp_y * abs_x / 2**11) + c3
    temp_res = y + np.floor(c1 * abs_x / 2**11) - c4
    temp_res = temp_res * y
    res = np.floor(temp_res / 2**11) + c5 + x / 2

    res = np.floor(res)

    res[x > np.floor(2.7 * 2**11)] = x[x > np.floor(2.7 * 2**11)]
    res[x < np.floor(-2.7 * 2**11)] = 0
    return res

def approx_tanh(x):

    x_1 = np.abs(x)

    a = -0.013232131886235352
    b = 0.09948747962825866
    c = -0.20093640347818847
    d = -0.17616532856475706
    e = 1.0542492677156243
    f = -0.0024920889620412097

    a = np.floor(a * 2**12)
    b = np.floor(b * 2**12)
    c = np.floor(c * 2**12)
    d = np.floor(d * 2**12)
    e = np.floor(e * 2**12)
    f = np.floor(f * 2**12)

    x_2 = np.floor(x_1*x_1 / (2**12))
    x_3 = np.floor(x_2*x_1 / (2**12))
    x_4 = np.floor(x_2*x_2 / (2**12))
    x_5 = np.floor(x_4*x_1 / (2**12))

    x_5_a = np.floor(x_5*a/ (2**12))
    x_4_b = np.floor(x_4*b/ (2**12))
    x_3_c = np.floor(x_3*c/ (2**12))
    x_2_d = np.floor(x_2*d/ (2**12))
    x_1_e = np.floor(x_1*e/ (2**12))

    poly = x_5_a + x_4_b + x_3_c + x_2_d + x_1_e + f

    res = poly / (2**12)

    res[x < 0] = -res[x < 0]

    res[x > np.floor(2.855 * 2**12)] = 1
    res[x < np.floor(-2.855 * 2**12)] = -1

    return res




def linear1(H1, Wq, Wk, Wv, Bq, Bk, Bv, input_mask, layer_count):
    assert Wq.shape == (12, 768, 64) and Bq.shape == (12, 64)

    # H1_scale = 12 - x_scale[0, layer_count]
    # H1 = np.floor(H1 / 2**H1_scale)

    if args.online_prune and layer_count >= 1:
        softmax_input = np.zeros((12, 96, 96))
        Q = np.zeros((12, 96, 64))
        K = np.zeros((12, 96, 64))
        V = np.zeros((12, 96, 64))
        softmax_output = np.zeros((12, 96, 96))
    else:
        softmax_input = np.zeros((12, 128, 128))
        Q = np.zeros((12, 128, 64))
        K = np.zeros((12, 128, 64))
        V = np.zeros((12, 128, 64))
        softmax_output = np.zeros((12, 128, 128))
    for i in range(12):
        Q[i] = np.matmul(H1, Wq[i]) + Bq[i]
        K[i] = np.matmul(H1, Wk[i]) + Bk[i]
        # softmax_input[i] = np.matmul(Q[i], K[i].T)
        softmax_input[i] = np.matmul(Q[i], K[i].T)
        V[i] = np.matmul(H1, Wv[i]) + Bv[i]

    # HACK: rescale

    # max_linear1[layer_count] = np.max([max_linear1[layer_count], np.abs(softmax_input).max()])

    # if np.abs(softmax_input).max() > p_large / 2:
    #     print('linear 1 layer ', layer_count, np.abs(softmax_input).max())

    
    softmax_input_scale = 22 - 12

    softmax_input = np.floor(softmax_input / 2**softmax_input_scale)

    for i in range(12):
        softmax_output[i] = softmax(softmax_input[i] + input_mask * 100)

    # HACK: scale
    # softmax_output = np.floor(np.abs(softmax_output) * (2**12)) * np.sign(softmax_output)


    attention_res = []
    for i in range(12):
        attention_res.append(np.matmul(softmax_output[i], V[i]))
    attention_res = np.hstack(attention_res)
    # assert attention_res.shape == (128, 768)

    # attention_res scale 23
    return attention_res

def linear2(H2, W_attO, B_attO):
    assert W_attO.shape == (768, 768) and B_attO.shape == (768,)

    res = np.matmul(H2, W_attO) + B_attO
    return res

def linear3(H4, W_inter, B_inter):
    assert W_inter.shape == (768, 3072) and B_inter.shape == (3072,)
    res = np.matmul(H4, W_inter) + B_inter
    return res

def linear4(H6, W_out, B_out):
    assert W_out.shape == (3072, 768) and B_out.shape == (768,)
    res = np.matmul(H6, W_out) + B_out
    return res

def attention_layer(H1, Wq, Wk, Wv, Bq, Bk, Bv, W_attO, B_attO, W_inter, B_inter, W_out, B_out, W_layernorm_att, B_layernorm_att, W_layernorm_out, B_layernorm_out, input_mask, layer_count):
    H2 = linear1(H1, Wq, Wk, Wv, Bq, Bk, Bv, input_mask, layer_count)

    # scale back to 6
    H2 = np.floor(H2 / 2**17)

    H3 = linear2(H2, W_attO, B_attO)

    H3 += H1*2**7

    # H4 scale 12
    H4 = layer_norm_fix(H3, W_layernorm_att, B_layernorm_att)

    # rescale to 
    H4_rescale = np.floor(H4 / 2**7)

    res_linear3 = linear3(H4_rescale, W_inter, B_inter)

    # H6 scale 11 
    H6 = approx_gelu(res_linear3)

    # rescale to 
    H6_rescale = np.floor(H6 / 2**7)
    #H_8 scale 8 or 9
    H8 = linear4(H6_rescale, W_out, B_out)
    H8_scale = 9
    if layer_count ==  9 or layer_count ==10:
        H8_scale = 8

    H8_scale_12 = H8 * 2**(12 - H8_scale)
    H8_scale_12 += H4

    H9 = layer_norm_fix(H8_scale_12, W_layernorm_out, B_layernorm_out)
    # H9 scale 12
    return H9

def pool_classify_layer(X, W_p, W_c, B_p, B_c):
    pool_linear = np.matmul(X, W_p) + B_p*2**12

    pool_linear = np.floor(pool_linear / 2**12)
    pool_linear = pool_linear / 2**12
    
    pool_res = np.tanh(pool_linear) 
    pool_res = np.floor(pool_res * 2**12)
    
    result = np.matmul(pool_res, W_c) + B_c*2**12
    return result / 2**24

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--sparse', action="store_true")
    parser.add_argument('--online_prune', action="store_true")
    parser.add_argument('--sample_num', type=int)


    args = parser.parse_args()

    data_dir = '/home/ubuntu/mrpc/quantize/weights_txt/'
    # if args.sparse:
    #     data_dir += 'sparse/'
    # else:
    #     # data_dir += 'original/'
    #     data_dir += 'robust/'
    #     # data_dir += 'trash/'
    # data_dir += args.task_name
    # # data_dir += '/biased_weights_txt/'
    # data_dir += '/weights_txt/'
    # data_dir += '/floor_weights/'

    acc = 0
    total = 0

    pred = []
    label = []
    results = {}

    Wq, Wk, Wv, Bq, Bk, Bv, W_attO, B_attO, W_inter, B_inter, W_out, B_out, W_layernorm_att, B_layernorm_att, W_layernorm_out, B_layernorm_out = ([] for i in range(16))

    for layer_num in range(12):
        # print('layer num: ', layer_num)
        Wq.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.query.weight.txt', delimiter=',').astype(np.int64).reshape((12, 768, 64)))
        Wk.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.key.weight.txt', delimiter=',').astype(np.int64).reshape((12, 768, 64)))
        Wv.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.value.weight.txt', delimiter=',').astype(np.int64).reshape((12, 768, 64)))

        Bq.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.query.bias.txt', delimiter=',').astype(np.int64).reshape((12, 64)))
        Bk.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.key.bias.txt', delimiter=',').astype(np.int64).reshape((12, 64)))
        Bv.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.self.value.bias.txt', delimiter=',').astype(np.int64).reshape((12, 64)))

        W_attO.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.output.dense.weight.txt', delimiter=',').astype(np.int64))
        B_attO.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.output.dense.bias.txt', delimiter=',').astype(np.int64))

        W_inter.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.intermediate.dense.weight.txt', delimiter=',').astype(np.int64))
        B_inter.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.intermediate.dense.bias.txt', delimiter=',').astype(np.int64))

        W_out.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.output.dense.weight.txt', delimiter=',').astype(np.int64))
        B_out.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.output.dense.bias.txt', delimiter=',').astype(np.int64))

        W_layernorm_att.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.output.LayerNorm.weight.txt', delimiter=','))
        B_layernorm_att.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.attention.output.LayerNorm.bias.txt', delimiter=','))

        W_layernorm_out.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.output.LayerNorm.weight.txt', delimiter=','))
        B_layernorm_out.append(np.loadtxt(data_dir + 'bert.encoder.layer.' + str(layer_num) + '.output.LayerNorm.bias.txt', delimiter=','))

    
    W_pool = np.loadtxt(data_dir + 'bert.pooler.dense.weight.txt', delimiter=',').astype(np.int64)
    B_pool = np.loadtxt(data_dir + 'bert.pooler.dense.bias.txt', delimiter=',').astype(np.int64)

    W_classify = np.loadtxt(data_dir + 'classifier.weight.txt', delimiter=',').astype(np.int64)
    B_classify = np.loadtxt(data_dir + 'classifier.bias.txt', delimiter=',').astype(np.int64)

    # pbar = tqdm.tqdm(range(args.sample_num))
    labels = np.loadtxt(data_dir + 'labels.txt', delimiter=',').astype(np.int64)

    # x_scale, w_scale, b_scale = automatic_scale(max_linear1, max_linear2, max_linear3, max_linear4, original=True)
    # x_scale = np.loadtxt(data_dir + 'x_scale.txt', delimiter=',')
    # w_scale = np.loadtxt(data_dir + 'w_scale.txt', delimiter=',')
    # b_scale = np.loadtxt(data_dir + 'b_scale.txt', delimiter=',')

    # print(x_scale, w_scale, b_scale)

    # x_scale[0, :] = 5
    # b_scale[0, :] = 11

    for data_sample in range(args.sample_num):
        # pbar.update(1)
        total += 1

        H1 = np.loadtxt(data_dir + 'inputs_' + str(data_sample) + '_data.txt', delimiter=',').astype(np.int64)
        input_mask = np.loadtxt(data_dir + 'inputs_' + str(data_sample) + '_mask.txt', delimiter=',').astype(np.int64)
        ground_truth = labels[data_sample]

        # rescale to 12
        H1 = H1 * 2**7
        
        for layer_num in range(12):
            # rescale back to 5
            H1 = np.floor(H1 / 2**7)
            H1 = attention_layer(H1, Wq[layer_num], Wk[layer_num], Wv[layer_num], Bq[layer_num], Bk[layer_num], Bv[layer_num], W_attO[layer_num], B_attO[layer_num], W_inter[layer_num], B_inter[layer_num], W_out[layer_num], B_out[layer_num], W_layernorm_att[layer_num], B_layernorm_att[layer_num], W_layernorm_out[layer_num], B_layernorm_out[layer_num], input_mask, layer_num)

        res = pool_classify_layer(H1[0], W_pool, W_classify, B_pool, B_classify)

        print(res)
        if 'sts-b' in data_dir:
            pred.append(res)
        else:
            pred.append(np.argmax(res))
        label.append(ground_truth)

        if (data_sample + 1) % 10 == 0:
            huggingface_eval = compute_metrics(args.task_name, np.array(pred), np.array(label))
            results.update(huggingface_eval)
            # pbar.set_postfix(results)

    huggingface_eval = compute_metrics(args.task_name, np.array(pred), np.array(label))
    results.update(huggingface_eval)

    if np.argmax(res) == ground_truth:
        acc += 1

    print(results)

    # print(max_linear1)
    # print(max_linear2)
    # print(max_linear3)
    # print(max_linear4)

    # x_scale, w_scale, b_scale = automatic_scale(max_linear1, max_linear2, max_linear3, max_linear4, original=False)

    # print(x_scale)
    # print(w_scale)
    # print(b_scale)

    # np.savetxt(data_dir + 'x_scale.txt', x_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'w_scale.txt', w_scale, delimiter=',', fmt='%d')
    # np.savetxt(data_dir + 'b_scale.txt', b_scale, delimiter=',', fmt='%d')

    # print(x_scale)
    # print(w_scale)
    # print(b_scale)