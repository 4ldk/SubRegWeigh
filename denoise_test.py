from collections import Counter

path1 = "./data/conllpp_test.txt"
path2 = "./data/conll_test_orgi.txt"
path3 = "./outputs/test_weights/conllt_fixed_noise1.txt"

weigh = False
weight_dict = {
    1.0: 1.0,
    0.7: 2 / 3,
    0.48999999999999994: 1 / 3,
    0.3429999999999999: 0,
    0.11764899999999996: 0,
    1.577753820348455e-05: 2 / 33,
    0.24009999999999995: 1 / 3,
    0.004747561509942996: 0,
    0.16806999999999994: 1 / 6,
    0.04035360699999998: 0,
    0.08235429999999996: 2 / 9,
    0.01384128720099999: 0,
    0.028247524899999984: 1 / 6,
    3.219905755813174e-05: 1 / 30,
    3.788186922656639e-06: 1 / 36,
    0.009688901040699992: 2 / 15,
    0.019773267429999988: 1 / 12,
    0.05764800999999997: 1 / 9,
    2.2539340290692216e-05: 0,
    7.73099371970743e-06: 0,
    0.001628413597910447: 0,
    0.006782230728489994: 1 / 15,
}

with open(path1) as f:
    data = f.readlines()

with open(path2) as f2:
    data2 = f2.readlines()

with open(path3) as f3:
    data3 = f3.readlines()

weights = []
miss_weights = []
correct_weights = []
correct_labels = []
wrong_labels = []
weight = 0
for d1, d2, d3 in zip(data, data2, data3):
    if d1.startswith("-DOCSTART-"):
        continue
    elif len(d1) < 3:
        if len(correct_labels) == 0:
            continue

        if correct_labels == wrong_labels:
            correct_weights.append(weight)
        else:
            miss_weights.append(weight)
        correct_labels = []
        wrong_labels = []
        continue

    weight = float(d3.strip().split()[-1])
    if weigh:
        if weight in weight_dict.keys():
            weight = weight_dict[weight]
    correct_labels.append(d1.strip().split()[-1])
    wrong_labels.append(d2.strip().split()[-1])


print(sum(correct_weights) / len(correct_weights), len(correct_weights))
print(sum(miss_weights) / len(miss_weights), len(miss_weights))
# 出現回数のカウント
counts = Counter(miss_weights)
result = {i: counts[i] for i in sorted(list(set(miss_weights)))}

print(result)
