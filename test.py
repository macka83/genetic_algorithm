res = [
    ("af4c0731", 3, "mid", -0.2247520219746681, 0, "mid", "mid3mid0"),
    ("d46be7cf", 0, "mid", 3.244315580650084, 3, "mid", "mid0mid3"),
    ("e2d75301", 2, "mid", -2.5941095681367314, 0, "out", "mid2out0"),
    ("8513bc23", 1, "mid", 1.8794445292232564, 3, "mid", "mid1mid3"),
    ("01079e8e", 1, "in", 0.9549214100412025, 3, "mid", "in1mid3"),
    ("20992ff5", 2, "in", -1.4987944452922326, 1, "out", "in2out1"),
    ("fecccf0d", 2, "mid", 2.4705631008698306, 1, "out", "mid2out1"),
    ("b7bd13ca", 3, "mid", -0.6184648252708683, 1, "out", "mid3out1"),
    ("a66aed82", 2, "mid", 3.422432473676179, 2, "mid", "mid2mid2"),
    ("48ad9402", 2, "in", 0.6253013886769418, 0, "out", "in2out0"),
]

dic = {}
for item in res:
    hexval, input_id, input_type, weight, output_id, output_type, differ_neuron = item
    total = dic.get(differ_neuron, 0) + weight
    n_output = f"{output_type}{output_id}"
    n_input = f"{input_type}{input_id}"
    dic[differ_neuron] = [n_input, n_output, total]
    print(dic)
