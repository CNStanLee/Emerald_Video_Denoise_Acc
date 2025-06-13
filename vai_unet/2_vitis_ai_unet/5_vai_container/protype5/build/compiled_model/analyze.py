import xir
g = xir.Graph.deserialize("UnetGenerator_u50.xmodel")
subgraphs = get_child_subgraph_dpu(g)

# 打印子图输入输出关系
for sg in subgraphs:
    inputs = [t.name for t in sg.get_input_tensors()]
    outputs = [t.name for t in sg.get_output_tensors()]
    print(f"{sg.get_name()} | 输入: {inputs} -> 输出: {outputs}")
