from xir import Graph

# 加载 XModel
graph = Graph.deserialize("training_pix2pix_denoiser_denoiser_facades_f32.onnx")

print("XModel 加载成功")

# 检查所有算子支持情况
for op in graph.get_ops():
    op_type = op.get_type()
    op_name = op.get_name()
    
    # 检查是否支持
    if not is_supported_op(op_type):
        print(f"不支持的算子: {op_type} (名称: {op_name})")
        
    # 检查属性
    attrs = op.get_attrs()
    for attr_name, attr_value in attrs.items():
        if not is_supported_attr(attr_name, attr_value):
            print(f"不支持的属性: {op_type}.{attr_name}={attr_value}")

# 检查张量形状
for tensor in graph.get_tensors():
    if not tensor.has_shape():
        print(f"动态形状张量: {tensor.name}")
