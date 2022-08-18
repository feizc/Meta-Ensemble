# predicted parameters for different networks, e.g., fc layers or total network except 

vgg11_predict_layers = [
    'features.0.weight',
    'features.4.weight',
    'features.8.weight',
    'features.11.weight',
]



resnet50_predict_layers = [
    'conv1.weight',
    'layer1.0.conv1.weight', 
    'layer1.0.conv2.weight',
    'layer1.0.conv3.weight',
]


resnet50_full_predict_layers = [
    'conv1.weight',
    'layer1.0.conv1.weight', 
    'layer1.0.conv2.weight',
    'layer1.0.conv3.weight',
    'layer1.0.downsample.0.weight', 
    'layer1.1.conv1.weight', 
    'layer1.1.conv2.weight',
    'layer1.1.conv3.weight',
    'layer1.2.conv1.weight', 
    'layer1.2.conv2.weight',
    'layer1.2.conv3.weight',
    'layer2.0.conv1.weight', 
    'layer2.0.conv2.weight',
    'layer2.0.conv3.weight',
    'layer2.0.downsample.0.weight', 
    'layer2.1.conv1.weight', 
    'layer2.1.conv2.weight',
    'layer2.1.conv3.weight',
    'layer2.2.conv1.weight', 
    'layer2.2.conv2.weight',
    'layer2.2.conv3.weight',
    'layer2.3.conv1.weight',
    'layer2.3.conv2.weight',
    'layer2.3.conv3.weight', 
    'layer3.0.conv1.weight', 
    'layer3.0.conv2.weight',
    'layer3.0.conv3.weight',
    'layer3.0.downsample.0.weight', 
    'layer3.1.conv1.weight', 
    'layer3.1.conv2.weight',
    'layer3.1.conv3.weight',
    'layer3.2.conv1.weight', 
    'layer3.2.conv2.weight',
    'layer3.2.conv3.weight',
    'layer3.3.conv1.weight',
    'layer3.3.conv2.weight',
    'layer3.3.conv3.weight', 
    'layer3.4.conv1.weight',
    'layer3.4.conv2.weight',
    'layer3.4.conv3.weight', 
    'layer3.5.conv1.weight',
    'layer3.5.conv2.weight',
    'layer3.5.conv3.weight', 
    'layer4.0.conv1.weight', 
    'layer4.0.conv2.weight',
    'layer4.0.conv3.weight',
    'layer4.0.downsample.0.weight', 
    'layer4.1.conv1.weight', 
    'layer4.1.conv2.weight',
    'layer4.1.conv3.weight',
    'layer4.2.conv1.weight', 
    'layer4.2.conv2.weight',
    'layer4.2.conv3.weight',
]

