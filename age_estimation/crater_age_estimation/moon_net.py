import torch
import torchvision.models as models
import torch.nn as nn

class Attr_Net(nn.Module):
    def __init__(self, attr_filters=(40, 128, 512)):
        super(Attr_Net, self).__init__()
        self.layer = self._make_layer(attr_filters)
        
    def _make_layer(self, attr_filters):
        layers = []
        for ii in range(len(attr_filters)-1):
            layers.append(nn.Linear(attr_filters[ii], attr_filters[ii+1]))
            layers.append(nn.Dropout(0.3))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)
        

class ResNet_Two(nn.Module):
    def __init__(self, model, pool_size=1, num_classes=5, attr_filters=(40, 128, 512)):
        super(ResNet_Two, self).__init__()
        # 去掉model的后两层
        self.model_layers = nn.Sequential(*list(model.children())[:-2])
        self.pool_layer = nn.AdaptiveAvgPool2d(pool_size)
        self.linear_layer = nn.Linear(2048+attr_filters[-1], 2048)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # 构建 attr net
        self.attr_layer = Attr_Net(attr_filters=attr_filters)
        
    def forward(self, x1, x2):
        x1 = self.model_layers(x1)
        x1 = self.pool_layer(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.attr_layer(x2)
        x = torch.cat((x1, x2), 1)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        

class DenseNet_Two(nn.Module):
    def __init__(self, model, pool_size=1, num_classes=5, attr_filters=(40, 128, 512)):
        super(DenseNet_Two, self).__init__()
        # 去掉model的后两层
        self.model_layers = model.features
        self.pool_layer = nn.AdaptiveAvgPool2d(pool_size)
        self.linear_layer = nn.Linear(model.classifier.in_features+attr_filters[-1], 2048)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # 构建 attr net
        self.attr_layer = Attr_Net(attr_filters=attr_filters)
        
    def forward(self, x1, x2):
        x1 = self.model_layers(x1)
        x1 = self.pool_layer(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.attr_layer(x2)
        x = torch.cat((x1, x2), 1)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        

class ResNeXt_Two(nn.Module):
    def __init__(self, model, pool_size=1, num_classes=5, attr_filters=(40, 128, 512)):
        super(ResNeXt_Two, self).__init__()
        # 去掉model的后两层
        self.model_layers = model._features
        self.pool_layer = nn.AdaptiveAvgPool2d(pool_size)
        self.linear_layer = nn.Linear(model._classifier.in_features+attr_filters[-1], 2048)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # 构建 attr net
        self.attr_layer = Attr_Net(attr_filters=attr_filters)
        
    def forward(self, x1, x2):
        x1 = self.model_layers(x1)
        x1 = self.pool_layer(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.attr_layer(x2)
        x = torch.cat((x1, x2), 1)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        

class Net_Two(nn.Module):
    def __init__(self, model, pool_size=1, in_features=2048, num_classes=5, attr_filters=(40, 128, 512)):
        super(Net_Two, self).__init__()
        # 去掉model的后两层
        self.model_layers = nn.Sequential(*model)
        self.pool_layer = nn.AdaptiveAvgPool2d(pool_size)
        self.linear_layer = nn.Linear(in_features+attr_filters[-1], 2048)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # 构建 attr net
        self.attr_layer = Attr_Net(attr_filters=attr_filters)
        
    def forward(self, x1, x2):
        x1 = self.model_layers(x1)
        x1 = self.pool_layer(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.attr_layer(x2)
        x = torch.cat((x1, x2), 1)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        

class InceptionNet_Two(nn.Module):
    def __init__(self, model, pool_size=1, in_features=2048, num_classes=5, attr_filters=(40, 128, 512)):
        super(InceptionNet_Two, self).__init__()
        # 去掉model的后两层
        self.model_layers1 = nn.Sequential(*model[:13])
        self.model_layers2 = nn.Sequential(*model[14:])
        self.pool_layer = nn.AdaptiveAvgPool2d(pool_size)
        self.linear_layer = nn.Linear(in_features+attr_filters[-1], 2048)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        # 构建 attr net
        self.attr_layer = Attr_Net(attr_filters=attr_filters)
        
    def forward(self, x1, x2):
        x1 = self.model_layers1(x1)
        x1 = self.model_layers2(x1)
        x1 = self.pool_layer(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.attr_layer(x2)
        x = torch.cat((x1, x2), 1)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
        

def resnet_50_two(num_classes=5, attr_filters=(40, 128, 512)):
    model = models.resnet50(pretrained=True)
    pool_size=1
    
    model = ResNet_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def resnet_101_two(num_classes=5, attr_filters=(40, 128, 512)):
    model = models.resnet101(pretrained=True)
    pool_size=1
    
    model = ResNet_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def resnet_152_two(num_classes=5, attr_filters=(40, 128, 512)):
    model = models.resnet152(pretrained=True)
    pool_size=1
    
    model = ResNet_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def densenet_201_two(num_classes=5, attr_filters=(40, 128, 512)):
    model = models.densenet201(pretrained=True)
    pool_size=1
    
    model = DenseNet_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def polynet_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['polynet'](num_classes=1000,
                                                 pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = Net_Two(model, pool_size=pool_size, in_features=in_features, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def senet_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['senet154'](num_classes=1000,
                                                  pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = Net_Two(model, pool_size=pool_size, in_features=in_features, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def se_resnet152_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['se_resnet152'](num_classes=1000,
                                                  pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = ResNeXt_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def se_resnet101_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000,
                                                  pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = ResNeXt_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def se_resnet50_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000,
                                                  pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = ResNeXt_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def se_resnext101_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000,
                                                  pretrained='imagenet')
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-3]
    
    model = ResNeXt_Two(model, pool_size=pool_size, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def inceptionv3_two(num_classes=5, attr_filters=(40, 128, 512)):
    model = models.inception_v3(pretrained=True)
    pool_size = 1
    in_features = list(model.children())[-1].in_features
    model = list(model.children())[:-1]
    
    model = InceptionNet_Two(model, pool_size=pool_size, in_features=in_features, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    
def dpn68b_two(num_classes=5, attr_filters=(40, 128, 512)):
    import pretrainedmodels
    model = pretrainedmodels.__dict__['dpn68b'](num_classes=1000,
                                                pretrained='imagenet+5k')
    pool_size = 1
    in_features = list(model.children())[-1].in_channels
    model = list(model.children())[:-1]
    
    model = Net_Two(model, pool_size=pool_size, in_features=in_features, num_classes=num_classes, attr_filters=attr_filters)
    
    return model
    

