import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
import cornet

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, layer_name):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.layer_name = layer_name

    def forward(self, x):
        for name, module in self.pretrained_model.named_children():
            x = module(x)
            if name == self.layer_name:
                return x.view(x.size(0), -1)  # Flatten the output
        return x

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            feat = model(X)
            features.append(feat.cpu().numpy())
    return np.concatenate(features)

# Load pretrained CORnet-S
pretrained_cornet = cornet.cornet_s(pretrained=True)
pretrained_cornet.to(device)

train_dl, val_dl, test_dl = create_dataloader(args, m, g_cpu, X_train,
    X_val, X_test, y_train, y_val, y_test)

# Extract features for each layer
layers = ['V1', 'V2', 'V4', 'IT']
features = {}
for layer in layers:
    feature_extractor = FeatureExtractor(pretrained_cornet, layer)
    feature_extractor.to(device)
    features[layer] = {
        'train': extract_features(feature_extractor, train_dl, device),
        'val': extract_features(feature_extractor, val_dl, device),
        'test': extract_features(feature_extractor, test_dl, device)
    }

# Train regression models for each layer and time point
regression_models = {}
for layer in layers:
    regression_models[layer] = []
    for t in range(y_train.shape[2]):  # For each time point
        model = Ridge(alpha=1.0)
        model.fit(features[layer]['train'], y_train[:, :, t])
        regression_models[layer].append(model)

# Predict EEG signals
predictions = {}
for layer in layers:
    predictions[layer] = np.zeros_like(y_test)
    for t in range(y_test.shape[2]):
        predictions[layer][:, :, t] = regression_models[layer][t].predict(features[layer]['test'])

# Evaluate predictions
for layer in layers:
    mse = np.mean((predictions[layer] - y_test) ** 2)
    print(f"MSE for {layer}: {mse}")