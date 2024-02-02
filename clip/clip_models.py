import torch.nn as nn
import clip
import torch

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


class ImageCLIP(nn.Module):
    """
    Wrapper for the CLIP model
    Load zero shot weights obtained from the text encoder
    """
    def __init__(self, model, preprocess, zs_weights_path=None):
        super(ImageCLIP, self).__init__()
        self.model = model
        self.preprocess = preprocess
        self.load_zs_weights(zs_weights_path)

    def forward(self, image):

        """
        Computes the similarity of image features with the class embeddings
        generated from the text encoder.
        """
        image_features = self.model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self.zs_weights["class_embeddings"]).softmax(dim=-1)

        return similarity

    def load_zs_weights(self, path):
        """
        Load the zero shot weights obtained from the text encoder
        """
        self.zs_weights = torch.load(path)


    def freeze(self):
        for name, param in self.named_parameters():
            param.requires_grad = False


def get_clip_image_model(clip_variant_name, zs_weights_path=None):
    model, preprocess = clip.load(clip_variant_name, jit=False)
    model_image = ImageCLIP(model, preprocess, zs_weights_path)

    return model_image, preprocess


def get_clip_text_model(clip_variant_name):
    model, preprocess = clip.load(clip_variant_name, jit=False)
    model_text = TextCLIP(model)

    return model_text, preprocess


def freeze_clip_image_model(model):
    for name, param in model.named_parameters():
        if 'visual' in name:
            param.requires_grad = False
    return model


def freeze_clip_text_model(model):
    for name, param in model.named_parameters():
        if 'visual' not in name:
            param.requires_grad = False
    return model


def freeze_clip_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


if __name__ == '__main__':
    import os
    import clip
    import torch
    from torchvision.datasets import CIFAR100
    from helpers import accuracy

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = get_clip_image_model('ViT-B/32',
                                             '../notebooks/normalised_zeroshot_vit_b-32_cifar100_weights.pt')

    # Download the dataset
    cifar100 = CIFAR100(root="/tmp", download=True, train=False)
    cifar100 = CIFAR100(root="/tmp", download=True, transform=preprocess)
    loader = torch.utils.data.DataLoader(cifar100, batch_size=128)

    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            acc1 = accuracy(logits, labels, topk=(1,))[0].item()
            correct += acc1*len(labels)

    print(f"Accuracy: {correct/len(cifar100)}")
