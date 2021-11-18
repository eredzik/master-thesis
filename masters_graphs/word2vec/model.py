import torch.nn as nn
import torch.nn.functional as F


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)
        self.context_size = context_size
        self.embedding_dim = embedding_dim

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(
            (-1, self.context_size * self.embedding_dim)
        )
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def get_embeddings(self):
        return self.embeddings.weight


class NeuralNonlinearClassifier(nn.Module):
    def __init__(self, input_size, n_classes=2):
        super(NeuralNonlinearClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.log_softmax(self.linear3(x), dim=1)
