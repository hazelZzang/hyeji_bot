from hyejibot.dataset.domain_loader import get_loaders
from hyejibot.utils.config import config
from hyejibot.models.selfAttn_classifier import SelfAttention
from hyejibot.models.phoneme_embedding import PhonemeEmbedding
from hyejibot.trainer.domain_trainer import DomainTrainer
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path,"data","train","domain.pkl")
loaders, src_lang, trg_lang = get_loaders(file_path, [config.batch,1,1], [0.8,0.2,0.0])

embedding = PhonemeEmbedding(src_lang.n_words, config.hidden)
model = SelfAttention(config.hidden, config.batch, config.hidden,
                      num_layers=config.layers, dropout=config.dropout, n_classes=trg_lang.n_words)
trainer = DomainTrainer(embedding, model, config,
                        src_lang = src_lang,trg_lang = trg_lang,
                        data_loader=loaders[0], valid_data_loader=loaders[1])
trainer.train()
