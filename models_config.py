
GENERAL_MODELS = {
    # https://arxiv.org/abs/2103.00020
    'CLIP-ViT-B/32': 'ViT-B/32',
    'CLIP-RN50': 'RN50',
    'CLIP-ViT-L/14': 'ViT-L/14',
    'CLIP-RN50x64': 'RN50x64',
}

VISION_LANGUAGE_MODELS = {
    # https://arxiv.org/abs/2102.03334
    # https://huggingface.co/docs/transformers/model_doc/vilt
    'ViLT': "dandelin/vilt-b32-finetuned-coco",
    # https://arxiv.org/abs/2111.07991
    # https://huggingface.co/docs/transformers/model_doc/vision-text-dual-encoder
    'LiT': {'vision': "google/vit-base-patch16-224", 'text': "bert-base-uncased"},
}

VISION_MODELS = {
    # https://arxiv.org/abs/2010.11929
    # ViT-Large model (ViT-L/32), ImageNet-1k weights fine-tuned from in21k @ 384x384
    'ViT': 'vit_large_patch32_384',

    # https://arxiv.org/abs/2103.14030
    # Swin-L @ 384x384, pretrzained ImageNet-22k, fine tune 1k
    'Swin': 'swin_large_patch4_window12_384',

    # https://arxiv.org/abs/2012.12877
    # DeiT base model @ 384x384,ImageNet-1k weights from https://github.com/facebookresearch/deit.
    'DeiT': 'deit_base_patch16_384',

    # https://arxiv.org/pdf/2201.03545.pdf
    'ConvNeXt': 'convnext_large',
}

TEXT_TRANSFORMERS_MODELS = {
    ## 3 top models for semantic search, from here: https://www.sbert.net/docs/pretrained_models.html

    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    'MPNet': 'all-mpnet-base-v2',

    # https://huggingface.co/sentence-transformers/all-mpnet-base-v2
    'MPNet QA': 'multi-qa-mpnet-base-dot-v1',

    # https://huggingface.co/sentence-transformers/all-distilroberta-v1
    'Distil RoBERTa': 'all-distilroberta-v1',
}

WORD2VEC_MODELS  = {'Word2Vec': 'word2vec'}

MODELS_MAP = {**GENERAL_MODELS, **VISION_LANGUAGE_MODELS, **VISION_MODELS, **TEXT_TRANSFORMERS_MODELS, **WORD2VEC_MODELS}
