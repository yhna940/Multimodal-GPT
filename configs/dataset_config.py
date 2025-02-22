visual_datasets = [
    dict(
        type='llava',
        vis_root='data/coco/train2017',
        ann_paths=[
            'data/llava/detail_23k.json',
            'data/llava/complex_reasoning_77k.json',
        ],
    ),
    dict(
        type='llava_dial',
        vis_root='data/coco/train2017',
        ann_paths=[
            'data/llava/conversation_58k.json',
        ],
    ),
    dict(
        type='aokvqa',
        vis_root='data/coco/images',
        ann_paths=[
            'data/aokvqa/annotations/aokvqa_v1p0_train.json',
        ],
        sample=5000,
    ),
    dict(
        type='minigpt4',
        vis_root='data/cc_sbu_align/image',
        ann_paths=[
            'data/cc_sbu_align/filter_cap.json',
        ],
    ),
    dict(
        type='coco_caption',
        vis_root='data/coco',
        ann_paths=[
            'data/coco/annotations/coco_karpathy_train_converted.json',
            'data/coco/annotations/coco_karpathy_val.json',
        ],
        sample=512,
    ),
    dict(
        type='ocr_vqa',
        vis_root='data/OCR_VQA/image',
        ann_paths=[
            'data/OCR_VQA/downloaded_dataset.json',
        ],
        sample=512,
    ),
]

language_datasets = [
    dict(
        type='dolly',
        ann_path='data/dolly/databricks-dolly-15k.jsonl',
    ),
    dict(
        type='alpaca_gpt4',
        ann_path='data/alpaca_gpt4/alpaca_gpt4_data.json',
    ),
]
