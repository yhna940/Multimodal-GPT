import json
import os
import random

from PIL import Image

from .vqa_dataset import VQADataset, VQAPrompter

QUESTIONS = [
    'please describe the image',
    'can you describe the image',
    'Could you provide a description of the image?',
    'What do you see in this image?',
    'Share your thoughts on the content of the image.',
    "Please narrate what's happening in the picture.",
    'Can you give a brief explanation of the image?',
    'Describe the main elements and details present in the image.',
    'In your own words, what is depicted in the image?',
    'Can you outline the key aspects of the image?',
    'What are the most striking features in this image?',
    "Please provide a summary of the image's content.",
    'Describe the overall theme or concept captured in the image.',
    "How would you explain the image's composition and focus?",
    'What is the focal point or main subject of the image?',
    'How do the different components of the image interact with each other?',
    'What would be a fitting caption for this image?',
    'Can you create a concise description that captures the essence of the image?',
    'How would you briefly summarize the content of this image in a phrase or sentence?',
    'Please provide a catchy and relevant caption for this picture.',
    'If you were to give this image a title, what would it be?',
    'Describe the image in one creative sentence.',
    "Please suggest a memorable phrase that encapsulates the image's content.",
    'What engaging phrase would best represent this image?',
    'Can you create an expressive caption that highlights the main theme of the image?',
    "How would you sum up the image's story for a caption?",
    "Provide an eye-catching caption that conveys the image's core message.",
    'If you were to give this image a headline, what would it say?',
    'Can you craft a captivating caption that communicates the essence of the image?',
    "How would you describe the image's content in a powerful caption?",
    'Please provide an inventive title to summarize the scene depicted in the image.',
    "Compose a concise and striking phrase that reflects the image's key elements.",
    'If you were to create a caption for this image, what would it be?',
    'Offer a compelling caption that highlights the central focus of the image.',
    "Can you produce a unique caption that encapsulates the image's overall mood?",
    'Please generate an attention-grabbing caption that would best illustrate the events captured in this image',
    "How would you express the image's main idea in an impactful sentence?",
    'Please create a vivid and concise title that conveys the essence of the picture.',
    "Compose an imaginative caption that reflects the image's most striking features.",
    'What memorable statement would best represent the scene illustrated in this image?',
    'Draft an evocative caption that brings the image to life for the reader.',
    'Can you suggest an insightful caption that highlights the underlying message of the image?',
    'What engaging phrase would effectively convey the action or subject matter depicted in this picture?',
    "How would you encapsulate the image's core theme in a concise and expressive manner?",
    'Please provide a creative and impactful title that captures the spirit of the image.',
    "Craft a captivating caption that showcases the image's most prominent attributes.",
    'What intriguing statement would best sum up the scene presented in this image?',
    'Develop a descriptive caption that paints a vivid picture for the viewer.',
    "Can you give a detailed account of the image's contents?",
    'What are the key elements and features visible in this image?',
    'How would you narrate the events or actions depicted in the picture?',
    'Please share your observations about the various components present in the image.',
    'What is the overall theme or concept captured in this image? Can you describe it?',
]


class CcSbuAlignDataset(VQADataset):

    def __init__(self,
                 tokenizer,
                 vis_processor,
                 vis_root,
                 ann_paths,
                 add_eos=True,
                 ignore_instruction=True):
        self.tokenizer = tokenizer
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path))['annotations'])

        self.vis_processor = vis_processor
        self.prompter = VQAPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction

    def process_text(self, ann):
        # random select a question
        question = random.choice(QUESTIONS)
        answer = ann['caption']
        instruction = self.prompter(question)
        return dict(instruction=instruction, answer=answer)

    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann['image_id'] + '.jpg')
        image = Image.open(image_path).convert('RGB')

        image = self.vis_processor(image)
        return image


def build_ccsbualign_dataset(
    tokenizer,
    vis_processor,
    vis_root='data/cc_sbu_align/image/',
    ann_paths=['data/cc_sbu_align/filter_cap.json'],
    **kwargs,
):
    return CcSbuAlignDataset(
        tokenizer=tokenizer,
        vis_processor=vis_processor,
        vis_root=vis_root,
        ann_paths=ann_paths,
    )
