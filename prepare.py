from g2p.parser import parse_text_to_segments
import opencc
import os, json
from tqdm import tqdm

converter = opencc.OpenCC('t2s')

for file in tqdm([f for f in os.listdir('data/text/txt') if f.endswith('.txt')]):
    phone_lists = []
    with open(f'data/text/txt/{file}', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        phones = []
        segments = parse_text_to_segments(converter.convert(line).replace('姊','姐').replace('著','着'))
        for segment in segments:
            phones += segment.syllables
        phone_lists.append(phones)
    with open(f'data/text/txt/{file[:-3]}json', 'w', encoding='utf-8') as f:
        json.dump(phone_lists, f, ensure_ascii=False)
