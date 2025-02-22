import datasets
from datasets import concatenate_datasets
import os
import sys
from pathlib import Path
import re
import pyarrow as pa

pa.set_cpu_count(1)
datasets.config.TORCH_ARROW_USE_64_BIT_OFFSETS = True

ROOT = str(Path(__file__).resolve().parents[1])
CUR = str(Path(__file__).resolve().parents[0])
sys.path.append(ROOT)
sys.path.append(CUR)

def truncate_content(content: str, max_length: int = 10000) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
                content[: max_length // 2]
                + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
                + content[-max_length // 2:]
        )

def process_train(output_path):
    data_path = os.path.join(ROOT, 'data', 'raw', 'multimodal-open-r1-8k-verified')

    dataset = datasets.load_dataset(data_path)
    train_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    def make_map_fn(split):
        def process_fn(example, idx):

            question_raw = example.pop('problem')
            question_img = example.pop('image')

            # path = os.path.join(image_save_dir, '{:04d}.png'.format(idx))
            # question_img.save(path)
            # image_path = path.replace(ROOT + "/", '')

            question_raw = truncate_content(question_raw)

            question = question_raw + '\n' + instruction_following

            solution_raw = example.pop('solution')
            print(solution_raw)
            # print(image_path)
            print('-' * 100)

            pattern = r"<answer>(.+)</answer>"
            answer = re.findall(pattern, solution_raw)
            answer = answer[0] if answer else ""

            data = {
                "data_source": "multimodal_math",
                "prompt": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": question_img},
                        {"type": "text", "text": question},
                    ],
                }],
                "ability": "multimodal_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset.to_parquet(os.path.join(output_path, 'train.parquet'))

def process_test(output_path):
    data_path = os.path.join(ROOT, 'data', 'raw', 'MMMU')

    names = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    datas = []

    for name in names:
        dataset = datasets.load_dataset(data_path, name)
        datas.append(dataset['validation'])

    test_dataset = concatenate_datasets(datas)

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    def make_map_fn(split):
        def process_fn(example, idx):

            question_raw = example.pop('question')
            question_img = example.pop('image_1')
            options_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            options = eval(example.pop('options'))
            options_string = ""
            if len(options) > 0:
                for i, option in enumerate(options):
                    options_string += f"{options_label[i]}. {option}\n"
            else:
                options_string = ""
            question_raw = question_raw + '\n' + options_string if options_string else question_raw
            question_raw = truncate_content(question_raw)
            question = question_raw + '\n' + instruction_following

            solution_raw = example.pop('explanation')

            answer = example.pop('answer')

            print(options, answer)
            print('-' * 100)

            data = {
                "data_source": "multimodal_math",
                "prompt": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": question_img},
                        {"type": "text", "text": question},
                    ],
                }],
                "ability": "multimodal_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(output_path, 'test.parquet'))

if __name__ == '__main__':
    output_path = os.path.join(ROOT, 'data', 'processed', 'multimodal_math')
    os.makedirs(output_path, exist_ok=True)

    process_train(output_path)
    process_test(output_path)