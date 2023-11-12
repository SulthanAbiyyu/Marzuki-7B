import os
import datasets
import pandas as pd
import numpy as np
from pathlib import Path

class DatasetPreprocessor:
    def __init__(self, dir: str, out_dir: str = "./data"):
        self.dir = dir
        self.out_dir = out_dir
        self.kata_tanya = [
            "Jelaskan tentang ",
            "Apa yang dimaksud dengan ",
            "Apa itu ",
            "Bantu saya menjelaskan tentang ",
            "Apa yang bisa kamu jelaskan tentang ",
            "Bagaimana "
        ]
        
    def get_alpaca_id(self, repo_id: str = "Ichsan2895/alpaca-gpt4-indonesian"):
        alpaca_id = datasets.load_dataset(repo_id)
        
        if not os.path.exists(f"{self.dir}/alpaca"):
            os.makedirs(f"{self.dir}/alpaca")
        
        for split, dataset in alpaca_id.items():
            dataset.to_json(f"{self.dir}/alpaca/alpaca-id-{split}.jsonl")

    def get_oasst_id(self, repo_id: str = "Ichsan2895/OASST_Top1_Indonesian"):
        alpaca_id = datasets.load_dataset(repo_id)
        
        if not os.path.exists(f"{self.dir}/oasst"):
            os.makedirs(f"{self.dir}/oasst")
        
        for split, dataset in alpaca_id.items():
            dataset.to_json(f"{self.dir}/oasst/oasst-id-{split}.jsonl")

    def get_wikipedia_id(self, repo_id: str = "wikipedia", language: str = "id", date: str = "20231001"):
        wiki_id = datasets.load_dataset(
            repo_id, 
            language=language,
            date=date,
            beam_runner='DirectRunner')
        
        if not os.path.exists(f"{self.dir}/wikipedia"):
            os.makedirs(f"{self.dir}/wikipedia")

        for split, dataset in wiki_id.items():
            dataset.to_json(f"{self.dir}/wikipedia/wikipedia-id-{split}.jsonl")

    def get_idk_mrc(self, repo_id: str = "rifkiaputri/idk-mrc"):
        dataset = datasets.load_dataset(repo_id)
        train = dataset['train']['qas']
        validation = dataset['validation']['qas']
        test = dataset['test']['qas']
        all_data = train + validation + test
        all_context = dataset['train']['context'] + dataset['validation']['context'] + dataset['test']['context']
        possible_questions = []
        possible_context = []
        possible_answers = []
        for i, data in enumerate(all_data):
            for d in data:
                if d["is_impossible"] == False:
                    possible_questions.append(d["question"])
                    possible_context.append(all_context[i])
                    possible_answers.append(d["answers"][0]["text"])
                    
        idk_mrc = pd.DataFrame({
            "question": possible_questions,
            "context": possible_context,
            "answer": possible_answers
        })
        
        if not os.path.exists(f"{self.dir}/idk-mrc"):
            os.makedirs(f"{self.dir}/idk-mrc")

        idk_mrc.to_json(f"{self.dir}/idk-mrc/idk-mrc.jsonl", orient="records", lines=True)

    def preprocess(self):
        all_path = [os.getcwd() + "/" + f for f in os.listdir('./data')]
        
        dfs = []
        for path in all_path:
            if "wikipedia" in path:
                data = pd.read_json(path, lines=True)[["title", "text"]]
                data.columns = ['instruction', 'output']
                data['instruction'] = data['instruction'].apply(
                    lambda x: np.random.choice(self.kata_tanya, 1)[0] + x
                )
            elif "alpaca" in path:
                data = pd.read_json(path, lines=True)[['input', 'output']]
                data.columns = ['instruction', 'output']
            elif "idk_mrc" in path:
                data = pd.read_json(path, lines=True)
                data.columns = ['instruction', 'context', 'output']
                data["instruction"] = "Konteks:\n" + data["context"] + \
                                            "Pertanyaan:\n"  + data["instruction"]
                data = data[['instruction', 'output']]
            elif "oasst" in path:
                data = pd.read_json(path, lines=True)[["instruction_id", "output_id"]]
                data.columns = ['instruction', 'output']
            else:
                raise NotImplementedError("Dataset not found")
            dfs.append(data)
        
        merged_data = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
        merged_data = merged_data.sample(frac=1).reset_index(drop=True)
        merged_data.to_json(f'./{self.out_dir}/kedungkandang.jsonl', orient='records', lines=True)
