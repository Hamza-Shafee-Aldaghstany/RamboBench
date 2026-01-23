from transformers import AutoTokenizer
from utils.utils import Tools
from extract_repo_elements import ExtractRepoElements
from vector_repo_elements import VectorRepoElements
from utils.vector_utils import BagOfWordsEmbedding, UniXcoderEmbedding
from EEI_build import SketchPromptBuilder
from RUE_build import RUEPromptBuilder

repo_base_dir = './repositories/rambo/rambo'
parsed_repo_base_dir = './parsed_repositories/parsed_project/bamboo'
repo_list = './repositories/rambo/rambo/repo_names.txt'
# repo_base_dir = 'repositories/rambo'
# parsed_repo_base_dir = 'parsed_repositories/rambo'
# repo_list = 'repositories/rambo/repo_names.txt'

repos = open(repo_list, 'r').read().split('\n')

vectorizer = BagOfWordsEmbedding()
# vectorizer = UniXcoderEmbedding()
# extractor = ExtractRepoElements('rambo', repo_base_dir, parsed_repo_base_dir, repos)
# extractor.extract_elements()
# VectorRepoElements('rambo', vectorizer, repos).vector_elements()

# tasks = Tools.load_jsonl('datasets/rambo_2k_context_with_type_paramters.jsonl')
model_id = 'deepseek-ai/deepseek-coder-1.3b-base'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, )
# sketchbuilder = SketchPromptBuilder('rambo', 'bow', repos, tasks, tokenizer)
# sketchbuilder.build_prompt('prompts/rambo_2k_context_with_type_paramters_sketch.jsonl')

tasks = Tools.load_jsonl('/home/user/Desktop/Code_Bench/exploded_predictions_ramb.jsonl')
RUEPromptBuilder(benchmark='rambo', reranker_type="bow",repo_base_dir=repo_base_dir, repos=repos, tasks=tasks, tokenizer=tokenizer).build_prompt('prompts/rambo_prompts_code-1.3b-base-RUE.jsonl')