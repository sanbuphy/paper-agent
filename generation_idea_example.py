from src.generate_idea import generate_ideas, check_idea_novelty
import pprint

model = 'ernie-4.0-turbo-8k'

# generate ideas
# 会自动保存 ideas 的结果到文件中，下次运行时会直接从文件中加载
ideas = generate_ideas(
    base_dir="./generation_idea_template/small_object_attention/",
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
    model=model,
)

# check novelty
# 会自动更新 novelty 的结果到 ideas.json 文件中，下次运行时会直接从文件中加载
novelty_ideas = check_idea_novelty(
    ideas=ideas,
    base_dir="./generation_idea_template/small_object_attention/",
    model=model,
)

pprint.pp(novelty_ideas)

