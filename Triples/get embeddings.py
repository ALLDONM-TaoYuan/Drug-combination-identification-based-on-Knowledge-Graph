import numpy as np
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import torch
import os

def load_and_split_triples(file_path, split_ratio=(0.8, 0.1, 0.1), random_state=42):
    """加载CSV格式的三元组数据并进行分割"""
    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)
    # 创建三元组工厂
    tf = TriplesFactory.from_labeled_triples(
        triples=df.iloc[:, :3].values,
        create_inverse_triples=False,
    )
    # 分割数据集
    train_tf, test_tf, valid_tf = tf.split(
        ratios=split_ratio,
        random_state=random_state,
    )

    return train_tf, test_tf, valid_tf, tf  # 返回原始TriplesFactory对象

def save_embeddings_transe(model, tf, directory="embeddings"):

    model.cpu()
    torch.cuda.empty_cache()

    # 创建输出目录
    os.makedirs(directory, exist_ok=True)

    # 提取实体嵌入
    entity_rep = model.entity_representations[0]
    with torch.no_grad():
        entity_emb = entity_rep(indices=None).detach().cpu().numpy()

    # 提取关系嵌入
    relation_rep = model.relation_representations[0]
    relation_emb = relation_rep(indices=None).detach().cpu().numpy()

    # 保存嵌入向量
    np.save(os.path.join(directory, "entity_emb.npy"), entity_emb)
    np.save(os.path.join(directory, "relation_emb.npy"), relation_emb)
    print(f"实体嵌入已保存至: {directory}/entity_emb.npy")
    print(f"关系嵌入已保存至: {directory}/relation_emb.npy")

    # 获取实体和关系映射
    entity_id_to_label = tf.entity_id_to_label
    relation_id_to_label = tf.relation_id_to_label
    mapped_triples = tf.mapped_triples

    # 实体映射
    entity_mapping = [(i, label) for i, label in entity_id_to_label.items()]
    df_entity = pd.DataFrame(entity_mapping, columns=['id', 'entity'])
    df_entity.to_csv(os.path.join(directory, 'entity_mapping.csv'), index=False)
    print(f"实体映射已保存至: {directory}/entity_mapping.csv")

    # 关系映射
    relation_mapping = [(i, label) for i, label in relation_id_to_label.items()]
    df_relation = pd.DataFrame(relation_mapping, columns=['id', 'relation'])
    df_relation.to_csv(os.path.join(directory, 'relation_mapping.csv'), index=False)
    print(f"关系映射已保存至: {directory}/relation_mapping.csv")

    # 保存映射后的三元组
    np.save(os.path.join(directory, "mapped_triples.npy"), mapped_triples)
    print(f"映射三元组已保存至: {directory}/mapped_triples.npy")


def main():
    # 配置路径
    triples_file = "./Triples/KG/all_triples.csv"
    output_dir = "embeddings"

    # 加载并分割三元组数据
    print("加载并分割三元组数据...")
    train_tf, test_tf, valid_tf, original_tf = load_and_split_triples(
        file_path=triples_file,
        split_ratio=(0.8, 0.1, 0.1),
        random_state=42,
    )
    # 训练TransE模型
    print("开始训练TransE模型...")
    results = pipeline(
        training=train_tf,
        testing=test_tf,
        validation=valid_tf,
        model="TransE",
        model_kwargs={"embedding_dim": 128},
        training_loop="SLCWA",
        training_kwargs={"num_epochs": 50, "batch_size": 256},
        loss="SoftplusLoss",
        loss_kwargs={"reduction": "mean"},
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.001},
        negative_sampler_kwargs={"num_negs_per_pos": 10},
        random_seed=42,
        evaluator="RankBasedEvaluator",
        evaluator_kwargs={"filtered": True},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 保存嵌入向量 - 使用原始的TriplesFactory对象
    print("保存嵌入向量...")
    save_embeddings_transe(results.model, original_tf, directory=output_dir)

    print("Complete!")

if __name__ == "__main__":
    main()
