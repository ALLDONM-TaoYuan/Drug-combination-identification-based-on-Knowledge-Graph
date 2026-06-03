import pandas as pd
from pykeen.triples import TriplesFactory
import os


def load_and_combine_triples(file_paths):
    dfs = []
    for path in file_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件 {path} 不存在")

        # 读取文件，强制所有列为字符串类型
        df = pd.read_csv(
            path,
            header=None,
            dtype=str,
            low_memory=False
        )

        # 去除可能的空白字符
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

        # 检查列数
        if df.shape[1] != 3:
            raise ValueError(f"文件 {path} 列数不正确，应为3列（头实体，关系，尾实体），实际为 {df.shape[1]}列")

        dfs.append(df)

    # 合并DataFrame并去重
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df.drop_duplicates()


def save_knowledge_graph(triples_factory, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # 保存三元组为CSV文件
    output_path = os.path.join(output_dir, "all_triples.csv")
    pd.DataFrame(triples_factory.triples).to_csv(
        output_path,
        index=False,
        header=False
    )

    # 保存实体和关系映射
    entity_mapping_path = os.path.join(output_dir, "entity_id_map.csv")
    pd.DataFrame(list(triples_factory.entity_to_id.items()),
                 columns=['entity', 'id']).to_csv(entity_mapping_path, index=False)

    relation_mapping_path = os.path.join(output_dir, "relation_id_map.csv")
    pd.DataFrame(list(triples_factory.relation_to_id.items()),
                 columns=['relation', 'id']).to_csv(relation_mapping_path, index=False)


def main():
    input_files = [
        "../Database/DDI.csv",
        "../Database/DTI.csv",
    ]
    output_directory = "./"

    try:
        print("开始加载和合并三元组文件...")
        combined_df = load_and_combine_triples(input_files)
        print(f"成功加载 {len(combined_df)} 条三元组")

        print("创建TriplesFactory实例...")
        triples_factory = TriplesFactory.from_labeled_triples(
            triples=combined_df.values,
            create_inverse_triples=False
        )

        print(f"实体数量: {triples_factory.num_entities}")
        print(f"关系数量: {triples_factory.num_relations}")
        print(f"三元组数量: {triples_factory.num_triples}")

        print("保存知识图谱文件...")
        save_knowledge_graph(triples_factory, output_directory)
        print(f"知识图谱已保存至 {output_directory} 目录")

    except Exception as e:
        print(f"错误: {str(e)}")

        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
